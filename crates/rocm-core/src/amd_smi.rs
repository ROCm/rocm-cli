// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Locating and launching the `amd-smi` binary.
//!
//! Location prefers the binary shipped in the newest managed ROCm SDK runtime
//! recorded in the registry, then falls back to the home-directory venv and
//! legacy locations, then to the bare `amd-smi` name for a `PATH` lookup.
//!
//! Launching goes through [`amd_smi_json`], the one supported way for a
//! synchronous caller to start the binary: it runs the `/dev/kfd` pre-flight in
//! `kfd_readable` first, and that pre-flight protects nothing unless every
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

/// Device node of the AMD kernel fusion driver. Bare-metal Linux only. WSL
/// compiles as `target_os = "linux"` but reaches the GPU through `/dev/dxg` and
/// the Windows host driver, and has no `/dev/kfd` at all — see `docs/wsl.md`
/// and [`crate::examine::WslFacts`].
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
/// A path seam rather than the existing [`crate::hardware_root`] re-rooting,
/// because the two cannot do the same job here: [`kfd_readable`] keeps the real
/// node's veto on purpose, so a simulated root can only ever make the gate
/// decline, never allow. That is the invariant stopping a test hook from
/// launching `amd-smi` on a machine with no GPU — and it also means no
/// simulated root can reach the *allowed* branch on the GPU-less lane that runs
/// per PR.
///
/// An override, not a bypass: it relocates the probe and does not skip it. With
/// the feature on and the variable unset — every unit test here, and every E2E
/// scenario that does not set it — the gate is `/dev/kfd` exactly as in
/// production.
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
/// # What the predicate is
///
/// "This process can reach the KFD", not "`amd-smi` would hang". The two are
/// different sets, and declining the difference is the intended trade:
///
/// * An open that **never returns** is the hazard itself — `amd-smi` against
///   such a device blocks in uninterruptible kernel sleep, where no signal
///   escapes, so neither Ctrl-C nor `SIGKILL` nor any caller's timeout reaches
///   it. [`KFD_OPEN_TIMEOUT`] bounds the probe and reads the overrun as "not
///   readable", so the gate declines instead of inheriting the stall.
/// * An open that **fails with an errno** — `ENOENT` for an absent node,
///   `EACCES` outside the render group, whatever a driver mid-reset returns —
///   cannot park anything by itself. The gate declines these too.
///
/// The second class is broader than the hazard on purpose. Narrowing it means
/// deciding per errno which kernel states are safe to launch into, and that
/// list cannot be checked against a resetting or wedged device by any test this
/// repo can run; a gate that declined only on the timeout would be assuming
/// every errno it had not anticipated is harmless. Not resting on an assumption
/// of that shape is the whole reason the open is bounded in the first place
/// (see [`KFD_OPEN_TIMEOUT`]). A refusal is recoverable and names what it
/// refused; a hang is neither.
///
/// # What this does not prove
///
/// It is a **mitigation heuristic, not a proof**. All this establishes is that
/// `open("/dev/kfd")` returned inside [`KFD_OPEN_TIMEOUT`]. The hang it exists
/// to prevent is attributed to `amd-smi`'s *later* ioctls against a wedged
/// device, so a device that opens and then wedges on the next ioctl is outside
/// what the gate can see, and the 2s bound is chosen rather than measured. No
/// run against a genuinely unhealthy driver backs any of this up: the lane that
/// runs these tests has no `amd-smi` and no GPU, and the FIFO fixtures model a
/// blocking `open`, the closest reachable analogue rather than the real
/// failure. What the gate buys is that the hazard's common precondition — a
/// device the kernel will not hand out — refuses the spawn, instead of betting
/// on a timeout that cannot reach a process in uninterruptible sleep.
///
/// # What the broad refusal costs
///
/// On a host outside the render group `amd-smi` might well have answered — much
/// of its inventory comes from sysfs — so `gpu_policy::detect_gpu_count` now
/// returns `None` there and, unlike `gpu_vram_usage`, has no sysfs fallback of
/// its own. That is a real narrowing. It is not a loss of GPU detection: on the
/// hosts this branch can fire on, `serve` resolves ordinals from
/// [`crate::usable_amd_gpu_indices`], which counts KFD topology nodes under
/// `/sys/class/kfd` and AMD DRM cards under `/sys/class/drm` — both
/// world-readable, neither needing the device node open — and
/// `validate_pinned_gpu_index` prefers that set outright. What is lost is the
/// secondary count `--gpu auto` ranks with, so on a *multi*-GPU host in that
/// state auto-selection returns no selection and the engine's own device probe
/// picks. `rocm diagnose` already names the host, as `fix-4-render-group`.
///
/// # Platforms with no KFD
///
/// Off Linux the check is `true` without probing. Load-bearing rather than
/// pedantic: `amd-smi.exe` is a supported Windows binary (see
/// [`crate::managed_runtime::managed_sdk_tool_path`]), so reading an absent
/// `/dev/kfd` as "unsafe" would silently disable GPU detection on every Windows
/// host.
///
/// WSL needs the same wave-through and cannot get it from `cfg`: it compiles as
/// Linux yet has no `/dev/kfd` (see [`KFD_DEVICE`]), so the real open fails on
/// every WSL host. WSL identity alone is not the condition, though — `/dev/dxg`
/// is vendor-agnostic, and a WSL host whose ROCDXG plumbing is not ready is one
/// where `amd-smi` still hangs, which a bare "is this WSL" test cannot tell
/// apart from a healthy one. The condition is therefore
/// `is_wsl_host() && has_usable_amd_gpu()`, the same conjunction
/// `apps/rocm/src/dash.rs::gpu_reachable_for_preflight` feeds the dashboard, so
/// `serve`, `rocmd` and the dashboard keep one cross-surface verdict. On bare
/// metal `is_wsl_host()` short-circuits and the behaviour is a plain
/// `/dev/kfd` open.
///
/// # Hardware-probe routing
///
/// Like the dashboard pre-flight, this reads `host_path("/dev/kfd")` *and*
/// keeps the real node's veto (see [`crate::hardware_root`]). A simulated E2E
/// root can make the gate decline on a machine that has a GPU, but can never
/// make it allow a launch on a machine that does not.
#[cfg(target_os = "linux")]
fn kfd_readable(device: &Path) -> bool {
    kfd_readable_with(
        // Second argument is a closure, not a value: extracting this condition
        // into a named function turned a short-circuit into two eager
        // evaluations, so `has_usable_amd_gpu` was running on every Linux
        // launch rather than on WSL only. Passing the probe unevaluated keeps
        // the original behaviour and still lets the condition be tested.
        wsl_wave_through(is_wsl_host(), has_usable_amd_gpu),
        device,
        &host_path(device),
    )
}

/// The WSL wave-through's own condition: a WSL host **and** a reachable GPU.
///
/// Extracted so it can be pinned by a truth table. The tests below drive
/// [`kfd_readable_with`] with literal booleans, which says nothing about how
/// the caller computes that argument — `&&` could become `||`, or
/// `has_usable_amd_gpu()` could be dropped entirely, and every one of them
/// would stay green on a GPU-less lane. The GPU probe is taken unevaluated so
/// the short-circuit survives the extraction: on bare metal `is_wsl` is false
/// and `has_usable_amd_gpu` must not run at all. Both conjuncts are
/// load-bearing in
/// opposite directions: without `is_wsl_host` a bare-metal host with a GPU
/// would skip the device check the gate exists for, and without
/// `has_usable_amd_gpu` an unready WSL host would be waved through to an
/// `amd-smi` that cannot work.
#[cfg(target_os = "linux")]
fn wsl_wave_through(is_wsl: bool, has_usable_gpu: impl FnOnce() -> bool) -> bool {
    is_wsl && has_usable_gpu()
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
/// A plain blocking `open` would make the gate's own safety rest on an
/// assumption — that `kfd_open` always returns promptly, so only `amd-smi`'s
/// later ioctls can enter D-state — which cannot be checked against a
/// reset-in-progress device by any test here. Were it wrong, the gate would
/// have moved the hazard into a synchronous call with *less* protection than
/// the spawn it replaced, there being no child to kill.
///
/// Bounding the open removes the need to be right about it: an open that has
/// not returned in this long reads as "not readable", the declining answer, so
/// the unverifiable case fails safe. Short, because a healthy `kfd_open` takes
/// microseconds and a longer wait only buys a longer stall on a host that is
/// already unhealthy.
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
/// `open` until the kernel lets it go, which against the wedged device this
/// gate exists to refuse may be never. `rocmd`'s telemetry watcher and `serve`'s
/// probes both poll, so without a cap a host stuck in that state accumulates one
/// parked thread per poll until the process runs out of its thread budget.
///
/// Small on purpose: a larger budget only buys more parked threads on a host
/// already answering "not readable". A healthy `kfd_open` returns in
/// microseconds, so one slot would serve every caller this crate has; 8 leaves
/// headroom for concurrent ones without approaching a thread limit.
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
/// cheaper than a blocked caller. The opened handle is dropped on the probe
/// thread, so a late success closes its own descriptor rather than leaking one.
///
/// # Why both failure branches decline
///
/// A probe that cannot claim a `budget` slot, and one whose thread cannot be
/// spawned at all, both answer `false` rather than falling back to a direct
/// `open` on the calling thread. Neither failure is independent of the state
/// this gate refuses: the way threads run out *here* is one parked `open` per
/// poll against a wedged device, so such a fallback would fire exactly when the
/// device is most dangerous and perform the unbounded blocking `open` the bound
/// exists to prevent — on the caller's own thread, with nothing left to kill.
///
/// Refusing instead is recoverable and visible: [`amd_smi_json`] says it
/// declined and names the device, GPU-aware callers degrade to service state
/// alone, and the next poll on a recovered host finds its slots freed and
/// allows the launch again. So both branches give the same `false` the timeout
/// gives, for the same reason.
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

/// A read probe of `device` that cannot park the caller.
///
/// The same bounded open this module uses for its own `/dev/kfd` pre-flight,
/// exposed for `rocm-dash-collectors`, which gates the dashboard's GPU-metrics
/// `amd-smi` launch on the same device. Shared rather than restated there so
/// the bound, the budget and the reasoning on `device_readable_within_budget`
/// stay in one place; a second copy would be free to drift from all three.
///
/// Off Linux this is a direct open, matching [`kfd_readable`]'s wave-through:
/// there is no `/dev/kfd` to wedge, and the bound would buy nothing.
#[cfg(target_os = "linux")]
pub fn device_readable_bounded(device: &Path) -> bool {
    device_readable(device)
}

/// Off-Linux counterpart of [`device_readable_bounded`]: a direct open, with no
/// bound and no wave-through.
///
/// It does **not** mirror [`kfd_readable`]'s off-Linux wave-through, and an
/// earlier revision that made it do so was wrong. The two answer different
/// questions. `kfd_readable` gates *this crate's* `amd-smi` launch, which off
/// Linux has no `/dev/kfd` to reason about, so waving through is right there.
/// This function is the shared probe `rocm-dash-collectors` calls for its own
/// pre-flight, and that pre-flight is expected to decline a missing device on
/// every platform — `missing_kfd_and_no_reachability_fails_preflight` and
/// `real_kfd_keeps_a_veto_over_the_host_root_view` assert exactly that, and a
/// wave-through turns both green while removing the veto they exist to protect.
///
/// The bound is Linux-only because the hazard is: `/dev/kfd` is what can park a
/// process in uninterruptible sleep. An ordinary file open elsewhere returns or
/// errors.
#[cfg(not(target_os = "linux"))]
pub fn device_readable_bounded(device: &Path) -> bool {
    std::fs::OpenOptions::new().read(true).open(device).is_ok()
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
/// The supported way to launch `amd-smi` from a synchronous caller. The
/// pre-flight has to happen at every launch site or it protects nothing, and
/// routing every launch through one function is what keeps that true as sites
/// are added.
///
/// `timeout` is defence in depth, not the mitigation: [`crate::process::run_with_timeout`]
/// kills the timed-out child and waits on it, and neither dislodges a process
/// in uninterruptible sleep — nor does the `wait` on one return. It bounds the
/// ordinary failures (a slow or wedged-but-signalable `amd-smi`); the
/// pre-flight is what keeps the D-state case away.
///
/// # Latency
///
/// `timeout` is not the worst case. The pre-flight runs first and is bounded
/// separately by [`KFD_OPEN_TIMEOUT`] (2s on Linux), so the worst case is
/// `KFD_OPEN_TIMEOUT + timeout` — 12s for a 10s `timeout`, 4s for the 2s one
/// `rocmd`'s watcher passes. A caller documenting its own worst case has to add
/// the pre-flight's bound to the one it passes here.
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
             permission denied, no KFD GPU node, or a reset in progress); the pre-flight \
             refuses an unreachable device because amd-smi can block unkillably in \
             uninterruptible sleep against one whose open never returns",
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
        // generic "no output". `ExitStatus`'s own `Display` already reads
        // "exit status: 1" (and "exit code: 1" on Windows), so prefixing it
        // again produced "exit status exit status: 1".
        bail!(
            "amd-smi {} failed: {}",
            args.join(" "),
            crate::probe_exec::truncate_to_chars(
                crate::command_failure_detail_or_else(&output, || output.status.to_string()),
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
    /// it. Same lock `fix.rs` takes, for the same reason; without the feature it
    /// is uncontended and costs nothing.
    #[cfg(target_os = "linux")]
    fn host_root_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::hardware_root::HOST_ROOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The pre-flight's bound and budget are quoted as literal numbers in
    /// several places that cannot see this constant, and in at least two of
    /// them the quoted figure is *derived* — the caller's own timeout plus this
    /// one. Changing [`KFD_OPEN_TIMEOUT`] without touching those leaves prose
    /// that confidently states the wrong worst case, and nothing else in the
    /// tree notices. This test is the tripwire: it fails on the change, and its
    /// message is the list of what to update.
    ///
    /// Not a claim that 2s and 8 are the right values — both are judgement
    /// calls, argued where they are declared. It only makes changing them cost
    /// a deliberate pass over every restatement.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_probe_bound_and_budget_are_restated_in_these_places() {
        assert_eq!(
            KFD_OPEN_TIMEOUT,
            Duration::from_secs(2),
            "the pre-flight's open bound changed. Figures derived from it are restated in: \
             this module's `KFD_OPEN_TIMEOUT` and `amd_smi_json` docs; \
             `apps/rocm/src/gpu_policy.rs::AMD_SMI_SERVE_PROBE_TIMEOUT` (one probe 2+10=12s, \
             two sequential probes ~24s, `--model` readiness ~12s); \
             `apps/rocmd/src/host_snapshot.rs::AMD_SMI_PROBE_TIMEOUT` (2+2=4s); \
             `crates/rocm-dash-collectors/src/amd_smi.rs`'s \
             `the_device_preflight_does_not_run_on_the_async_worker` doc; and \
             the `amd-smi KFD pre-flight` section of `docs/testing.md`. \
             Update every one of them, then update this assertion."
        );
        assert_eq!(
            MAX_INFLIGHT_KFD_PROBES, 8,
            "the in-flight probe budget changed; it is restated in this module's own docs and \
             in the `amd-smi KFD pre-flight` section of `docs/testing.md`"
        );
    }

    /// The gate has to stop the *spawn*, not merely report a failure after it:
    /// the launch that already happened is the launch that hangs. The fake
    /// `amd-smi` leaves a marker file, so its absence is direct evidence
    /// nothing ran — and running the same fixture against a readable device
    /// proves the fake would otherwise have executed, which is what stops the
    /// blocked case passing vacuously.
    ///
    /// Linux-only: off Linux `kfd_readable` is deliberately a no-op, so there
    /// is no gate to exercise.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_kfd_gate_stops_amd_smi_before_it_is_spawned() -> Result<()> {
        // WSL is waved through deliberately, so the blocked case cannot hold
        // there.
        if is_wsl_host() {
            skip_notice("this is a WSL host, which the KFD gate waves through by design");
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

    /// WSL has no `/dev/kfd` either, but unlike Windows cannot be waved through
    /// by `cfg`: it compiles as Linux, so the real open runs, always fails, and
    /// would decline every routed `amd-smi` launch on a supported platform —
    /// silently, since both `serve` probes drop the error with `.ok()?`.
    ///
    /// The *conjunction* is what is pinned, not WSL identity: a WSL host whose
    /// ROCDXG plumbing is not ready is one where `amd-smi` still hangs, so a
    /// bare identity check would wave it through. The unready-WSL case
    /// distinguishes the two; the bare-metal case is the control that stops the
    /// test passing with the gate deleted.
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

    /// The conjunction `kfd_readable` hands to the wave-through, as a truth
    /// table.
    ///
    /// The test above drives `kfd_readable_with` with literal booleans, so it
    /// cannot see how that argument is produced: `&&` to `||`, or dropping
    /// `has_usable_amd_gpu()`, leaves it green on a GPU-less lane. Only the
    /// `(true, false)` and `(false, true)` rows here fail for those mutations.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_wsl_wave_through_needs_both_a_wsl_host_and_a_reachable_gpu() {
        assert!(
            wsl_wave_through(true, || true),
            "ready WSL host waves through"
        );
        assert!(
            !wsl_wave_through(true, || false),
            "a WSL host with no reachable GPU must not wave through: amd-smi \
             cannot work there, and `||` would allow it"
        );
        assert!(
            !wsl_wave_through(false, || true),
            "bare metal must reach the device check even with a usable GPU, or \
             the gate this whole module exists for is skipped"
        );
        assert!(!wsl_wave_through(false, || false));
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

    /// The gate's own probe must not be able to hang: an unbounded `open` would
    /// move the D-state hazard into a synchronous call with *less* protection
    /// than the spawn it replaced — no child to kill, no timeout.
    ///
    /// A FIFO makes the blocking case real rather than simulated:
    /// `open(O_RDONLY)` on one with no writer blocks in the kernel
    /// indefinitely, the observable shape of the hazard. The deadline assertion
    /// is what distinguishes a bounded probe from an unbounded one — without
    /// it, deleting the timeout would hang the suite instead of failing here,
    /// and a hang is not a readable failure. The readable control stops the test
    /// passing vacuously against a probe mutated to return `false` always.
    ///
    /// Uses a budget of its own rather than [`KFD_PROBE_BUDGET`]: it parks a
    /// probe thread for the lifetime of the test binary, and spending a
    /// production slot would quietly shrink the budget for every later test in
    /// this process.
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
        // Watched on its own thread with a much looser deadline rather than
        // called inline: a mutant dropping the bound would block forever on this
        // FIFO, and called inline that is a CI timeout rather than a failure
        // naming its cause.
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
    /// The escalation: against a wedged device every poll leaves a thread
    /// parked in `open`, so an uncapped probe accumulates them until the process
    /// runs out of threads — at which point a fallback to a direct blocking open
    /// would reinstate exactly the hang the gate exists to refuse.
    ///
    /// Non-vacuous by construction. The capped probe is pointed at an ordinary
    /// readable file, so an uncapped implementation — or one whose exhausted
    /// branch fell back to a blocking open — answers `true`; only a probe that
    /// refuses on an exhausted budget answers `false`. The control re-runs the
    /// same device against a budget with a free slot and must get `true`, so a
    /// mutant returning `false` always fails too.
    ///
    /// Uses budgets of its own, limit one: the parked FIFO probe spends the
    /// slot and never frees it, so sharing the production budget would silently
    /// shrink it for every other test in this process.
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

    /// A probe that returns must give its slot back — eventually.
    ///
    /// Nothing else asserts this. The budget test above only shows that an
    /// exhausted budget declines; a `ProbeSlot::drop` that forgot to decrement
    /// would satisfy it too, and would then decline every probe after the first
    /// eight for the life of the process — the gate silently becoming "no GPU,
    /// ever".
    ///
    /// "Eventually" is the honest invariant, and writing this test is what
    /// showed why. The slot is released when the probe *thread* drops it, not
    /// when the caller returns: `device_readable_within_budget` returns as soon
    /// as `recv_timeout` has the answer, and the thread may not have unwound
    /// yet. A first draft reused a one-slot budget back-to-back and failed on
    /// the second call for exactly that reason — a real race, not a flake. The
    /// bound below is therefore a poll, and the budget is sized so the test
    /// cannot depend on the timing it just discovered.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_completed_probe_frees_its_slot() -> Result<()> {
        static REUSED: ProbeBudget = ProbeBudget::new(1);

        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-kfd-slot-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let readable = temp_root.join("readable");
        fs::write(&readable, b"")?;

        assert!(
            device_readable_within_budget(&readable, KFD_OPEN_TIMEOUT, &REUSED),
            "a readable file must probe true"
        );

        // Poll rather than assert immediately: release happens on the probe
        // thread's unwind. A `drop` that never decrements never reaches zero,
        // which is the mutation this exists to catch.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while REUSED.in_flight.load(Ordering::Acquire) != 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "the slot was never released: in_flight stayed at {}",
                REUSED.in_flight.load(Ordering::Acquire)
            );
            thread::sleep(Duration::from_millis(10));
        }

        // The budget is genuinely reusable once released, which is the point.
        assert!(
            device_readable_within_budget(&readable, KFD_OPEN_TIMEOUT, &REUSED),
            "the one slot must serve a second probe after the first released"
        );

        let _ = fs::remove_dir_all(&temp_root);
        Ok(())
    }

    /// A FIFO under `root`, or `None` on a host with no `mkfifo(1)`.
    ///
    /// Shelled out rather than calling `libc::mkfifo`, because the workspace
    /// lints deny `unsafe_code`. A missing coreutils is not evidence about the
    /// probe, so the caller skips rather than fails — but it says so on stderr
    /// (see [`skip_notice`]), because a silent skip here is a green result for
    /// a test that checked nothing.
    #[cfg(target_os = "linux")]
    fn make_fifo(root: &Path, name: &str) -> Option<PathBuf> {
        let fifo = root.join(name);
        let made = Command::new("mkfifo")
            .arg(&fifo)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if matches!(&made, Ok(status) if status.success()) {
            return Some(fifo);
        }
        skip_notice("no usable mkfifo(1), so no device whose open really blocks can be built here");
        None
    }

    /// Announce that a test gave up without asserting its subject.
    ///
    /// Every conditional skip in this module goes through here. The tests that
    /// have one are the ones that need a host feature (`mkfifo(1)`) or a host
    /// *shape* (bare-metal Linux rather than WSL) to have anything to assert;
    /// where that is missing they return `Ok(())`, which is indistinguishable
    /// from a pass unless something says otherwise. `cargo test` shows this on
    /// a failing run and under `--nocapture`; matches the `skip:` form already
    /// used in `rocm-dash-daemon`'s server and end-to-end tests.
    #[cfg(target_os = "linux")]
    fn skip_notice(reason: &str) {
        eprintln!("skip: {reason}; this assertion did NOT run");
    }

    /// Pins the production wiring the injected helpers cannot: that
    /// [`amd_smi_json`] gates on [`KFD_DEVICE`] itself.
    ///
    /// Every other gate test reaches `amd_smi_json_with` with a device of its
    /// own choosing, so swapping the constant out of the public wrapper — which
    /// disables the gate for every production caller — leaves them all green.
    /// Comparing the wrapper's observable verdict against the gate read
    /// directly means the two can only agree while the wrapper passes the real
    /// device through.
    ///
    /// Non-vacuous on *every* host in an `e2e-test-hooks` build: the empty
    /// simulated root has no `dev/kfd`, so the gate declines whatever the
    /// machine has, while a wrapper passing any other path — `/` being the
    /// obvious mutant — re-roots to the root directory itself, which opens.
    /// Without the feature it is non-vacuous on any host with no readable
    /// `/dev/kfd` (every non-GPU CI runner).
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
    /// The seam's whole justification is that it cannot weaken a shipped build,
    /// and both halves are load-bearing: a `preflight_kfd_device` reading the
    /// variable unconditionally would fail the unset half and hand any process
    /// on the host a way to disable the gate, and one returning a readable path
    /// outright — the bypass this deliberately is not — would fail the
    /// declining half.
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
            // The gate waves WSL through, so the fake amd-smi would run and
            // this would stop testing the quote bound rather than the gate.
            skip_notice("this is a WSL host, where the gate does not reach the launch path");
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
