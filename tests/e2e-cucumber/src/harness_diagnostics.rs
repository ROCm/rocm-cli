// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Make the harness's own death legible instead of silent.
//!
//! The self-hosted WSL2 lane fails by stopping mid-suite with nothing to go on: the
//! last line is a passing step, then `cargo`'s `error: test failed` with no
//! `Caused by:` — which narrows it to exit code 101, an unwinding panic — and no
//! panic message anywhere in the job log. `report.json` and `junit.xml` are left at
//! zero bytes and `report.html` is never written, so the artifact says nothing
//! either. Every guess at the cause costs a full CI round trip on a scarce runner.
//!
//! **Why the message is missing.** cucumber's runner replaces the panic hook with
//! an empty one for the whole run and restores it afterwards
//! (`cucumber-0.23.0/src/runner/basic.rs:929`, restored at `:1070`) — deliberately,
//! so a step's panic is reported by the writer rather than printed twice. A panic
//! *raised by the writer itself* escapes that path: it unwinds out of `run()` past
//! the restore, with the silencing hook still installed. Nothing prints, and the
//! exit code is all that is left. That is not specific to any lane; it is why every
//! failure of this shape has been unreadable.
//!
//! So this does not install a hook — one would be taken away again on the next
//! line. [`run_or_record`] catches the panic where it escapes, at the `run()`
//! boundary, and records it two ways: to stderr, which the silenced hook would have
//! written, and to a log inside the results directory the lane already uploads,
//! which survives even if stderr is the thing that broke. The panic is then
//! re-raised, so the process still dies with 101 and nothing downstream has to
//! learn a new signal.
//!
//! The same log records the state of the standard streams at start-up and again at
//! the panic, so a descriptor that changed underneath the process is visible rather
//! than inferred. This only observes the streams; it does not modify them.
//!
//! **What this does not recover.** Only the panic *message* survives: the
//! `file:line` and any backtrace are handed to the panic hook, not carried in the
//! unwind payload, and the hook installed at that moment is cucumber's empty one.
//! So a one-line entry with no location is everything there is, not a truncation.
//! And only a panic that unwinds out of `run()` reaches this boundary — one inside
//! a fixture server's task or thread (`http_server::spawn` drops its
//! `JoinHandle`) is contained there, and while the empty hook is installed it
//! prints nothing either. The step that needed that server then fails on a
//! symptom — a refused connection, a timeout — with no trace of the cause.
//!
//! The whole boundary lives in the library target rather than inline in the
//! `harness = false` cucumber test binary, so its logic gets real `#[test]`
//! coverage — the custom harness never executes plain `#[test]` functions placed
//! inside it. Same reason as [`crate::panic_capture`].

use std::future::Future;
use std::io::Write as _;
use std::path::Path;
use std::pin::Pin;

use futures::FutureExt as _;

/// Name of the log inside the results directory. Picked up by the lane's existing
/// `upload-artifact` of the whole directory, so nothing in CI needs to change.
pub const LOG_NAME: &str = "harness-diagnostics.log";

/// Describe the standard streams: their `O_NONBLOCK` state and what they point at.
///
/// Read-only. Best-effort and infallible — a description that cannot be obtained is
/// reported as such, because this runs on a path where failing would destroy the
/// very diagnostic it exists to produce.
#[must_use]
pub fn describe_std_streams() -> Vec<String> {
    #[cfg(unix)]
    {
        use std::os::fd::{AsFd as _, AsRawFd as _, BorrowedFd};

        #[allow(unsafe_code)] // libc FFI
        fn flags(fd: BorrowedFd<'_>) -> String {
            // SAFETY: `fd` is a valid borrowed descriptor; `F_GETFL` takes no
            // pointer arguments.
            let raw = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
            if raw < 0 {
                return format!("flags unreadable ({})", std::io::Error::last_os_error());
            }
            let blocking = if raw & libc::O_NONBLOCK == 0 {
                "blocking"
            } else {
                "NON-BLOCKING"
            };
            format!("{blocking} (F_GETFL=0o{raw:o})")
        }

        // `/proc/self/fd/N` names the open file the descriptor currently refers to
        // ("pipe:[12345]", a tty, a path). Two snapshots naming different targets
        // is the signature of a descriptor being replaced underneath the process.
        // Linux-only despite the `cfg(unix)` above: elsewhere there is no `/proc`,
        // so this half reads `target unknown (..)` and only the flags are reported.
        fn target(n: i32) -> String {
            std::fs::read_link(format!("/proc/self/fd/{n}")).map_or_else(
                |e| format!("target unknown ({e})"),
                |p| p.display().to_string(),
            )
        }

        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let stderr = std::io::stderr();
        vec![
            format!("stdin  -> {} [{}]", target(0), flags(stdin.as_fd())),
            format!("stdout -> {} [{}]", target(1), flags(stdout.as_fd())),
            format!("stderr -> {} [{}]", target(2), flags(stderr.as_fd())),
        ]
    }
    #[cfg(not(unix))]
    {
        vec!["stream description is POSIX-only; not collected on this platform".to_owned()]
    }
}

/// Append a section to the diagnostics log, creating it if needed.
///
/// Silently gives up on any I/O error: a diagnostic must never be the reason a run
/// fails, and there is nowhere left to report the failure of the reporting channel.
fn append(dir: &Path, heading: &str, lines: &[String]) {
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(LOG_NAME))
    else {
        return;
    };
    let _ = writeln!(file, "=== {heading} ===");
    for line in lines {
        let _ = writeln!(file, "{line}");
    }
    let _ = writeln!(file);
    let _ = file.flush();
}

/// Record the run's starting conditions.
///
/// Written unconditionally — not only when something looks wrong. A successful run
/// leaves the baseline the next failing run is read against, and "checked, nothing
/// to report" is a different fact from "never ran".
pub fn record_startup(dir: &Path) {
    append(dir, "harness start", &describe_std_streams());
}

/// Record a panic that escaped the cucumber run, to the log and to stderr.
///
/// Both destinations on purpose: stderr is what a reader of the job log expects and
/// is what the silenced hook would have produced, while the log survives a stderr
/// that is not reaching anyone. `writeln!` rather than `eprintln!` so a failing
/// stderr yields a missing line rather than a second panic on the way out.
pub fn record_fatal_panic(dir: &Path, message: &str) {
    record_fatal_panic_to(dir, message, &mut std::io::stderr());
}

/// [`record_fatal_panic`] with its second destination injected, so the text written
/// there can be asserted on.
///
/// Split out purely for that: with `std::io::stderr()` hard-coded, deleting the
/// write left every test in this module green, which is the one destination the
/// lane this exists for actually reads. The pointer to the log is the part a reader
/// needs and the part most likely to rot, so it is what the test pins.
fn record_fatal_panic_to(dir: &Path, message: &str, out: &mut dyn std::io::Write) {
    let mut lines = vec![format!("message: {message}")];
    lines.extend(describe_std_streams());
    append(dir, "fatal panic (escaped the cucumber run)", &lines);

    let _ = writeln!(
        out,
        "E2E suite aborted by a panic inside the cucumber run: {message}\n\
         (cucumber silences the panic hook while running, so this would otherwise \
         print nothing; see {LOG_NAME} in the results artifact)"
    );
}

/// Drive `fut` to completion, recording any panic that escapes it before re-raising.
///
/// This is the boundary described in the module docs: the last point at which the
/// payload of a writer panic still exists, because cucumber's silencing hook is
/// still installed as it unwinds past the restore.
///
/// The panic is **re-raised, never absorbed** — `resume_unwind` diverges, so the
/// process still dies with exit 101 exactly as it did before. A diagnostic that
/// swallowed a failure would turn a red run green, which is strictly worse than the
/// silence it replaces.
///
/// `std::panic::catch_unwind` cannot span an `.await`, hence `FutureExt`.
///
/// Returns a boxed future rather than being an `async fn`: the cucumber run it
/// wraps is enormous, and an `async fn` would store it inline and hand the caller a
/// future of the same size (`clippy::large_futures`). Allocating it once here keeps
/// that detail out of every call site.
pub fn run_or_record<'a, F>(dir: &'a Path, fut: F) -> Pin<Box<dyn Future<Output = F::Output> + 'a>>
where
    F: Future + 'a,
{
    Box::pin(async move {
        let guarded = Box::pin(std::panic::AssertUnwindSafe(fut).catch_unwind());
        match guarded.await {
            Ok(output) => output,
            Err(payload) => {
                record_fatal_panic(dir, &crate::panic_capture::panic_message(&payload));
                std::panic::resume_unwind(payload);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stream snapshot the module promises at *both* ends of the log.
    ///
    /// Asserted per-section rather than over the whole file: `record_startup` also
    /// emits one, so a whole-log `contains` would still pass if the panic-time
    /// snapshot were dropped — which is the headline claim (comparing the two is
    /// how a descriptor replaced mid-run becomes visible).
    fn assert_has_stream_snapshot(section: &str, which: &str, log: &str) {
        #[cfg(unix)]
        let (needle, what) = ("stdout ->", "a stream line");
        #[cfg(not(unix))]
        let (needle, what) = ("POSIX-only", "the POSIX-only placeholder");
        assert!(
            section.contains(needle),
            "the {which} section carries no stream snapshot ({what} expected):\n{log}"
        );
    }

    /// Start-up state and a later fatal panic have to land in the same file, in
    /// order, so one read of the artifact shows what changed between them.
    #[test]
    fn the_startup_baseline_and_a_later_panic_share_one_log() {
        let dir = tempfile::tempdir().expect("failed to create a temp dir");
        record_startup(dir.path());
        record_fatal_panic(dir.path(), "failed to write into terminal: boom");

        let log = std::fs::read_to_string(dir.path().join(LOG_NAME)).expect("no log written");
        let start = log.find("=== harness start ===").expect("no start section");
        let panic_at = log.find("=== fatal panic").expect("no panic section");
        assert!(start < panic_at, "sections out of order:\n{log}");
        assert!(
            log.contains("message: failed to write into terminal: boom"),
            "{log}"
        );

        let (startup_section, panic_section) = log.split_at(panic_at);
        assert_has_stream_snapshot(startup_section, "start-up", &log);
        assert_has_stream_snapshot(panic_section, "panic", &log);
    }

    /// Every stream is named even when nothing is wrong, so a later reader can tell
    /// "checked, was fine" from "never collected" — and where the flags are POSIX
    /// only, the log has to say that rather than go quiet, which reads the same as
    /// a collection that never ran.
    #[test]
    fn the_stream_description_accounts_for_every_standard_stream() {
        let described = describe_std_streams().join("\n");
        assert!(!described.is_empty(), "the description must never be empty");

        #[cfg(unix)]
        for stream in ["stdin", "stdout", "stderr"] {
            assert!(
                described.contains(stream),
                "{stream} missing from: {described}"
            );
        }
        #[cfg(not(unix))]
        assert!(
            described.contains("POSIX-only"),
            "a platform without these flags must say so, got: {described}"
        );
    }

    /// A destination that cannot be written to must not take the run down with it —
    /// this runs on the path that is already handling a fatal panic.
    #[test]
    fn an_unwritable_destination_is_survived_silently() {
        let dir = tempfile::tempdir().expect("failed to create a temp dir");
        let missing = dir.path().join("does").join("not").join("exist");
        record_startup(&missing);
        record_fatal_panic(&missing, "boom");
        assert!(!missing.exists(), "the helper must not create the tree");

        // Prove those calls were real work that the unwritable path swallowed, and
        // not a no-op: without this, an `append` that wrote nothing anywhere would
        // satisfy the assertion above just as well.
        record_startup(dir.path());
        assert!(
            dir.path().join(LOG_NAME).exists(),
            "a writable destination must still be written"
        );
    }

    /// The boundary's whole purpose: the message survives, and the failure still
    /// fails. Absorbing the panic would turn a red run green.
    #[tokio::test]
    async fn a_panic_escaping_the_run_is_recorded_and_re_raised() {
        let dir = tempfile::tempdir().expect("failed to create a temp dir");

        // The future yields a value on its non-panicking path, as the real one
        // does. Without that, the block's type is `!`, which makes "swallow the
        // panic and return something" a compile error rather than a test failure —
        // and an assertion that no reachable mutation can trip is not a guard.
        let escaped = std::panic::AssertUnwindSafe(run_or_record(dir.path(), async {
            assert!(
                !std::hint::black_box(true),
                "failed to write into terminal: injected"
            );
            "a summary that is never produced"
        }))
        .catch_unwind()
        .await;

        assert!(
            escaped.is_err(),
            "the panic must propagate, not be absorbed into a normal return"
        );
        let log = std::fs::read_to_string(dir.path().join(LOG_NAME)).expect("no log written");
        assert!(
            log.contains("message: failed to write into terminal: injected"),
            "the escaping panic's message must reach the log:\n{log}"
        );
    }

    /// The second destination is the one the lane actually reads.
    ///
    /// stderr is what reaches the job log, and the suite lane this boundary exists
    /// for publishes no artifact at all — so the log is unreachable there and this
    /// line is the whole diagnostic. Left unasserted, deleting the write kept every
    /// other test in this module green, which would quietly restore exactly the
    /// silence the change removes.
    ///
    /// Pins the pointer to the log as well as the message: a reader who gets this
    /// line still needs to be told where the fuller record is, and that is the half
    /// most likely to rot.
    #[test]
    fn the_panic_is_announced_on_the_second_destination_too() {
        let dir = tempfile::tempdir().expect("failed to create a temp dir");
        let mut out = Vec::new();

        record_fatal_panic_to(dir.path(), "failed to write into terminal: boom", &mut out);

        let written = String::from_utf8(out).expect("the announcement must be UTF-8");
        assert!(
            written.contains("failed to write into terminal: boom"),
            "the panic message must reach the second destination, got: {written}"
        );
        assert!(
            written.contains(LOG_NAME),
            "the announcement must point at the fuller record, got: {written}"
        );
    }

    /// A run that completes normally passes straight through — the boundary must
    /// not alter the value or leave a panic section behind on a healthy run.
    #[tokio::test]
    async fn a_run_that_does_not_panic_is_passed_through_untouched() {
        let dir = tempfile::tempdir().expect("failed to create a temp dir");

        let summary = run_or_record(dir.path(), async { "the real summary" }).await;

        assert_eq!(summary, "the real summary");
        let log = std::fs::read_to_string(dir.path().join(LOG_NAME)).unwrap_or_default();
        assert!(
            !log.contains("fatal panic"),
            "a healthy run must not record a panic:\n{log}"
        );
    }
}
