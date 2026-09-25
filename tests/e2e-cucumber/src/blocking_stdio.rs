// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Give the suite's own stdout/stderr the blocking semantics `std::io` assumes.
//!
//! `O_NONBLOCK` is a property of the *open file description*, so it arrives with
//! whatever stdio the parent handed us — nothing in this process sets it. Under
//! `wsl.exe` the guest's std streams are relayed over descriptors that are
//! non-blocking, and a synchronous writer has no answer for that: once the log
//! consumer stalls and the pipe fills, the next write returns `EAGAIN` instead of
//! waiting, and `write_all` surfaces it as `ErrorKind::WouldBlock`.
//!
//! That is fatal here. cucumber's `writer::Basic` turns any write error into
//! `panic!("failed to write into terminal: {e}")` on the main thread, so a
//! momentarily full pipe aborts the whole run mid-scenario. Worse, where the relay
//! merges the two streams the panic hook's own write hits the same congestion and
//! is dropped, leaving no reason behind: no summary, no reconciliation line, and a
//! `report.json`/`junit.xml` pair created but never written.
//!
//! Clearing the flag restores the POSIX default the rest of the stack is written
//! against: a full pipe makes the write wait for the reader rather than fail. It
//! also puts the diagnostics back, since a panic from any source can then be
//! printed.
//!
//! The trade is deliberate: a consumer that stalls *permanently* now hangs the run
//! until the job's own timeout instead of aborting it in seconds. That is the same
//! exposure `cargo`, `bash`, and every other process in the tree already carry, and
//! a wrong answer in seconds is worth less than a right one late.
//!
//! Lives in the library target (not in the `harness = false` cucumber test binary)
//! so its logic gets real `#[test]` coverage, for the same reason as
//! [`crate::panic_capture`].

#[cfg(unix)]
mod imp {
    use std::io::{self, Write as _};
    use std::os::fd::{AsFd as _, AsRawFd as _, BorrowedFd};

    /// Clear `O_NONBLOCK` on `fd`, reporting whether it had actually been set.
    ///
    /// # Errors
    ///
    /// Returns the underlying `fcntl` error if the flags cannot be read or written.
    #[allow(unsafe_code)] // libc FFI
    pub fn clear_nonblocking(fd: BorrowedFd<'_>) -> io::Result<bool> {
        let raw = fd.as_raw_fd();
        // SAFETY: `raw` comes from a `BorrowedFd`, so it is open and stays valid for
        // the duration of the call. `F_GETFL`/`F_SETFL` take no pointer arguments.
        let flags = unsafe { libc::fcntl(raw, libc::F_GETFL) };
        if flags < 0 {
            return Err(io::Error::last_os_error());
        }
        if flags & libc::O_NONBLOCK == 0 {
            return Ok(false);
        }
        // SAFETY: as above; `flags` is the value just read back from this same
        // descriptor, with one bit cleared.
        if unsafe { libc::fcntl(raw, libc::F_SETFL, flags & !libc::O_NONBLOCK) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(true)
    }

    /// Clear `O_NONBLOCK` on stdout and stderr, then report what was changed.
    ///
    /// Best-effort: a stream that cannot be adjusted is noted and the run
    /// continues, because refusing to start would be a worse outcome than the
    /// intermittent write failure this guards against. Both streams are adjusted
    /// before anything is printed — the note below is itself a write to stderr, and
    /// on a congested non-blocking stream that write is exactly what would be lost.
    pub fn restore_blocking_stdio() {
        let stdout = io::stdout();
        let stderr = io::stderr();
        let results = [
            ("stdout", clear_nonblocking(stdout.as_fd())),
            ("stderr", clear_nonblocking(stderr.as_fd())),
        ];
        for (name, result) in results {
            // `writeln!`, not `eprintln!`: this runs before we know the stream is
            // writable, and a diagnostic must not be the thing that panics.
            match result {
                // Worth saying out loud rather than fixing silently: it names the
                // hosts whose relayed stdio arrives non-blocking, which is not
                // otherwise visible from a job log.
                Ok(true) => {
                    let _ = writeln!(
                        io::stderr(),
                        "E2E: {name} arrived non-blocking (O_NONBLOCK); cleared it so a full \
                         pipe makes writes wait for the reader instead of failing the run"
                    );
                }
                Ok(false) => {}
                Err(e) => {
                    let _ = writeln!(
                        io::stderr(),
                        "E2E: could not clear O_NONBLOCK on {name} ({e}); a stalled log \
                         consumer may abort this run"
                    );
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        use std::time::Duration;

        #[allow(unsafe_code)] // libc FFI
        fn flags_of(fd: BorrowedFd<'_>) -> i32 {
            // SAFETY: `fd` is a valid borrowed descriptor; `F_GETFL` takes no
            // pointer arguments.
            let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
            assert!(flags >= 0, "F_GETFL failed: {}", io::Error::last_os_error());
            flags
        }

        #[allow(unsafe_code)] // libc FFI
        fn set_nonblocking(fd: BorrowedFd<'_>) {
            let flags = flags_of(fd);
            // SAFETY: as in `flags_of`; `flags` was just read from this descriptor.
            let rc =
                unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) };
            assert!(rc >= 0, "F_SETFL failed: {}", io::Error::last_os_error());
        }

        /// The exact failure mode the self-hosted WSL2 lane hit: a payload larger
        /// than the pipe buffer, written to a non-blocking descriptor whose reader
        /// has not caught up, fails instead of waiting — and after the flag is
        /// cleared the identical write completes.
        #[test]
        fn a_congested_nonblocking_pipe_fails_writes_until_the_flag_is_cleared() {
            let (reader, writer) = io::pipe().expect("failed to create a pipe");
            set_nonblocking(writer.as_fd());

            // Comfortably past any platform's default pipe capacity, so the write
            // cannot be absorbed by the buffer alone.
            let payload = vec![b'x'; 1 << 20];
            let err = (&writer)
                .write_all(&payload)
                .expect_err("a non-blocking pipe nobody is draining must refuse a 1 MiB write");
            assert_eq!(
                err.kind(),
                io::ErrorKind::WouldBlock,
                "expected the EAGAIN that aborts the suite, got: {err}"
            );

            assert!(
                clear_nonblocking(writer.as_fd()).expect("failed to clear O_NONBLOCK"),
                "the flag was set, so clearing it must report a change"
            );

            // Drain only after a pause, so the write genuinely has to wait rather
            // than finding room already available.
            let drain = std::thread::spawn(move || {
                let mut reader = reader;
                std::thread::sleep(Duration::from_millis(50));
                io::copy(&mut reader, &mut io::sink()).expect("failed to drain the pipe")
            });

            (&writer)
                .write_all(&payload)
                .expect("a blocking pipe must wait for the reader, not fail");
            drop(writer);

            let drained = drain.join().expect("the draining thread panicked");
            assert!(
                drained >= payload.len() as u64,
                "the reader saw {drained} bytes, fewer than the {} written",
                payload.len()
            );
        }

        /// The common case — every other lane, and every local run — must be left
        /// exactly as it was found.
        #[test]
        fn an_already_blocking_stream_is_reported_unchanged_and_left_alone() {
            let (_reader, writer) = io::pipe().expect("failed to create a pipe");
            let before = flags_of(writer.as_fd());

            assert!(
                !clear_nonblocking(writer.as_fd()).expect("failed to inspect the pipe"),
                "a blocking descriptor must report that nothing was changed"
            );
            assert_eq!(
                flags_of(writer.as_fd()),
                before,
                "the descriptor's flags must be untouched"
            );
        }
    }
}

#[cfg(not(unix))]
mod imp {
    /// No-op: `O_NONBLOCK` is a POSIX descriptor flag with no Windows equivalent,
    /// and the Windows lanes have never shown this failure.
    ///
    /// `const` only to satisfy `clippy::missing_const_for_fn` on an empty body —
    /// CI lints on Linux, so nothing would have caught it here. Drop the `const`
    /// if this ever grows a body.
    pub const fn restore_blocking_stdio() {}
}

pub use imp::restore_blocking_stdio;
