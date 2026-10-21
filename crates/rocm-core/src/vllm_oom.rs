// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The vLLM out-of-memory log-tail reader shared by the engine's post-failure
//! hint and the `rocm serve` summary.
//!
//! Splits a captured engine log tail into candidate lines, scores each with the
//! very rule `rocm diagnose`'s vLLM-OOM checker uses (through
//! [`crate::diagnose::vllm_oom_symptom_is_diagnosable`]), selects the user's own
//! proximate failing line, and renders it for display. Also owns the two text
//! markers — [`VLLM_OOM_DETECTED_PREFIX`] and [`VLLM_OOM_DIAGNOSE_COMMAND`] —
//! that `rocm-engine-vllm` and `apps/rocm` must agree on byte for byte, and that
//! the selector keys on to avoid quoting rocm-cli's own hint back at the user.

use crate::{diagnose, terminal};

/// Lead-in of the vLLM engine's own post-failure out-of-memory hint.
///
/// Public because two crates have to agree on it byte for byte:
/// `rocm-engine-vllm` renders its hint with this prefix, and
/// [`vllm_oom_diagnose_symptom`] recognises it in order *not* to quote
/// rocm-cli's own sentence back at the user as though the engine had printed
/// it. See [`is_rocm_authored_log_line`].
pub const VLLM_OOM_DETECTED_PREFIX: &str = "Detected an out-of-memory failure (";

/// The `rocm diagnose` invocation the OOM surfaces print, as it appears in text.
///
/// Shared for the same reason as [`VLLM_OOM_DETECTED_PREFIX`]: it is how a line
/// rocm-cli authored is told apart from a line the engine printed.
pub const VLLM_OOM_DIAGNOSE_COMMAND: &str = "rocm diagnose --symptom";

/// Whether this log segment is text *rocm-cli itself* wrote into the engine log,
/// rather than output the engine produced.
///
/// A managed launch shares one log between the supervisor and the engine: when
/// vLLM OOMs, the engine bails with its startup-log context — which already
/// carries this crate's own OOM hint — and the CLI then writes that whole error
/// back into the same file. The hint quotes the user's failing line and names
/// the allocator keywords, so it scores as an OOM just as strongly as the real
/// line does, and it lands *after* it. A newest-first scan would therefore
/// always select rocm-cli's own sentence, which contains `'` and backticks, so
/// [`quotable_in_single_quotes`](terminal::quotable_in_single_quotes) would reject it and the printed command would
/// fall back to [`VLLM_OOM_CANONICAL_SYMPTOM`](diagnose::VLLM_OOM_CANONICAL_SYMPTOM) every single time in real use —
/// exactly the "quote the user's real line" promise the note is built on.
///
/// Matching on rocm-cli's own wording is deliberate, rather than truncating the
/// tail at the engine's `Error:` block: the error block also *echoes* the log
/// tail verbatim, and those echoed lines are the user's real output and remain
/// legitimate choices. Only the sentences this crate composed are excluded.
/// vLLM never prints either marker.
///
/// Pinned by `the_engine_hint_is_never_quoted_back_as_the_users_own_line`.
fn is_rocm_authored_log_line(segment: &str) -> bool {
    segment.contains(VLLM_OOM_DETECTED_PREFIX) || segment.contains(VLLM_OOM_DIAGNOSE_COMMAND)
}

/// Whether one log segment carries a genuine out-of-memory failure, scored with
/// the *same* rule as the `rocm diagnose` vLLM-OOM checker (`check_16_vllm_oom`,
/// reached through [`vllm_oom_symptom_is_diagnosable`](diagnose::vllm_oom_symptom_is_diagnosable)) so the engine hint, the
/// post-failure serve summary and `rocm diagnose` never disagree on what counts
/// as a vLLM OOM. The segment is vLLM's own process output, so the checker's
/// required vLLM anchor is supplied by context: it is scored as
/// `vllm: <segment>` and clears the checker's `MIN_SCORE_FOR_MATCH` or not.
///
/// This deliberately rejects a bare `out of memory` with no allocator-shaped
/// signature — a kernel OOM-killer line, a dependency's log, or an unrelated
/// subprocess — the same false positive the checker's threshold guards against,
/// rather than driving the user-facing memory note off any stray OOM substring
/// in the tail. A real vLLM/HIP allocation failure (`HIP out of memory`,
/// `CUDA out of memory`, `hipErrorOutOfMemory`, or `torch.OutOfMemoryError`
/// corroborated by an allocator message) still clears it.
///
/// Scoring each [`terminal::rendered_lines`] row of the segment, rather than the
/// segment whole, is what actually keeps this in step with the checker.
/// [`vllm_anchored_lines`] re-splits the string it is handed with
/// `rendered_lines` and keeps only the rows carrying the `vllm` anchor — and
/// `rendered_lines` ends a row on far more than the `\n`/`\r` that
/// [`vllm_log_segments`] splits on: every non-`SGR` CSI (`ESC[2K`, `ESC[1A`),
/// `ESC E`/`ESC D`, `VT`, `FF`, `U+0085` and `U+2028`/`U+2029`. So a real
/// repaint such as `Loading weights: 90%\r\x1b[2Ktorch.OutOfMemoryError: ...`
/// becomes the segment `\x1b[2Ktorch.OutOfMemoryError: ...`, which the checker
/// renders as `["vllm: ", "torch.OutOfMemoryError: ..."]`. Only the *empty*
/// anchored row would be scored, the tail would read as "no OOM", and both the
/// engine hint and the serve-summary note would silently vanish for the shape of
/// failure they exist to explain. Anchoring each rendered row instead closes
/// that gap for the whole break set at once, rather than for `\r` alone.
///
/// This costs nothing on the quoting side, and provably so: a segment that
/// `rendered_lines` splits into more than one row must contain a control
/// character or an `ESC`, and [`quotable_in_single_quotes`](terminal::quotable_in_single_quotes) rejects every one of
/// those. So a multi-row segment can only ever reach the printed command as the
/// canonical fallback — it can never be stripped into a runnable lookalike of
/// the user's error. The raw segment, control bytes intact, is still what
/// [`vllm_oom_diagnose_symptom`] returns, so that guard keeps working.
fn segment_shows_oom(segment: &str) -> bool {
    terminal::rendered_lines(segment)
        .iter()
        .any(|row| diagnose::vllm_oom_symptom_is_diagnosable(&format!("vllm: {row}")))
}

/// The log tail split into candidate symptom lines.
///
/// Splits on `\r` as well as `\n` because vLLM draws its progress bar with bare
/// carriage returns, so a real failure arrives as
/// `Loading weights: 90%\rtorch.OutOfMemoryError: ...` on a single `\n` line and
/// the exception has to become a candidate in its own right.
///
/// Deliberately a raw split and *not* [`terminal::rendered_lines`], even though
/// that is what the diagnostic re-splits with. `rendered_lines` also discards
/// escape sequences and other control bytes, and the segment returned here
/// becomes the `--symptom` argument: sanitising it means
/// [`quotable_in_single_quotes`](terminal::quotable_in_single_quotes) only ever sees clean text, never rejects it,
/// and the canonical-symptom fallback stops firing — so a colourised OOM line
/// would be stripped into a lookalike and printed rather than refused. Keeping
/// the raw bytes here and doing the *scoring* on rendered rows inside
/// [`segment_shows_oom`] is what lets both properties hold at once.
///
/// The bound on that guarantee is worth stating, because the obvious reading of
/// it is wrong. It is a property of *this read of this text*: whatever text a
/// caller hands in, a colourised OOM line inside it is refused and the canonical
/// symptom printed. It is **not** an end-to-end property of a managed launch,
/// and the reason is the engine's own bail. `startup_log_context` echoes the log
/// tail as [`terminal::rendered_lines`] rows — it has to, since that text is
/// printed to a terminal — and the supervisor appends that whole bail to the
/// same shared log. The echo is then the *only* copy of the failing line inside
/// the CLI's tail window: measured on a 200-line log, the engine's raw
/// colourised line ends up 84 lines from the end and the window is
/// `DEFAULT_LOG_TAIL_LINES` = 80. So the serve summary, which reads after the
/// bail, selects the rendered row and quotes it instead of falling back.
///
/// That outcome is sound — the row is text the terminal really drew, it carries
/// no control bytes or quotes, and quotable still implies diagnosable — but it
/// does mean the engine and the CLI can print *different* `--symptom` arguments
/// for the same colourised failure: the engine the canonical symptom, the CLI
/// the user's rendered line. Filtering the echo out to force agreement is not an
/// option: measured, it leaves no OOM line in the window at all and the serve
/// note disappears for every real managed launch. Pinned, both halves, by
/// `the_colourised_round_trip_quotes_the_row_the_terminal_drew`.
fn vllm_log_segments(log: &str) -> impl Iterator<Item = String> + '_ {
    log.lines()
        .flat_map(|line| line.split('\r'))
        .map(|segment| segment.trim().to_owned())
        .filter(|segment| !segment.is_empty())
}

/// The `rocm diagnose --symptom` string to route a vLLM OOM log tail to, or
/// `None` when the tail carries no OOM line to route.
///
/// Returns the user's actual failing *raw* line (so `diagnose` echoes their real
/// error), selected with [`segment_shows_oom`], which is the checker's own rule.
/// Shared by the vLLM engine's post-failure hint and the `rocm` CLI serve summary
/// so both surfaces apply one selection rule to the text they are given, instead
/// of hand-rolling it twice. It is also the OOM *gate* for both: `None` is the
/// "this tail is not an OOM" answer, so no caller needs a separate predicate
/// first.
///
/// Sharing the rule is not the same as the two surfaces always printing the same
/// command, and the difference is real rather than theoretical: they read the
/// shared log at different points, and the engine's bail puts a sanitized echo of
/// the tail into it in between. See [`vllm_log_segments`] for the measurement and
/// for why closing the gap would cost the serve note entirely.
///
/// The exact guarantee is narrower than "whatever comes back is diagnosable",
/// and the difference is load-bearing. What holds is:
///
/// > if [`quotable_in_single_quotes`](terminal::quotable_in_single_quotes) accepts the returned value, then
/// > [`vllm_oom_symptom_is_diagnosable`](diagnose::vllm_oom_symptom_is_diagnosable) accepts it too.
///
/// That is what the callers need, because quotability is what they gate the
/// printed command on, and it is exactly enough: a returned line that is *not*
/// quotable is replaced by [`VLLM_OOM_CANONICAL_SYMPTOM`](diagnose::VLLM_OOM_CANONICAL_SYMPTOM), which is diagnosable,
/// so the printed command always reports a cause either way.
///
/// The stronger claim is false, and necessarily so. A segment carrying an
/// escape — `\x1b[2Ktorch.OutOfMemoryError: ...` from a progress-bar repaint —
/// is scored by [`segment_shows_oom`] on its rendered rows and is a genuine OOM,
/// but handed to the checker *whole* it splits into `["vllm: ", "torch..."]`,
/// leaving the anchor on the empty row and scoring 0. Returning the rendered row
/// instead would make the claim true and reintroduce the hole it exists to
/// close: the stripped row is quotable, so a colourised line would be printed as
/// a runnable lookalike of the user's error rather than refused. The raw bytes
/// are kept, the quoting guard does its job, and the fallback covers the rest.
/// Pinned by `every_rendered_line_break_keeps_the_gate_in_step_with_the_scorer`.
///
/// Lines rocm-cli wrote into the log itself are skipped — see
/// [`is_rocm_authored_log_line`] for why the newest-first scan would otherwise
/// always land on this crate's own hint instead of the user's error.
///
/// Scanning with `.rev()` is deliberate, not incidental: when several lines of
/// the tail mention running out of memory, the *last* one is the proximate
/// failure — the point where the process actually gave up — and the earlier ones
/// are usually the allocator's own retry chatter leading up to it. The cost is
/// real and worth stating: on a tail like "torch.OutOfMemoryError: ... Tried to
/// allocate 7.21 GiB. GPU 0 has a total capacity of 24.00 GiB." followed by a
/// bare "RuntimeError: ... killed: out of memory", the vaguer line wins even
/// though the first carries the allocation size and the capacity. Both are
/// diagnosable, so this picks which line is quoted, never whether one is. The
/// user still sees every line: callers print the whole log tail beside the hint.
///
/// The return value is *log text*, not a shell-safe token: some callers only
/// display it. A caller that renders it inside a quoted command must gate it on
/// [`quotable_in_single_quotes`](terminal::quotable_in_single_quotes) first and fall back to
/// [`VLLM_OOM_CANONICAL_SYMPTOM`](diagnose::VLLM_OOM_CANONICAL_SYMPTOM) otherwise. Guaranteeing quotability here
/// instead would impose the command-builder's constraint on the display-only
/// callers, silently withholding the user's real error from text that is never
/// pasted anywhere.
#[must_use]
pub fn vllm_oom_diagnose_symptom(log_tail: &str) -> Option<String> {
    vllm_log_segments(log_tail)
        .filter(|segment| !is_rocm_authored_log_line(segment))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .find(|segment| segment_shows_oom(segment))
        .map(|segment| format!("vllm: {segment}"))
}

/// The text to *show* the user for a segment [`vllm_oom_diagnose_symptom`]
/// selected, with the `vllm: ` anchor already removed by the caller.
///
/// Not the same operation as stripping the segment's control bytes, and the
/// difference is the whole point. [`strip_terminal_control_sequences`](terminal::strip_terminal_control_sequences) deletes
/// every break token — `\r`, `ESC[1A`, `NEL`, `VT`, `U+2028`… — *without a
/// separator*, so a progress-bar repaint such as
/// `Loading weights: 90%\x1b[1Atorch.OutOfMemoryError: HIP out of memory` comes
/// back as the single run-together line
/// `Loading weights: 90%torch.OutOfMemoryError: HIP out of memory`. That string
/// is text vLLM never printed: two rows the terminal drew separately, welded
/// into one. Displaying it beside "Detected an out-of-memory failure" presents a
/// fabricated line as the user's own output, and — once rocm-cli writes its bail
/// back into the shared log — a *later* read finds that welded line, scores it
/// as an OOM, finds it quotable (the control bytes are gone) and quotes it into
/// the `rocm diagnose --symptom` command the user is invited to paste.
///
/// So this returns the [`terminal::rendered_lines`] row that actually scored —
/// the same per-row rule [`segment_shows_oom`] gates on — which is one row the
/// terminal really drew. When no row scores (the caller passed a segment that is
/// not an OOM at all) it falls back to stripping, which is lossless for the
/// single-row case that is by far the common one.
///
/// When more than one row scores, the **last** one is taken, for the same reason
/// [`vllm_oom_diagnose_symptom`] scans its segments with `.rev()`: the newest
/// line is the proximate failure and the earlier ones are the allocator's retry
/// chatter leading up to it. Picking the first instead would make the display
/// disagree with the selector about which line "the failure" is, inside the one
/// message that prints both.
///
/// Pinned by `the_displayed_oom_line_is_a_row_the_terminal_actually_drew`.
#[must_use]
pub fn vllm_oom_display_line(segment: &str) -> String {
    terminal::rendered_lines(segment)
        .into_iter()
        .rev()
        .find(|row| diagnose::vllm_oom_symptom_is_diagnosable(&format!("vllm: {row}")))
        .unwrap_or_else(|| terminal::strip_terminal_control_sequences(segment))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VLLM_GPU_MEMORY_UTILIZATION_HINT, VLLM_OOM_CANONICAL_SYMPTOM};

    /// The OOM gate both production surfaces actually use. There is no separate
    /// classifier predicate: `vllm_oom_diagnose_symptom(..).is_some()` *is* the
    /// gate, so these threshold assertions are made against the function that
    /// ships rather than against a parallel one no caller reaches.
    fn log_shows_oom(log: &str) -> bool {
        vllm_oom_diagnose_symptom(log).is_some()
    }

    #[test]
    fn vllm_oom_gate_matches_allocator_signatures_but_not_generic_failures() {
        // A real allocator message (the `MIN_SCORE_FOR_MATCH`-clearing shape the
        // `rocm diagnose` vLLM-OOM checker recognises) is a genuine OOM, whichever
        // module path the exception carries.
        assert!(log_shows_oom(
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB."
        ));
        assert!(log_shows_oom(
            "torch.cuda.OutOfMemoryError: CUDA out of memory"
        ));
        assert!(log_shows_oom("RuntimeError: hipErrorOutOfMemory"));
        // Case-insensitive.
        assert!(log_shows_oom("HIP OUT OF MEMORY"));

        // Sub-threshold on their own — the same lines the diagnose checker keeps
        // below `MIN_SCORE_FOR_MATCH` so a stray OOM phrase does not carry the
        // verdict. Detecting these here would contradict that checker (they are
        // the false positives the shared threshold exists to reject).
        //
        // The bare exception class every PyTorch OOM raises, with no allocator
        // message to corroborate it.
        assert!(!log_shows_oom("TORCH.OUTOFMEMORYERROR"));
        // A spaced "out of memory" with no allocator anchor phrase.
        assert!(!log_shows_oom("HIP error: out of memory"));
        // A kernel OOM-killer line from an unrelated subprocess must NOT drive
        // the vLLM memory note.
        assert!(!log_shows_oom(
            "Out of memory: Killed process 4242 (python)"
        ));
        // The generic EngineCore wrapper is NOT an OOM signature.
        assert!(!log_shows_oom(
            "EngineCore failed: engine core initialization failed"
        ));
        assert!(!log_shows_oom("OSError: model weights not found"));
        assert!(!log_shows_oom(""));
    }

    #[test]
    fn every_rendered_line_break_keeps_the_gate_in_step_with_the_scorer() {
        // The scorer (`vllm_anchored_lines`) re-splits with `rendered_lines`,
        // which ends a row on far more than the `\n`/`\r` the segmenter splits
        // on. Splitting the log on `\r` alone therefore left the anchor on an
        // empty leading piece for every *other* break, the tail scored as "no
        // OOM", and both the engine hint and the serve-summary note silently
        // vanished. Each prefix below is a separate `rendered_lines` boundary.
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        for (label, break_seq) in [
            ("bare CR", "\r"),
            ("erase-line CSI after a CR repaint", "\r\u{1b}[2K"),
            ("cursor-up CSI", "\u{1b}[1A"),
            ("NEL escape", "\u{1b}E"),
            ("IND escape", "\u{1b}D"),
            ("vertical tab", "\u{b}"),
            ("form feed", "\u{c}"),
            ("C1 NEL", "\u{85}"),
            ("line separator", "\u{2028}"),
            ("paragraph separator", "\u{2029}"),
        ] {
            let tail = format!("Loading weights: 90%{break_seq}{exception}\n");
            assert!(
                log_shows_oom(&tail),
                "{label}: an allocator exception after this break must still read as an OOM"
            );
            let symptom = vllm_oom_diagnose_symptom(&tail)
                .unwrap_or_else(|| panic!("{label}: an OOM tail must select a line"));

            // The contract the callers actually rely on: *if* the line is
            // quotable it must be diagnosable, because quotability is what they
            // gate the printed command on. When it is not quotable the canonical
            // symptom stands in, and that is diagnosable by definition — so the
            // printed command reports a cause on every branch.
            //
            // The stronger "the returned line is always diagnosable" does NOT
            // hold here and must not be asserted: a segment carrying an escape
            // scores on its rendered rows but splits into an empty anchored row
            // when the checker is handed it whole. Making that claim true would
            // mean returning the stripped row, which is quotable, and a
            // colourised line would then be printed as a runnable lookalike of
            // the user's error instead of being refused.
            if terminal::quotable_in_single_quotes(&symptom) {
                assert!(
                    diagnose::vllm_oom_symptom_is_diagnosable(&symptom),
                    "{label}: a quotable selected line must be diagnosable"
                );
            } else {
                assert!(
                    diagnose::vllm_oom_symptom_is_diagnosable(VLLM_OOM_CANONICAL_SYMPTOM),
                    "{label}: the fallback the callers substitute must report a cause"
                );
            }

            // Either way the user still sees their real error: the raw bytes
            // survive selection, so rendering for display recovers the
            // exception rather than the progress-bar text that preceded it.
            //
            // Asserted as exact equality, not `contains`. `contains` passed on
            // the welded `Loading weights: 90%torch.OutOfMemoryError: ...` that
            // plain stripping produced, so it could not fail on the very bug
            // this line is here to catch.
            let displayed =
                vllm_oom_display_line(symptom.strip_prefix("vllm: ").unwrap_or(&symptom));
            assert_eq!(
                displayed, exception,
                "{label}: the displayed line must be the exception row on its own, with no \
                 progress-bar text welded onto its front"
            );

            // An escape-bearing segment must never pass the quoting guard, or it
            // would reach the printed command stripped into a lookalike.
            if break_seq.contains('\u{1b}') {
                assert!(
                    !terminal::quotable_in_single_quotes(&symptom),
                    "{label}: an escape-bearing symptom must not pass the quoting guard"
                );
            }
        }
    }

    #[test]
    fn the_displayed_oom_line_is_a_row_the_terminal_actually_drew() {
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";

        // The bug: `strip_terminal_control_sequences` deletes the break token
        // with nothing in its place, welding the progress bar onto the front of
        // the exception. The welded string is text vLLM never emitted, and it is
        // quotable (no control bytes left), so it would reach the printed
        // `rocm diagnose --symptom` command as a runnable lookalike.
        let repaint = format!("Loading weights: 90%\u{1b}[1A{exception}");
        let welded = terminal::strip_terminal_control_sequences(&repaint);
        assert_eq!(
            welded,
            format!("Loading weights: 90%{exception}"),
            "this pins the shape of the bug, so the assertion below cannot pass vacuously"
        );
        assert!(
            terminal::quotable_in_single_quotes(&welded),
            "the welded line is quotable, which is exactly why it must never be displayed \
             or written back into the shared log"
        );

        assert_eq!(
            vllm_oom_display_line(&repaint),
            exception,
            "the displayed line must be the rendered row that scored, not the whole segment \
             with its break tokens deleted"
        );

        // A plain single-row line is unchanged, so the common case is lossless.
        assert_eq!(vllm_oom_display_line(exception), exception);
        // A line that is not an OOM at all has no scoring row to pick, so the
        // helper still returns something printable rather than an empty string.
        assert_eq!(
            vllm_oom_display_line("\u{1b}[31mOSError: model weights not found\u{1b}[0m"),
            "OSError: model weights not found"
        );

        // When several rows of one segment score, the newest wins — the same
        // choice `vllm_oom_diagnose_symptom` makes with `.rev()` across
        // segments. The two run inside one message, so picking opposite ends
        // would have the hint display one line while the `--symptom` argument
        // names another.
        let retry = "RuntimeError: hipErrorOutOfMemory";
        let proximate = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        assert!(
            diagnose::vllm_oom_symptom_is_diagnosable(&format!("vllm: {retry}"))
                && diagnose::vllm_oom_symptom_is_diagnosable(&format!("vllm: {proximate}")),
            "both rows must score, or this fixture cannot discriminate which end is taken"
        );
        assert_eq!(
            vllm_oom_display_line(&format!("{retry}\u{1b}[1A{proximate}")),
            proximate,
            "the displayed row must be the last one that scored, not the first"
        );
    }

    #[test]
    fn the_engine_hint_is_never_quoted_back_as_the_users_own_line() {
        // A managed launch shares one log between the supervisor and the engine,
        // and the engine's bail (which embeds this crate's own OOM hint) is
        // written back into it *after* the real failure. That hint quotes the
        // user's line and names the allocator keywords, so it scores as an OOM
        // too — and being last, a newest-first scan always selected it. Since it
        // contains `'`, the quoting guard then rejected it and the printed
        // command fell back to the canonical symptom in every real run, which is
        // precisely the promise the note is built on.
        let real_line = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let tail = format!(
            "{real_line}\n\
             Error: vLLM server process exited before becoming ready (status: exit status: 1)\n\
             \n\
             Startup log, last 50 lines as the terminal drew them:\n\
             {real_line}\n\
             \n\
             {VLLM_OOM_DETECTED_PREFIX}{real_line}). {VLLM_GPU_MEMORY_UTILIZATION_HINT}\n\
             For conditional remediation, run `{VLLM_OOM_DIAGNOSE_COMMAND} \
             'vllm: {real_line}'`.\n"
        );
        let symptom = vllm_oom_diagnose_symptom(&tail).expect("the real OOM line is still found");
        assert_eq!(
            symptom,
            format!("vllm: {real_line}"),
            "the user's own failing line must win over rocm-cli's own hint"
        );
        // ...and because it is the user's real line, it is quotable, so the
        // command the user is told to run actually carries their error.
        assert!(
            terminal::quotable_in_single_quotes(&symptom),
            "the selected line must survive the quoting guard, or the note falls \
             back to the canonical symptom exactly as it did before this fix"
        );
    }

    #[test]
    fn vllm_oom_diagnose_symptom_selects_the_failing_line_or_reports_no_oom() {
        // The selector *is* the gate: a tail with an OOM always yields a line,
        // and a tail without one always yields `None`, which is what lets
        // callers use this directly instead of asking the same question twice.
        let tail = concat!(
            "  File \"/opt/vllm/worker.py\", line 212, in load_model\n",
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.\n",
            "INFO shutting down worker\n"
        );
        assert_eq!(
            vllm_oom_diagnose_symptom(tail).as_deref(),
            Some("vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB."),
            "the user's own failing line must be routed into `rocm diagnose --symptom`"
        );
        // Whatever comes back must be diagnosable, or the printed command would
        // report no known cause.
        let symptom = vllm_oom_diagnose_symptom(tail).expect("an OOM tail selects a line");
        assert!(diagnose::vllm_oom_symptom_is_diagnosable(&symptom));
        // The *last* OOM line wins: a tail ends with the failure that killed the
        // launch, and an earlier retry's line would misreport it.
        assert_eq!(
            vllm_oom_diagnose_symptom("RuntimeError: hipErrorOutOfMemory\nHIP OUT OF MEMORY\n")
                .as_deref(),
            Some("vllm: HIP OUT OF MEMORY")
        );
        // A progress bar repaints with a bare `\r`, so vLLM's own output puts
        // the repaint and the exception that follows it on ONE `\n` line. The
        // diagnostic re-splits on `\r` and keeps only anchored segments, so
        // anchoring the physical line once left the exception unanchored and
        // discarded: the tail read as "no OOM" and both the engine hint and the
        // serve-summary note vanished for the commonest shape of this failure.
        let repainted = "Loading weights: 90%\rtorch.OutOfMemoryError: HIP out of memory. \
                         Tried to allocate 7.21 GiB.\n";
        assert!(
            log_shows_oom(repainted),
            "an allocator exception after a progress-bar carriage return is still an OOM"
        );
        let repainted_symptom =
            vllm_oom_diagnose_symptom(repainted).expect("the CR-split tail still selects a line");
        assert_eq!(
            repainted_symptom,
            "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
            "the exception segment must be quoted, not the progress-bar segment that \
             happened to carry the anchor"
        );
        // ...and what comes back is still diagnosable, so the printed command
        // reports a cause rather than reading as "checked, found nothing".
        assert!(diagnose::vllm_oom_symptom_is_diagnosable(
            &repainted_symptom
        ));
        // No OOM line, no symptom — including the sub-threshold shapes the
        // shared classifier rejects, which must not be reported as a cause.
        assert_eq!(
            vllm_oom_diagnose_symptom("OSError: model weights not found"),
            None
        );
        assert_eq!(
            vllm_oom_diagnose_symptom("Out of memory: Killed process 4242 (python)"),
            None
        );
        assert_eq!(vllm_oom_diagnose_symptom(""), None);
    }
}
