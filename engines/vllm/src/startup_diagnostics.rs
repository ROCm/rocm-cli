// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::Result;
use rocm_core::terminal::{
    is_control_or_format, is_control_or_line_separator, strip_terminal_control_sequences,
};
use std::path::Path;

const STARTUP_FAILURE_LOG_TAIL_LINES: usize = 80;

/// Builds a human-readable summary of the tail of the startup log, if available.
/// Returns an empty string when no log is present or it cannot be read.
///
/// The tail is sanitized before it is embedded. It is untrusted subprocess
/// output and this whole string is `bail!`-ed to the user's terminal, so
/// stripping control bytes out of the one echoed line further down while
/// printing the raw tail a few lines above would have left the escapes a clear
/// path to the terminal inside the same message. Stripping is per line so the
/// tail keeps its shape; `oom_utilization_hint` and `import_failure_hint` still
/// read the *raw* tail, because what they select from it is quoted into a
/// command and has its own, stricter admissibility test.
pub(crate) fn startup_log_context(log_path: Option<&Path>) -> String {
    let summary = log_path
        .and_then(|p| summarize_startup_log_tail(p, STARTUP_FAILURE_LOG_TAIL_LINES).ok())
        .unwrap_or_default();
    if summary.is_empty() {
        return String::new();
    }
    let hint = format!(
        "{}{}",
        oom_utilization_hint(&summary),
        import_failure_hint(&summary)
    );
    let summary = summary
        .lines()
        .map(strip_terminal_control_sequences)
        .collect::<Vec<_>>()
        .join("\n");
    format!("\n\nLast {STARTUP_FAILURE_LOG_TAIL_LINES} lines of startup log:\n{summary}{hint}")
}

/// Appends [`rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT`] when the startup-log
/// tail shows an out-of-memory failure, and nothing otherwise so healthy
/// failures are not cluttered with an irrelevant memory hint. Shares its wording
/// with the `rocm` CLI's pre-launch low-VRAM note so both surfaces point the
/// user at the same fix rather than drifting into different phrasing.
fn oom_utilization_hint(log_tail: &str) -> String {
    if !log_tail_shows_oom(log_tail) {
        return String::new();
    }
    // Route the user's *actual* failing line into the `--symptom` example when
    // the diagnose checker would actually score it *and* it can be rendered as
    // one intact single-quoted argument; otherwise fall back to the canonical
    // symptom so the printed command always reports a cause. The detector here
    // is a coarse substring scan that accepts lines the scorer rates
    // sub-threshold (e.g. a bare "... out of memory"), so without this fallback
    // the diagnose command could report nothing -- which reads as "the tool
    // checked and there's no known cause", worse than not printing it.
    // `.rev()` is deliberate, not incidental: when several lines of the tail
    // mention running out of memory, the *last* one is the proximate failure —
    // the point where the process actually gave up — and the earlier ones are
    // usually the allocator's own retry chatter leading up to it. The cost is
    // real and worth stating: on a tail like "torch.OutOfMemoryError: ... Tried
    // to allocate 7.21 GiB. GPU 0 has a total capacity of 24.00 GiB." followed
    // by a bare "RuntimeError: ... killed: out of memory", the vaguer line wins
    // even though the first carries the allocation size and the capacity. Both
    // are diagnosable, so this picks which line is quoted, never whether one is.
    // The user still sees every line: the log tail is printed above this hint.
    let raw_line = log_tail
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty() && log_tail_shows_oom(line))
        .unwrap_or("out of memory");
    // The line is subprocess output, so it is echoed only after the terminal
    // control bytes vLLM's colourised logger emits are removed.
    let symptom_line = strip_terminal_control_sequences(raw_line);
    let candidate = format!("vllm: {raw_line}");
    let symptom = if quotable_in_single_quotes(&candidate)
        && rocm_core::vllm_oom_symptom_is_diagnosable(&candidate)
    {
        candidate
    } else {
        rocm_core::VLLM_OOM_CANONICAL_SYMPTOM.to_owned()
    };
    format!(
        "\n\nDetected an out-of-memory failure ({symptom_line}). {}\n\
         For conditional remediation, run `rocm diagnose --symptom '{symptom}'`.",
        rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT
    )
}

/// Whether `symptom` can be placed inside a `'...'` shell word verbatim.
///
/// The value is untrusted subprocess output (the vLLM startup log tail) and the
/// message it lands in invites the user to paste the command into a shell, so a
/// bare `'` would close the quote and let text nobody vetted become shell
/// syntax. An apostrophe is routine in Python error text (`can't allocate`,
/// `model 'foo'`), and a line only has to mention running out of memory to be
/// selected, so this is an ordinary case rather than an exotic one.
///
/// Rejecting instead of escaping (`'` -> `'\''`) is deliberate. Escaping keeps
/// the exact bytes but yields a command a reader cannot check by eye, and a
/// wrong escape is *runnable* and misleading rather than obviously broken;
/// control bytes would still reach the terminal. The canonical fallback is the
/// branch that already exists for "this line cannot be used", and it is
/// guaranteed to report a cause. The user's own line stays visible in the
/// human-readable sentence above the command (and in the log tail printed with
/// it), so nothing is lost but the copy-paste convenience.
///
/// The character test is [`is_control_or_format`] together with
/// [`is_control_or_line_separator`], because each covers scalars the other does
/// not and `symptom` is the *raw* log line, not the stripped one: a `Cf` bidi
/// override reordered how the printed command renders, and a `Zl` line
/// separator broke it across two rows.
fn quotable_in_single_quotes(symptom: &str) -> bool {
    !symptom.contains('\'')
        && !symptom
            .chars()
            .any(|c| is_control_or_format(c) || is_control_or_line_separator(c))
}

/// Case-insensitive scan for the out-of-memory signatures vLLM/PyTorch emit on a
/// HIP allocation failure (e.g. `torch.OutOfMemoryError: HIP out of memory`).
fn log_tail_shows_oom(log_tail: &str) -> bool {
    let lower = log_tail.to_ascii_lowercase();
    lower.contains("out of memory") || lower.contains("outofmemory")
}

/// Points at `rocm diagnose` when the startup-log tail shows the engine import
/// failure (`OSError: libtorch_cuda.so: cannot open shared object file`), and
/// prints nothing otherwise. It is the sibling of [`oom_utilization_hint`]:
/// without it, reaching the catalog entry for this failure meant already knowing
/// to paste the error into `rocm diagnose`.
///
/// The message says only what was detected. The catalog entry carries the
/// conditions under which its remediation applies, so this does not advise a
/// reinstall itself.
///
/// The quoted symptom follows the same rule as the out-of-memory hint: the
/// user's own line when the checker scores it and it renders as one intact
/// single-quoted argument, the canonical symptom otherwise, so the printed
/// command always reports a cause.
fn import_failure_hint(log_tail: &str) -> String {
    let Some(raw_line) = import_failure_line(log_tail) else {
        return String::new();
    };
    // The line is subprocess output, so it is echoed only after the terminal
    // control bytes the engine's colourised logger emits are removed.
    let symptom_line = strip_terminal_control_sequences(raw_line);
    let symptom = if quotable_in_single_quotes(raw_line)
        && rocm_core::vllm_import_symptom_is_diagnosable(raw_line)
    {
        raw_line
    } else {
        rocm_core::VLLM_IMPORT_CANONICAL_SYMPTOM
    };
    format!(
        "\n\nDetected an engine import failure ({symptom_line}).\n\
         For conditional remediation, run `rocm diagnose --symptom '{symptom}'`."
    )
}

/// The line of the tail that names the import failure, if there is one.
///
/// The line that names the missing library is the proximate failure, so it wins
/// and the last one is taken. It must also contain `OSError:` or
/// `ImportError:`. That is a substring match and not a prefix match, because
/// the real line begins with a process prefix such as `(EngineCore pid=412) `.
/// A non-fatal warning that quotes the exception in its repr form, such as
/// `Failed to import from vllm._C with ImportError('libtorch_cuda.so ...')`,
/// names the library too, but it is not the failure, and it is excluded. A
/// warning written with a colon is not excluded. The traceback frame of the
/// extension's loader (`File "...", line N` naming `_optional_torch_c_dlpack`)
/// is the fallback.
/// `torch_c_dlpack_ext` alone is deliberately not a signature: it also appears
/// in healthy package listings. A bare mention of the shim module is not one
/// either, because a `warnings.warn` from that module prints
/// `.../_optional_torch_c_dlpack.py:NN: UserWarning: ...` on a healthy start.
/// A hint on an unrelated failure is worse than none.
fn import_failure_line(log_tail: &str) -> Option<&str> {
    let lines = || {
        log_tail
            .lines()
            .rev()
            .map(str::trim)
            .filter(|line| !line.is_empty())
    };
    lines()
        .find(|line| {
            let lower = line.to_ascii_lowercase();
            lower.contains("libtorch_cuda.so")
                && (lower.contains("oserror:") || lower.contains("importerror:"))
        })
        .or_else(|| {
            lines().find(|line| {
                let lower = line.to_ascii_lowercase();
                lower.contains("file \"") && lower.contains("_optional_torch_c_dlpack")
            })
        })
}

/// Reads the last N lines from a log file and returns them as a formatted string.
/// Delegates the seek/tail logic to `rocm_core::tail_lines`.
fn summarize_startup_log_tail(log_path: &Path, limit: usize) -> Result<String> {
    let lines = rocm_core::tail_lines(log_path, limit, crate::state::MAX_TAIL_READ)?;
    if lines.is_empty() {
        return Ok(String::new());
    }
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::current_unix_millis;
    use std::fs;

    #[test]
    fn oom_utilization_hint_fires_on_out_of_memory_log_tails() {
        // The exact PyTorch/HIP signature from the field report.
        let torch = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        assert!(log_tail_shows_oom(torch));
        let hint = oom_utilization_hint(torch);
        assert!(
            hint.contains("--gpu-memory-utilization"),
            "an OOM tail must surface the utilization workaround: {hint}"
        );
        assert!(hint.contains("--gpu <index>"));
        assert!(
            hint.contains("rocm diagnose --symptom"),
            "an OOM tail must route the user to the conditional catalog entry: {hint}"
        );
        assert!(
            hint.contains(
                "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB."
            ),
            "the user's real failing line must be routed into --symptom, not a canned string: {hint}"
        );

        // Detection is case-insensitive and also matches the spaced phrasing.
        assert!(log_tail_shows_oom("HIP OUT OF MEMORY"));
        assert!(log_tail_shows_oom("RuntimeError: CUDA out of memory"));
    }

    #[test]
    fn every_emitted_oom_symptom_is_diagnosable_and_names_the_branch_that_produced_it() {
        // Closes the loop between the two layers: the engine prints
        // `rocm diagnose --symptom '<symptom>'`, so whatever it emits must
        // actually score for the diagnose checker -- including for lines the
        // coarse substring detector accepts but the scorer rates sub-threshold
        // on their own (those fall back to the canonical symptom).
        //
        // Each case pins the *exact* symptom, not just that one is diagnosable.
        // Diagnosability alone cannot fail for the defect this test is named
        // for: the canonical symptom is diagnosable by construction, so a
        // mutant that deletes the "route the user's real failing line" feature
        // entirely and always returns VLLM_OOM_CANONICAL_SYMPTOM satisfies
        // every such assertion. Naming the expected symptom per line is what
        // makes the routing branch and the fallback branch separately
        // falsifiable -- and it is why the table below deliberately mixes the
        // two: three lines score on their own and must be quoted verbatim,
        // three do not and must fall back.
        let accepted_lines = [
            (
                "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
                "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
            ),
            (
                "RuntimeError: hipErrorOutOfMemory",
                "vllm: RuntimeError: hipErrorOutOfMemory",
            ),
            ("CUDA out of memory", "vllm: CUDA out of memory"),
            // Sub-threshold on their own: the class name with no allocator
            // message, the allocator message with no class name, and a bare
            // kill notice. These must reach the canonical fallback.
            (
                "torch.cuda.OutOfMemoryError",
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            ),
            (
                "HIP error: out of memory",
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            ),
            (
                "the process was killed: out of memory",
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            ),
        ];
        let mut routed_verbatim = 0;
        for (line, expected) in accepted_lines {
            assert!(log_tail_shows_oom(line), "detector must accept: {line}");
            let hint = oom_utilization_hint(line);
            let symptom = quoted_symptom_argument(&hint);
            assert_eq!(
                symptom, expected,
                "wrong branch for {line:?}: the engine must quote the user's own line when \
                 the checker scores it and fall back to the canonical symptom only when it \
                 does not"
            );
            assert!(
                rocm_core::vllm_oom_symptom_is_diagnosable(symptom),
                "emitted symptom must be diagnosable, got {symptom:?} for line {line:?}"
            );
            if symptom != rocm_core::VLLM_OOM_CANONICAL_SYMPTOM {
                routed_verbatim += 1;
            }
        }
        assert_eq!(
            routed_verbatim, 3,
            "the table must keep exercising both branches; if a scoring change moved a line \
             across the threshold, re-pick the fixture rather than relaxing the expectation"
        );
    }

    /// The `--symptom` value the hint actually hands the user, read back out of
    /// the rendered text exactly the way a shell would: everything between the
    /// opening quote that follows `--symptom ` and the next `'`.
    fn quoted_symptom_argument(hint: &str) -> &str {
        hint.split("--symptom '")
            .nth(1)
            .and_then(|rest| rest.split('\'').next())
            .expect("hint must carry a --symptom value")
    }

    #[test]
    fn an_apostrophe_in_the_failing_line_cannot_break_out_of_the_printed_command() {
        // A realistic vLLM/PyTorch failing line: apostrophes are routine in
        // Python error text, and this one scores well above MIN_SCORE_FOR_MATCH
        // (torch.OutOfMemoryError + HIP out of memory), so the "route the user's
        // real line" branch selects it.
        let log_tail = concat!(
            "ERROR 09-14 12:00:01 engine.py:389] torch.OutOfMemoryError: HIP out of memory. ",
            "Tried to allocate 7.21 GiB. GPU 0 can't allocate the model's weights."
        );
        assert!(log_tail_shows_oom(log_tail));
        let hint = oom_utilization_hint(log_tail);

        // The rendered command must be one intact single-quoted argument: no
        // user-controlled byte may close the quote, so nothing the subprocess
        // printed can land outside it in a command the user is told to paste.
        let command = hint
            .split("run `")
            .nth(1)
            .and_then(|rest| rest.split('`').next())
            .expect("hint must print a runnable command");
        assert_eq!(
            command.matches('\'').count(),
            2,
            "the --symptom argument must stay a single balanced quoted word: {command}"
        );
        let symptom = quoted_symptom_argument(&hint);
        // Pin which branch ran, not just that no apostrophe survived: "contains
        // no `'`" also holds if the apostrophes were silently deleted from the
        // user's line, which is the escaping-style behaviour the fallback exists
        // to avoid. Rejection means the canonical symptom, exactly.
        assert_eq!(
            symptom,
            rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            "a quote-bearing line must be rejected in favour of the canonical symptom, \
             not silently rewritten: {symptom:?}"
        );
        // ...and the command it does print must still report a cause.
        assert!(
            rocm_core::vllm_oom_symptom_is_diagnosable(symptom),
            "the fallback symptom must still be diagnosable: {symptom:?}"
        );
    }

    #[test]
    fn control_bytes_from_the_log_are_stripped_from_the_echoed_line() {
        // vLLM's logger colourises; an ANSI-coloured OOM line must not repaint
        // the user's terminal from inside rocm-cli's own error message.
        let log_tail = "\u{1b}[31mRuntimeError: HIP out of memory\u{1b}[0m\u{7}";
        let hint = oom_utilization_hint(log_tail);
        assert!(
            !hint.chars().any(|c| c.is_control() && c != '\n'),
            "no control byte may survive into the printed hint: {hint:?}"
        );
        // Pin the exact rendering rather than the absence of a few fragments.
        // Absence checks cannot fail for the defect they name: a stripper that
        // drops the escape byte and the `[` it introduces and nothing further
        // leaves `31m`/`0m` behind, which contains neither `[31m` nor `[0m` and
        // carries no control byte, so every absence check still passes.
        assert_eq!(
            strip_terminal_control_sequences(log_tail),
            "RuntimeError: HIP out of memory",
            "the ANSI sequence must be removed whole, not just its escape byte"
        );
        // The colourised line is not quotable (control bytes), so the command
        // falls back to the canonical symptom while the echoed sentence keeps
        // the user's own line, stripped.
        assert_eq!(
            hint,
            format!(
                "\n\nDetected an out-of-memory failure (RuntimeError: HIP out of memory). {}\n\
                 For conditional remediation, run `rocm diagnose --symptom '{}'`.",
                rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT,
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM
            ),
            "the echoed line must render exactly, with the whole escape sequence gone"
        );
    }

    #[test]
    fn the_stripper_follows_the_escape_grammar_not_just_the_colour_case() {
        // The `'m'`-terminated SGR case above is the *easy* one, and on its own
        // it pins almost nothing: narrowing the CSI final-byte range from
        // `0x40..=0x7E` to just `'m'` leaves it green. These cases pin the
        // range, the parameter/intermediate classes, and the introducers.
        //
        // They are not academic. This input is a *killed* process's output, so
        // truncated and interleaved sequences are the normal case on this code
        // path, and every one of them used to corrupt the message the user
        // reads -- the failures are quoted per case below.
        for (raw, expected, defect) in [
            // Non-`m` CSI finals: `K` (erase-in-line) and `A` (cursor-up) are
            // ordinary logger output, and narrowing the final-byte range to
            // `'m'` leaves them in the message verbatim.
            (
                "\u{1b}[2KRuntimeError: HIP out of memory",
                "RuntimeError: HIP out of memory",
                "a non-`m` CSI final must terminate the sequence",
            ),
            (
                "\u{1b}[1ARuntimeError: HIP out of memory",
                "RuntimeError: HIP out of memory",
                "a non-`m` CSI final must terminate the sequence",
            ),
            // Truncated CSI immediately followed by a well-formed one: the old
            // scan consumed the second sequence's `ESC [` as the first one's
            // parameters and stopped at `0`, yielding "0mKilled: out of memory".
            (
                "\u{1b}[1;2\u{1b}[0mKilled: out of memory",
                "Killed: out of memory",
                "a truncated CSI must not swallow the next sequence's introducer",
            ),
            // A multi-byte scalar can never be in `0x40..=0x7E`, so the old scan
            // ran past it and ate to the next byte that happened to land in
            // range -- this yielded "ut of memory", losing the `o`.
            (
                "\u{1b}[12\u{e9} out of memory",
                "\u{e9} out of memory",
                "an invalid CSI byte must end the sequence, not be scanned past",
            ),
            // `ESC ESC`: the old code dropped the second `ESC` as the first
            // one's argument, then emitted `[0m` as literal text.
            (
                "\u{1b}\u{1b}[0mRuntimeError: HIP out of memory",
                "RuntimeError: HIP out of memory",
                "a stray ESC must not consume the next sequence's introducer",
            ),
            // OSC: the old code took the `else` branch on `]`, so the window
            // title leaked into the message as "0;titleRuntimeError: ...".
            (
                "\u{1b}]0;title\u{7}RuntimeError: HIP out of memory",
                "RuntimeError: HIP out of memory",
                "an OSC body must not be emitted as text",
            ),
            // ...and with the ST terminator (`ESC \`) rather than BEL.
            (
                "\u{1b}]0;title\u{1b}\\RuntimeError: HIP out of memory",
                "RuntimeError: HIP out of memory",
                "an OSC terminated by ST must be consumed whole",
            ),
            // A two-character escape with no CSI at all.
            (
                "\u{1b}7RuntimeError: HIP out of memory",
                "RuntimeError: HIP out of memory",
                "a simple escape must consume exactly its final byte",
            ),
            // Cf format characters: `char::is_control` is category Cc only, so
            // U+202E survived and reversed how the rest of the line renders.
            (
                "RuntimeError: \u{202e}HIP out of memory",
                "RuntimeError: HIP out of memory",
                "a bidi override must not survive into the message",
            ),
            // The `Zl`/`Zp` pair that `is_control_or_line_separator` adds to
            // `char::is_control`: drop it from that predicate and both of these
            // go red, as do the quoting assertions below.
            (
                "RuntimeError: HIP\u{2028}out of memory",
                "RuntimeError: HIPout of memory",
                "a line separator must not survive into the message",
            ),
            (
                "RuntimeError: HIP\u{2029}out of memory",
                "RuntimeError: HIPout of memory",
                "a paragraph separator must not survive into the message",
            ),
            // The text-only control: nothing is removed from a clean line.
            (
                "RuntimeError: HIP out of memory",
                "RuntimeError: HIP out of memory",
                "a clean line must pass through untouched",
            ),
        ] {
            assert_eq!(
                strip_terminal_control_sequences(raw),
                expected,
                "{defect}: {raw:?}"
            );
        }

        // Every one of those scalars must also be inadmissible in the quoted
        // command, not merely stripped from the echoed sentence: the candidate
        // the guard sees is built from the raw line, not the stripped one, so
        // the second call site needs its own pins.
        assert!(
            !quotable_in_single_quotes("vllm: \u{202e}HIP out of memory"),
            "a bidi override must make a line unquotable, not ride into the command"
        );
        assert!(
            !quotable_in_single_quotes("vllm: HIP\u{2028}out of memory"),
            "a line separator must make a line unquotable, not ride into the command"
        );
        assert!(
            !quotable_in_single_quotes("vllm: HIP\u{2029}out of memory"),
            "a paragraph separator must make a line unquotable, not ride into the command"
        );
        assert!(
            quotable_in_single_quotes("vllm: HIP out of memory"),
            "the rejection must not be so broad that ordinary lines stop qualifying"
        );
    }

    #[test]
    fn the_printed_log_tail_is_sanitized_not_just_the_echoed_line() -> Result<()> {
        // The stripper's contract is that nothing in this message can repaint
        // the user's terminal. The log tail is printed a few lines above the
        // echoed line in the same `bail!`-ed string, and it used to go out raw,
        // so the escapes removed from one had a clear path to the terminal via
        // the other.
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-sanitize-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(
            &path,
            "\u{1b}[31mINFO starting\u{1b}[0m\n\u{1b}]0;pwned\u{7}RuntimeError: HIP out of memory\n",
        )?;
        let context = startup_log_context(Some(path.as_path()));
        fs::remove_file(&path).ok();

        assert!(
            !context.chars().any(|c| c.is_control() && c != '\n'),
            "no control byte may survive into the printed failure message: {context:?}"
        );
        assert!(
            !context.contains("pwned"),
            "an OSC body in the tail must not be emitted as text: {context:?}"
        );
        assert_eq!(
            context
                .split("startup log:\n")
                .nth(1)
                .and_then(|rest| rest.split("\n\nDetected").next()),
            Some("INFO starting\nRuntimeError: HIP out of memory"),
            "the tail must keep its text and its line structure, minus the escapes: {context:?}"
        );
        Ok(())
    }

    #[test]
    fn oom_utilization_hint_stays_quiet_for_unrelated_failures() {
        let unrelated = "ValueError: model architecture 'FooForCausalLM' is not supported";
        assert!(!log_tail_shows_oom(unrelated));
        assert!(
            oom_utilization_hint(unrelated).is_empty(),
            "non-OOM failures must not carry a memory hint"
        );
    }

    // ── The engine import failure: a pointer at `rocm diagnose` ──────────────

    /// A startup-log tail as the engine writes it when the extension loads its
    /// CUDA variant on a ROCm build of torch.
    const IMPORT_TRACEBACK: &str = "\
(EngineCore pid=412) Traceback (most recent call last):
(EngineCore pid=412)   File \"/opt/venv/lib/python3.12/site-packages/tvm_ffi/_optional_torch_c_dlpack.py\", line 98, in load_torch_c_dlpack_extension
(EngineCore pid=412)     lib = ctypes.CDLL(str(lib_path))
(EngineCore pid=412) OSError: libtorch_cuda.so: cannot open shared object file: No such file or directory";

    #[test]
    fn import_failure_hint_names_the_failing_line_and_the_diagnose_command() {
        let hint = import_failure_hint(IMPORT_TRACEBACK);
        let line = "(EngineCore pid=412) OSError: libtorch_cuda.so: cannot open shared object file: No such file or directory";
        assert_eq!(
            hint,
            format!(
                "\n\nDetected an engine import failure ({line}).\n\
                 For conditional remediation, run `rocm diagnose --symptom '{line}'`."
            ),
            "the line that names the missing library is the one to quote, verbatim"
        );
        assert!(
            rocm_core::vllm_import_symptom_is_diagnosable(quoted_symptom_argument(&hint)),
            "the command the hint prints must report a cause: {hint}"
        );
    }

    #[test]
    fn import_failure_hint_is_part_of_the_startup_failure_message() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("vllm.log");
        std::fs::write(&log, IMPORT_TRACEBACK).expect("write log");
        let context = startup_log_context(Some(&log));
        assert!(
            context.contains("Last 80 lines of startup log:"),
            "{context}"
        );
        assert!(
            context.contains(
                "rocm diagnose --symptom '(EngineCore pid=412) OSError: libtorch_cuda.so"
            ),
            "the user must be pointed at the diagnosis from the failure message: {context}"
        );
        assert!(
            !context.contains("--gpu-memory-utilization"),
            "an import failure must not carry the memory hint: {context}"
        );
    }

    #[test]
    fn every_emitted_import_symptom_is_diagnosable_and_names_the_branch_that_produced_it() {
        // The same loop-closing check as the out-of-memory sweep, and for the
        // same reason: the engine prints `rocm diagnose --symptom '<symptom>'`,
        // so whatever it emits has to score. Each case pins the exact symptom, so
        // the routing branch and the fallback branch are separately falsifiable
        // (a mutant that always returns the canonical symptom would otherwise
        // satisfy every diagnosability assertion).
        let cases = [
            // The library line scores on its own and is quoted verbatim. (Not the
            // canonical string itself, which would make the two branches
            // indistinguishable here.)
            (
                "ImportError: libtorch_cuda.so: cannot open shared object file",
                "ImportError: libtorch_cuda.so: cannot open shared object file",
            ),
            (
                "(EngineCore pid=1) OSError: libtorch_cuda.so: cannot open shared object file: No such file or directory",
                "(EngineCore pid=1) OSError: libtorch_cuda.so: cannot open shared object file: No such file or directory",
            ),
            // A frame that names only the shim's module is worth 35, below the
            // threshold, so it falls back. (The same frame with the loader
            // function's name adds a second signal and scores on its own.)
            (
                "File \"tvm_ffi/_optional_torch_c_dlpack.py\", line 98, in <module>",
                rocm_core::VLLM_IMPORT_CANONICAL_SYMPTOM,
            ),
            // An apostrophe cannot be quoted safely, so it falls back too.
            (
                "OSError: libtorch_cuda.so: can't open shared object file",
                rocm_core::VLLM_IMPORT_CANONICAL_SYMPTOM,
            ),
        ];
        for (line, expected) in cases {
            let hint = import_failure_hint(line);
            let symptom = quoted_symptom_argument(&hint);
            assert_eq!(
                symptom, expected,
                "wrong branch for {line:?}: quote the user's own line when the checker scores \
                 it and it is safe to quote, and fall back to the canonical symptom otherwise"
            );
            assert!(
                rocm_core::vllm_import_symptom_is_diagnosable(symptom),
                "emitted symptom must be diagnosable, got {symptom:?} for line {line:?}"
            );
        }
    }

    #[test]
    fn control_bytes_from_the_log_are_stripped_from_the_echoed_import_line() {
        let log_tail =
            "\u{1b}[31mOSError: libtorch_cuda.so: cannot open shared object file\u{1b}[0m\u{7}";
        let hint = import_failure_hint(log_tail);
        assert!(
            !hint.chars().any(|c| c.is_control() && c != '\n'),
            "no control byte may survive into the printed hint: {hint:?}"
        );
        assert_eq!(
            hint,
            format!(
                "\n\nDetected an engine import failure (OSError: libtorch_cuda.so: cannot open shared object file).\n\
                 For conditional remediation, run `rocm diagnose --symptom '{}'`.",
                rocm_core::VLLM_IMPORT_CANONICAL_SYMPTOM
            ),
            "the colourised line is not quotable, so the command falls back while the sentence \
             keeps the user's own line, stripped"
        );
    }

    #[test]
    fn import_failure_hint_stays_quiet_for_unrelated_failures() {
        for unrelated in [
            "ValueError: model architecture 'FooForCausalLM' is not supported",
            "torch.OutOfMemoryError: HIP out of memory",
            // An `OSError:` or `ImportError:` line that does not name the
            // library is not this failure. Without the library-name condition
            // in the primary match, every such line would raise the hint.
            "(EngineCore pid=412) OSError: [Errno 98] Address already in use",
            "ImportError: cannot import name 'FooForCausalLM' from 'vllm.model_executor'",
            // The extension named in a healthy package listing is not a failure.
            "INFO torch-c-dlpack-ext 0.2.1 installed; torch_c_dlpack_ext loaded",
            // A `warnings.warn` from the shim module on a healthy start names
            // the module on a line of its own. It is not a traceback frame.
            "/opt/venv/lib/python3.12/site-packages/tvm_ffi/_optional_torch_c_dlpack.py:98: UserWarning: falling back to the JIT path",
            "(EngineCore pid=412) /opt/venv/lib/python3.12/site-packages/tvm_ffi/_optional_torch_c_dlpack.py:98: UserWarning: falling back to the JIT path",
            "",
        ] {
            assert!(
                import_failure_hint(unrelated).is_empty(),
                "must stay quiet for {unrelated:?}"
            );
        }
    }

    #[test]
    fn import_failure_hint_stays_quiet_for_an_unrelated_traceback() {
        // A traceback frame that does not name the shim module is not an import
        // failure. Without the `_optional_torch_c_dlpack` condition in the
        // fallback, any `File "` frame in the tail would match.
        let tail = "\
(EngineCore pid=412) Traceback (most recent call last):
(EngineCore pid=412)   File \"/opt/venv/lib/python3.12/site-packages/vllm/engine/llm_engine.py\", line 12, in <module>
(EngineCore pid=412)     validate(config)
(EngineCore pid=412) ValueError: model architecture 'FooForCausalLM' is not supported";
        assert!(import_failure_hint(tail).is_empty());
        assert!(startup_log_context_for(tail).contains("ValueError"));
        assert!(!startup_log_context_for(tail).contains("rocm diagnose"));
    }

    #[test]
    fn a_non_fatal_warning_that_names_the_library_is_not_an_import_failure() {
        // Pinned: the primary match needs `OSError:` or `ImportError:` in the
        // line, so a warning that quotes the exception in its repr form as
        // `ImportError('...')` does not raise the hint when the real failure is
        // something else. A warning written with a colon would, which is a
        // known limit that `docs/vllm.md` states.
        let tail = "\
WARNING Failed to import from vllm._C with ImportError('libtorch_cuda.so: cannot open shared object file')
(EngineCore pid=1) ValueError: model architecture 'FooForCausalLM' is not supported";
        assert!(import_failure_hint(tail).is_empty());
    }

    /// Runs `startup_log_context` over a log file holding `tail`.
    fn startup_log_context_for(tail: &str) -> String {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("vllm.log");
        std::fs::write(&log, tail).expect("write log");
        startup_log_context(Some(&log))
    }

    #[test]
    fn startup_log_context_carries_the_memory_hint_for_an_out_of_memory_tail() {
        let context = startup_log_context_for("torch.OutOfMemoryError: HIP out of memory");
        assert!(
            context.contains("--gpu-memory-utilization"),
            "an out-of-memory failure must carry the memory hint: {context}"
        );
        assert!(
            !context.contains("Detected an engine import failure"),
            "{context}"
        );
    }

    #[test]
    fn startup_log_context_carries_both_hints_in_order_when_both_fire() {
        let tail = format!("torch.OutOfMemoryError: HIP out of memory\n{IMPORT_TRACEBACK}");
        let context = startup_log_context_for(&tail);
        let memory = context.find("--gpu-memory-utilization");
        let import = context.find("Detected an engine import failure");
        assert!(
            memory.is_some() && import.is_some(),
            "both paragraphs must be present: {context}"
        );
        assert!(
            memory < import,
            "the memory paragraph comes first, then the import paragraph: {context}"
        );
    }

    #[test]
    fn import_failure_hint_quotes_the_last_line_that_names_the_library() {
        // Two lines name the library, as when a failure is retried or chained.
        // The later one is the proximate failure, so it is the one quoted.
        let tail = "\
(EngineCore pid=1) OSError: libtorch_cuda.so: cannot open shared object file: first
(EngineCore pid=1) During handling of the above exception, another exception occurred:
(EngineCore pid=1) OSError: libtorch_cuda.so: cannot open shared object file: second";
        let hint = import_failure_hint(tail);
        assert!(hint.contains(": second"), "{hint}");
        assert!(!hint.contains(": first"), "{hint}");
    }

    #[test]
    fn a_loader_traceback_frame_is_the_fallback_when_no_line_names_the_library() {
        let frame = "File \"/opt/venv/lib/python3.12/site-packages/tvm_ffi/_optional_torch_c_dlpack.py\", line 98, in <module>";
        let hint = import_failure_hint(frame);
        assert!(hint.contains("Detected an engine import failure"), "{hint}");
        // The frame alone is below the threshold, so the command quotes the
        // canonical symptom rather than a line that would find nothing.
        assert_eq!(
            quoted_symptom_argument(&hint),
            rocm_core::VLLM_IMPORT_CANONICAL_SYMPTOM
        );
    }
}
