// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::Result;
use rocm_engine_protocol::DEFAULT_LOG_TAIL_LINES;
use std::path::Path;

/// Tail budget for the startup-failure summary (and its OOM hint).
///
/// Derived from [`DEFAULT_LOG_TAIL_LINES`] rather than spelled as its own
/// literal so this surface and the CLI's serve summary — which reads the same
/// protocol constant in `append_oom_serve_note` — cannot drift on what counts as
/// "the tail" of a failed launch. A change to the protocol budget moves both.
const STARTUP_FAILURE_LOG_TAIL_LINES: usize = DEFAULT_LOG_TAIL_LINES;

/// Builds a human-readable summary of the tail of the startup log, if available.
/// Returns an empty string when no log is present or it cannot be read.
///
/// The tail is sanitized before it is embedded. It is untrusted subprocess
/// output and this whole string is `bail!`-ed to the user's terminal, so
/// stripping control bytes out of the one echoed line further down while
/// printing the raw tail a few lines above would have left the escapes a clear
/// path to the terminal inside the same message. `oom_utilization_hint` and
/// `import_failure_hint` still read the *raw* tail, because what they select
/// from it is quoted into a command and has its own, stricter admissibility
/// test.
///
/// Sanitizing with [`rocm_core::terminal::rendered_lines`] rather than
/// `strip_terminal_control_sequences` is load-bearing, not a stylistic
/// preference. Stripping deletes every line-break token (`\r`, `ESC[1A`, `NEL`,
/// `VT`, `U+2028`…) *with no separator in its place*, so a progress-bar repaint
/// — `Loading weights: 90%\rtorch.OutOfMemoryError: HIP out of memory.`, which
/// vLLM emits as one `\n` line — was echoed as the single welded line
/// `Loading weights: 90%torch.OutOfMemoryError: HIP out of memory.`. That is
/// text the engine never printed, and it is worse than cosmetic: a managed
/// launch shares one log between the supervisor and the engine, so the CLI
/// writes this whole bail back into the same file, and the next read finds the
/// welded line, scores it as a genuine OOM, finds it quotable (the control bytes
/// are gone) and prints it inside the `rocm diagnose --symptom '…'` command the
/// serve summary invites the user to paste. Expanding each physical line into
/// the rows a terminal would have drawn keeps the echo faithful, keeps the
/// tail's shape, and leaves nothing weldable behind for the second read.
///
/// Pinned by `the_echoed_tail_never_welds_two_rendered_rows_into_one_line`.
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
        .flat_map(rocm_core::terminal::rendered_lines)
        .collect::<Vec<_>>()
        .join("\n");
    // The count is the *read* budget — how many physical lines of the log were
    // taken — not how many rows are printed below. Those differ, and can differ
    // a lot: each `\r` repaint of a progress bar expands into its own rendered
    // row, so three physical lines carrying 300 repaints are echoed as 300-odd
    // rows. Saying "last N lines" and then printing more of them is the kind of
    // small lie that costs a reader their trust in the rest of the message, so
    // the sentence says which N it means.
    format!(
        "\n\nStartup log, last {STARTUP_FAILURE_LOG_TAIL_LINES} lines as the terminal drew \
         them:\n{summary}{hint}"
    )
}

/// Appends [`rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT`] when the startup-log
/// tail shows an out-of-memory failure, and nothing otherwise so healthy
/// failures are not cluttered with an irrelevant memory hint. Shares its wording
/// with the `rocm` CLI's pre-launch low-VRAM note so both surfaces point the
/// user at the same fix rather than drifting into different phrasing.
fn oom_utilization_hint(log_tail: &str) -> String {
    // Route the user's *actual* failing line into the `--symptom` example when
    // it can be rendered as one intact single-quoted argument; otherwise fall
    // back to the canonical symptom so the printed command always reports a
    // cause rather than reading as "the tool checked and there's no known
    // cause". Selecting the line is the shared helper's job, so this surface and
    // the `rocm` CLI serve summary apply one selection *rule* to `rocm diagnose`
    // instead of hand-rolling it twice; it classifies each line with the *same*
    // rule the diagnose checker uses. One rule is not one answer: the two read
    // the shared log at different points, with this bail's own sanitized echo
    // written into it in between, so for a colourised line they can print
    // different (both sound) `--symptom` arguments -- this one the canonical
    // symptom, the CLI the rendered row. See `vllm_log_segments` in
    // `crates/rocm-core/src/vllm_oom.rs` for the measurement — spelled as a
    // file path, not a `rocm_core::` path, because that function is private to
    // its module and no such path resolves. Its guarantee is specifically that
    // a line passing `quotable_in_single_quotes` is diagnosable -- not that
    // every returned line is -- which is exactly what the branch below needs,
    // since the non-quotable branch substitutes the canonical symptom anyway.
    //
    // No line means the tail carries no OOM, so the helper's `None` is the OOM
    // gate too; there is no separate classifier predicate to ask first.
    //
    // The helper also skips lines rocm-cli wrote into the log itself, which
    // matters precisely here: this hint is rendered into the engine's bail, the
    // CLI writes that whole error back into the same shared log, and the hint
    // quotes the user's line and names the allocator keywords, so it scores as
    // an OOM just as strongly as the real line does. Being last, it would win
    // the newest-first scan on the next read — and it contains `'`, so the
    // printed command would fall back to the canonical symptom in every real
    // run. That is why the two markers below come from shared constants rather
    // than being spelled out here.
    let Some(symptom) = rocm_core::vllm_oom_diagnose_symptom(log_tail) else {
        return String::new();
    };
    // The line is subprocess output, so it is echoed only after the terminal
    // control bytes vLLM's colourised logger emits are removed -- but removing
    // them is not enough on its own. A break token deleted with nothing in its
    // place welds the two rows either side of it into a line the engine never
    // printed (`Loading weights: 90%torch.OutOfMemoryError: ...`), which would
    // then be presented here as the user's own output. The shared helper returns
    // the rendered row that actually scored, which is one row the terminal
    // really drew.
    let symptom_line =
        rocm_core::vllm_oom_display_line(symptom.strip_prefix("vllm: ").unwrap_or(&symptom));
    let symptom = if rocm_core::quotable_in_single_quotes(&symptom) {
        symptom
    } else {
        rocm_core::VLLM_OOM_CANONICAL_SYMPTOM.to_owned()
    };
    format!(
        "\n\n{}{symptom_line}). {}\n\
         For conditional remediation, run `{} '{symptom}'`.",
        rocm_core::VLLM_OOM_DETECTED_PREFIX,
        rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT,
        rocm_core::VLLM_OOM_DIAGNOSE_COMMAND,
    )
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
    let symptom_line = rocm_core::strip_terminal_control_sequences(raw_line);
    let symptom = if rocm_core::quotable_in_single_quotes(raw_line)
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
    use std::fmt::Write as _;
    use std::fs;

    /// The OOM gate [`oom_utilization_hint`] itself uses. There is no separate
    /// classifier predicate in `rocm-core`: the selector returning `Some` *is*
    /// the gate, so these tests assert against the function that ships.
    fn detects_oom(log: &str) -> bool {
        rocm_core::vllm_oom_diagnose_symptom(log).is_some()
    }

    #[test]
    fn the_hint_never_quotes_itself_back_on_a_second_read_of_the_shared_log() {
        // A managed launch shares one log between the supervisor and the engine.
        // The engine bails with `startup_log_context`, which embeds *this* hint,
        // and the CLI then writes that whole error back into the same file. So
        // on the read that drives the serve summary the hint is in the log, it
        // is last, and it scores as an OOM just as strongly as the real line.
        //
        // That is the end-to-end shape of the bug: the newest-first scan picked
        // this sentence, which contains `'`, the quoting guard rejected it, and
        // the note fell back to the canonical symptom in every real run — so the
        // "we quote your own failing line" promise never once held in the field.
        // A unit test on the hint alone cannot see it; only feeding the hint
        // back through the selector can.
        let real_line = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let hint = oom_utilization_hint(real_line);
        assert!(!hint.is_empty(), "the real line must produce a hint");

        let second_read = format!("{real_line}\nError: launch failed{hint}\n");
        let symptom = rocm_core::vllm_oom_diagnose_symptom(&second_read)
            .expect("the real OOM line must still be found under the hint");
        assert_eq!(
            symptom,
            format!("vllm: {real_line}"),
            "the user's own line must win over the hint rocm-cli wrote itself"
        );
        assert!(
            rocm_core::quotable_in_single_quotes(&symptom),
            "and it must survive the quoting guard, or the printed command falls \
             back to the canonical symptom exactly as it did before this fix"
        );
    }

    #[test]
    fn oom_utilization_hint_fires_on_out_of_memory_log_tails() {
        // The exact PyTorch/HIP signature from the field report.
        let torch = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        assert!(detects_oom(torch));
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
        assert!(detects_oom("HIP OUT OF MEMORY"));
        assert!(detects_oom("RuntimeError: CUDA out of memory"));
    }

    #[test]
    fn every_emitted_oom_symptom_is_diagnosable_and_names_the_branch_that_produced_it() {
        // Closes the loop between the two layers: the engine prints
        // `rocm diagnose --symptom '<symptom>'`, so whatever it emits must
        // actually score for the diagnose checker. What makes that true is the
        // pair of branches, not the selector alone: a *quotable* selected line
        // is diagnosable because the selector scores with the checker's own
        // rule, and a non-quotable one is replaced by the canonical symptom,
        // which is diagnosable by definition. This guards both branches.
        //
        // Each case pins the *exact* symptom, not just that one is diagnosable.
        // Diagnosability alone cannot fail for the defect this test is named
        // for: the canonical symptom is diagnosable by construction, so a
        // mutant that deletes the "route the user's real failing line" feature
        // entirely and always returns VLLM_OOM_CANONICAL_SYMPTOM satisfies
        // every such assertion. Naming the expected symptom per line is what
        // makes the routing branch and the fallback branch separately
        // falsifiable -- and it is why the table below deliberately mixes the
        // two.
        //
        // Since the detector was reconciled with the checker, the fallback is no
        // longer reached by sub-threshold scoring (those lines now produce no
        // hint at all -- see `sub_threshold_lines_carry_no_hint_at_all`); it is
        // reached by a line that scores but cannot be rendered as one intact
        // single-quoted argument. So the fallback rows here are the
        // *unquotable* ones, which is also what keeps the admissibility guard
        // falsifiable through the surface that actually prints the command.
        let accepted_lines = [
            // Score and are quotable: the user's own line must be quoted verbatim.
            (
                "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
                "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
            ),
            (
                "RuntimeError: hipErrorOutOfMemory",
                "vllm: RuntimeError: hipErrorOutOfMemory",
            ),
            (
                "torch.cuda.OutOfMemoryError: CUDA out of memory",
                "vllm: torch.cuda.OutOfMemoryError: CUDA out of memory",
            ),
            ("CUDA out of memory", "vllm: CUDA out of memory"),
            // Score but are inadmissible in a single-quoted word, so they must
            // reach the canonical fallback: an apostrophe (which would close the
            // quote), a colourised line (control bytes), and a bidi override
            // (a `Cf` format character, which `char::is_control` does not catch
            // and which reorders how the printed command renders).
            (
                "torch.OutOfMemoryError: HIP out of memory. GPU 0 can't allocate the model's weights.",
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            ),
            (
                "\u{1b}[31mtorch.OutOfMemoryError: HIP out of memory\u{1b}[0m",
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            ),
            (
                "torch.OutOfMemoryError: \u{202e}HIP out of memory. Tried to allocate 7.21 GiB.",
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            ),
        ];
        let mut routed_verbatim = 0;
        let mut fell_back = 0;
        for (line, expected) in accepted_lines {
            assert!(detects_oom(line), "detector must accept: {line:?}");
            let hint = oom_utilization_hint(line);
            let symptom = quoted_symptom_argument(&hint);
            assert_eq!(
                symptom, expected,
                "wrong branch for {line:?}: the engine must quote the user's own line when \
                 it can be rendered as one intact single-quoted argument and fall back to \
                 the canonical symptom only when it cannot"
            );
            assert!(
                rocm_core::vllm_oom_symptom_is_diagnosable(symptom),
                "emitted symptom must be diagnosable, got {symptom:?} for line {line:?}"
            );
            if symptom == rocm_core::VLLM_OOM_CANONICAL_SYMPTOM {
                fell_back += 1;
            } else {
                routed_verbatim += 1;
            }
        }
        // Both counters, not just the verbatim one: a table that drifted until
        // every row took the same branch would still satisfy every per-row
        // assertion above, which is the vacuity this test was written to close.
        assert_eq!(
            (routed_verbatim, fell_back),
            (4, 3),
            "the table must keep exercising both branches; if an admissibility change \
             moved a line across the quotability boundary, re-pick the fixture rather than \
             relaxing the expectation. Scoring is not what separates these rows — every row \
             here is comfortably accepted by the detector, and the three fallbacks are \
             fallbacks purely because they cannot be rendered as one single-quoted word"
        );
    }

    #[test]
    fn sub_threshold_lines_carry_no_hint_at_all() {
        // The reconciled detector rejects sub-threshold lines the loose scan used
        // to accept -- exactly the false positives the diagnose checker's
        // threshold guards against. Rejecting them here keeps the two layers from
        // disagreeing (the engine must not print a `--symptom` the checker would
        // then score as "no known cause").
        //
        // This is a different property from the one above: there the question is
        // *which* symptom an emitted hint carries, here it is whether a hint is
        // emitted at all. Before the detector was reconciled these lines reached
        // the canonical fallback instead; now they produce nothing, and printing
        // a memory hint for them would be a regression the symptom table cannot
        // see.
        let rejected_lines = [
            // The bare exception class every PyTorch OOM raises, uncorroborated.
            "torch.cuda.OutOfMemoryError",
            // A spaced "out of memory" with no allocator anchor phrase.
            "HIP error: out of memory",
            // A kernel OOM-killer line from an unrelated subprocess.
            "the process was killed: out of memory",
        ];
        for line in rejected_lines {
            assert!(
                !detects_oom(line),
                "detector must reject sub-threshold line: {line}"
            );
            assert!(
                oom_utilization_hint(line).is_empty(),
                "a sub-threshold line must not carry a memory hint: {line}"
            );
        }
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
        assert!(detects_oom(log_tail));
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
            !hint
                .chars()
                .any(|c| c.is_control() && c != '\n' && c != '\t'),
            "no cursor-moving control byte may survive into the printed hint — `\\t` is \
             exempt because `rendered_lines` keeps it deliberately, as intra-line \
             whitespace that moves along the row it is already on: {hint:?}"
        );
        // Pin the exact rendering rather than the absence of a few fragments.
        // Absence checks cannot fail for the defect they name: a stripper that
        // drops the escape byte and the `[` it introduces and nothing further
        // leaves `31m`/`0m` behind, which contains neither `[31m` nor `[0m` and
        // carries no control byte, so every absence check still passes.
        assert_eq!(
            rocm_core::strip_terminal_control_sequences(log_tail),
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
            !context
                .chars()
                .any(|c| c.is_control() && c != '\n' && c != '\t'),
            "no cursor-moving control byte may survive into the printed failure message — \
             `\\t` is exempt, see the hint assertion above: {context:?}"
        );
        assert!(
            !context.contains("pwned"),
            "an OSC body in the tail must not be emitted as text: {context:?}"
        );
        assert_eq!(
            context
                .split("as the terminal drew them:\n")
                .nth(1)
                .and_then(|rest| rest.split("\n\nDetected").next()),
            Some("INFO starting\nRuntimeError: HIP out of memory"),
            "the tail must keep its text and its line structure, minus the escapes: {context:?}"
        );
        Ok(())
    }

    #[test]
    fn the_echoed_tail_never_welds_two_rendered_rows_into_one_line() -> Result<()> {
        // vLLM draws its progress bar with bare carriage returns, so a real OOM
        // arrives as ONE `\n` line carrying two rows the terminal drew. The echo
        // used to sanitize with `strip_terminal_control_sequences`, which drops
        // the break with nothing in its place and welds them together.
        //
        // That is not cosmetic. A managed launch shares one log between the
        // supervisor and the engine: the CLI writes this whole bail back into
        // the same file, so the *second* read sees the welded line, scores it as
        // a genuine OOM, finds it quotable (the control bytes are gone) and
        // quotes it into the `rocm diagnose --symptom '...'` command the serve
        // summary invites the user to paste. The user would be told to paste a
        // line vLLM never printed.
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let welded = format!("Loading weights: 90%{exception}");
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-weld-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, format!("Loading weights: 90%\r{exception}\n"))?;
        let context = startup_log_context(Some(path.as_path()));
        fs::remove_file(&path).ok();

        assert!(
            !context.contains(&welded),
            "the echoed tail must not weld the progress bar onto the exception: {context:?}"
        );
        assert!(
            context.contains(&format!("Loading weights: 90%\n{exception}")),
            "each row the terminal drew must be echoed on its own line: {context:?}"
        );
        // The displayed line beside the hint is the scoring row, not the segment
        // with its breaks deleted.
        assert!(
            context.contains(&format!(
                "{}{exception})",
                rocm_core::VLLM_OOM_DETECTED_PREFIX
            )),
            "the hint must name the exception row alone: {context:?}"
        );
        assert!(
            !context
                .chars()
                .any(|c| c.is_control() && c != '\n' && c != '\t'),
            "no cursor-moving control byte may survive into the printed failure message — \
             `\\t` is exempt, see the hint assertion above: {context:?}"
        );

        // Now the second read: rebuild the shared log exactly as the supervisor
        // leaves it (the raw engine output, then the CLI's `Error:` echo of this
        // bail) and confirm the selector still picks the engine's real row.
        let second_read = format!(
            "Loading weights: 90%\r{exception}\n\
             Error: vLLM server process exited before becoming ready (status: exit status: 1)\
             {context}\n"
        );
        assert_eq!(
            rocm_core::vllm_oom_diagnose_symptom(&second_read).as_deref(),
            Some(format!("vllm: {exception}").as_str()),
            "the welded echo must never become the symptom the user is told to paste"
        );
        Ok(())
    }

    /// Two branches `the_echoed_tail_never_welds…` cannot reach, because its
    /// fixture breaks on a bare `\r` and so its *selected segment* is already
    /// one row by the time either branch runs. Both survived a mutation run
    /// against the whole `process.rs` suite before this test existed:
    ///
    ///   - computing `oom_utilization_hint` from the sanitized summary instead
    ///     of the raw tail (103 passed, exit 0), and
    ///   - swapping `vllm_oom_display_line` for
    ///     `strip_terminal_control_sequences` (103 passed, exit 0).
    ///
    /// `ESC[2K` is the discriminator. [`rocm_core::terminal::rendered_lines`]
    /// ends a row on it, but `vllm_log_segments` does not split on it at all, so
    /// the segment reaching both branches really does carry two rows and an
    /// escape — which is what vLLM's colourised progress bar actually emits.
    #[test]
    fn the_hint_reads_the_raw_tail_and_displays_the_row_that_scored() -> Result<()> {
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let repaint = format!("Loading weights: 90%\u{1b}[2K{exception}");
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-repaint-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, format!("{repaint}\n"))?;
        let context = startup_log_context(Some(path.as_path()));
        fs::remove_file(&path).ok();

        // Branch 1 — the hint is computed from the *raw* tail. There the segment
        // still carries the `ESC[2K`, so the quoting guard rejects it and the
        // printed command names the canonical symptom. Computed from the
        // sanitized summary instead, the escape is gone, the guard accepts, and
        // the exception would be quoted verbatim.
        assert!(
            context.contains(&format!(
                "`{} '{}'`",
                rocm_core::VLLM_OOM_DIAGNOSE_COMMAND,
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM
            )),
            "an escape-bearing segment must fall back to the canonical symptom: {context:?}"
        );
        assert!(
            !context.contains(&format!("'vllm: {exception}'")),
            "the raw segment is not quotable, so its rendered row must not be quoted: {context:?}"
        );

        // Branch 2 — the displayed line is the row that scored. Stripping the
        // segment instead welds the progress bar onto the exception, producing a
        // line vLLM never printed.
        assert!(
            context.contains(&format!(
                "{}{exception})",
                rocm_core::VLLM_OOM_DETECTED_PREFIX
            )),
            "the hint must display the scoring row alone: {context:?}"
        );
        assert!(
            !context.contains(&format!("Loading weights: 90%{exception}")),
            "the displayed line must never weld the progress bar onto the exception: {context:?}"
        );
        Ok(())
    }

    /// The end-to-end bound on the raw-split guarantee, measured rather than
    /// assumed — and the reason the obvious tightening is not available.
    ///
    /// `vllm_log_segments` keeps raw bytes so a colourised OOM line is refused
    /// by the quoting guard rather than printed as a stripped lookalike. That
    /// holds for each surface on the text it reads. It does not survive the
    /// engine's own bail: the echo here is necessarily sanitized (it goes to a
    /// terminal), the supervisor appends the bail to the same shared log, and
    /// the raw line then falls out of the CLI's `DEFAULT_LOG_TAIL_LINES` window.
    /// So the two surfaces can print different `--symptom` arguments for one
    /// failure. This pins both halves: that the difference exists and is
    /// harmless, and that filtering the echo out would cost the note entirely.
    #[test]
    fn the_colourised_round_trip_quotes_the_row_the_terminal_drew() -> Result<()> {
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let coloured = format!("\u{1b}[31m{exception}\u{1b}[0m");
        let mut log = String::new();
        for shard in 0..200 {
            writeln!(log, "INFO vllm.engine loading shard {shard}").expect("write to String");
        }
        writeln!(log, "{coloured}").expect("write to String");
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-roundtrip-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, &log)?;

        // Read 1 — the engine. The raw segment carries escapes, so the guard
        // rejects it and the canonical symptom is printed.
        let context = startup_log_context(Some(path.as_path()));
        assert!(
            context.contains(&format!(
                "`{} '{}'`",
                rocm_core::VLLM_OOM_DIAGNOSE_COMMAND,
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM
            )),
            "the engine must refuse the colourised line and fall back: {context:?}"
        );

        // The supervisor appends the whole bail to the same log.
        fs::write(
            &path,
            format!(
                "{log}Error: vLLM server process exited before becoming ready \
                 (status: exit status: 1){context}\n"
            ),
        )?;
        let tail = rocm_core::tail_lines(
            &path,
            STARTUP_FAILURE_LOG_TAIL_LINES,
            crate::state::MAX_TAIL_READ,
        )?
        .join("\n");
        fs::remove_file(&path).ok();

        // The measurement the bound rests on: the engine's raw line is no longer
        // in the window the CLI reads. Only the sanitized echo of it is.
        assert!(
            !tail.contains(&coloured),
            "the raw colourised line must be out of the window, or this test proves nothing"
        );

        // Read 2 — the CLI. It quotes the rendered row, not the canonical
        // symptom. Different from read 1, and sound: the row is text the
        // terminal drew, it is quotable, and quotable implies diagnosable.
        let symptom = rocm_core::vllm_oom_diagnose_symptom(&tail)
            .expect("the echoed tail must still carry a detectable OOM");
        assert_eq!(symptom, format!("vllm: {exception}"));
        assert!(rocm_core::quotable_in_single_quotes(&symptom));
        assert!(rocm_core::vllm_oom_symptom_is_diagnosable(&symptom));

        // And the half that forecloses the tightening. The echo is a full tail
        // window plus the hint, so by the time the CLI reads, the window has
        // scrolled past the `Startup log, last N lines` header entirely: it holds
        // echoed rows and rocm-cli's own hint, and nothing the engine wrote
        // directly. A filter anchored on that header therefore never fires.
        let header = format!(
            "Startup log, last {STARTUP_FAILURE_LOG_TAIL_LINES} lines as the terminal drew them:"
        );
        assert!(
            !tail.contains(header.as_str()),
            "the echo header must have scrolled out, or the window still holds raw engine output"
        );
        // Anchored on the hint instead and scanning newest-first — the only
        // other marker available — suppression swallows the whole window and
        // the serve summary's note disappears for every real managed launch.
        let mut inside_echo = false;
        let still_selectable: Vec<&str> = tail
            .lines()
            .flat_map(|line| line.split('\r'))
            .map(str::trim)
            .filter(|segment| !segment.is_empty())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .filter(|segment| {
                if segment.contains(rocm_core::VLLM_OOM_DIAGNOSE_COMMAND) {
                    inside_echo = true;
                }
                !inside_echo
            })
            .collect();
        assert!(
            still_selectable
                .iter()
                .all(|segment| rocm_core::vllm_oom_diagnose_symptom(segment).is_none()),
            "keeping the echo out of the CLI's selection leaves no OOM line at all: \
             {still_selectable:?}"
        );
        Ok(())
    }

    /// Pins a **known gap**, not desired behaviour: when the OOM line sits near
    /// the start of the window this engine scans, the `rocm serve` summary's
    /// note disappears even though this engine diagnosed the OOM.
    ///
    /// Both surfaces use the same tail *budget*
    /// (`rocm_engine_protocol::DEFAULT_LOG_TAIL_LINES`), but not the same
    /// window. This engine scans the last N raw lines and bails; the supervisor
    /// appends that whole bail — a header, the tail echoed as rendered rows, and
    /// the hint — to the same shared log; the CLI then reads N lines of the
    /// *longer* file. So the CLI's window always reaches less far back into the
    /// original log than this one did, and when the bail is itself longer than N
    /// rows it holds nothing the engine wrote directly. With the OOM line
    /// followed by 77 further lines it is diagnosed here and invisible there.
    ///
    /// Closing it is a design decision rather than a tweak, which is why it is
    /// pinned instead of fixed. Re-windowing the CLI to the N lines *before* the
    /// bail would restore agreement, but the raw colourised line comes back into
    /// view with it and the note stops quoting the user's own rendered row —
    /// trading this gap for the regression
    /// `the_colourised_round_trip_quotes_the_row_the_terminal_drew` exists to
    /// prevent. Recording the engine's selected symptom as structured data
    /// alongside the log would remove the class. **Update this test when the gap
    /// closes; do not delete it.**
    #[test]
    fn the_cli_window_can_miss_an_oom_the_engine_saw() -> Result<()> {
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let mut log = String::new();
        for shard in 0..200 {
            writeln!(log, "INFO vllm.engine loading shard {shard}").expect("write to String");
        }
        writeln!(log, "{exception}").expect("write to String");
        // Enough trailing lines that the OOM still sits inside this engine's
        // window (78 lines from the end, budget 80) but far enough from the end
        // that the appended bail pushes its echo out of the CLI's.
        for worker in 0..77 {
            writeln!(log, "INFO vllm.engine shutting down worker {worker}")
                .expect("write to String");
        }
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-window-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, &log)?;

        // Read 1 — the engine. It sees the OOM and says so.
        let context = startup_log_context(Some(path.as_path()));
        assert!(
            context.contains(rocm_core::VLLM_OOM_DETECTED_PREFIX),
            "the engine must diagnose this OOM, or the test proves nothing: {context:?}"
        );

        // The supervisor appends the whole bail to the same log.
        fs::write(
            &path,
            format!(
                "{log}Error: vLLM server process exited before becoming ready \
                 (status: exit status: 1){context}\n"
            ),
        )?;
        let tail = rocm_core::tail_lines(
            &path,
            STARTUP_FAILURE_LOG_TAIL_LINES,
            crate::state::MAX_TAIL_READ,
        )?
        .join("\n");
        fs::remove_file(&path).ok();

        // Read 2 — the CLI, same budget, later file. The gap.
        assert!(
            rocm_core::vllm_oom_diagnose_symptom(&tail).is_none(),
            "this test pins the known window gap; if the CLI now finds the OOM the gap is \
             closed — update the test, `append_oom_serve_note`'s comment and the README's \
             \"absence is not evidence\" list together: {tail:?}"
        );
        Ok(())
    }

    #[test]
    fn oom_utilization_hint_stays_quiet_for_unrelated_failures() {
        let unrelated = "ValueError: model architecture 'FooForCausalLM' is not supported";
        assert!(!detects_oom(unrelated));
        assert!(
            oom_utilization_hint(unrelated).is_empty(),
            "non-OOM failures must not carry a memory hint"
        );

        // vLLM's generic EngineCore wrapper is the terminal line for *any*
        // startup crash (unsupported arch, shm size, TP misconfig, missing
        // weights, OOM, ...); treating it as an OOM signature would misreport
        // those unrelated failures as memory exhaustion.
        let wrapper_only = "ERROR Engine core initialization failed";
        assert!(!detects_oom(wrapper_only));
        assert!(
            oom_utilization_hint(wrapper_only).is_empty(),
            "the generic EngineCore wrapper alone must not be treated as OOM"
        );
    }

    // ── The engine import failure: a pointer at `rocm diagnose` ────────────────

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
            context.contains("Startup log, last 80 lines as the terminal drew them:"),
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
