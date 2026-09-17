// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! One ECMA-48 walk over untrusted terminal output, shared by the two callers
//! that need it.
//!
//! Both callers are handling the same input — a raw terminal capture from a
//! subprocess, pasted or quoted back at the user — and both need the same
//! question answered: *what would a terminal have rendered here?* They use the
//! answer differently.
//!
//! * The vLLM engine strips the sequences out, so a colourised or bell-bearing
//!   log line cannot repaint the user's terminal from inside rocm-cli's own
//!   error message ([`strip_terminal_control_sequences`]).
//! * The vLLM-OOM diagnostic needs the *line boundaries*, so that two things the
//!   terminal drew on separate rows are not scored as one line
//!   ([`rendered_lines`]) — with the single exception documented below.
//!
//! A second, independent scan for the second caller would have been a second
//! grammar to get wrong, and the first one took three rounds to get right. So
//! the grammar lives here once, in one private stepping function, and each
//! caller interprets the classified tokens it yields.
//!
//! # The one exception to the never-merge property
//!
//! [`rendered_lines`] is an over-approximation of what a terminal would have
//! drawn: it may split one rendered line in two, or lose one altogether, and in
//! the other direction it merges two rendered rows in exactly one shape. A line
//! break *inside* the body of a string-argument sequence (`OSC`, `DCS`, `SOS`,
//! `PM`, `APC`) is consumed with that body, so the text either side of the
//! sequence joins up:
//!
//! ```text
//! rendered_lines("vllm\u{1b}]0;ti\ntle\u{7}llama.cpp OOM")
//!     == ["vllmllama.cpp OOM"]
//! rendered_lines("vllm\u{1b}]0;ti\ntle\u{1b}[0mllama.cpp OOM")
//!     == ["vllmllama.cpp OOM"]
//! ```
//!
//! A terminator is not what creates the shape, as the second example shows: it
//! contains no `BEL` and no `ST` anywhere, and the two rows still merge. The
//! body scan runs to the next `BEL`, to `ST` (`ESC \`), to an unrelated `ESC`
//! — which it leaves in place for the next step to re-read, so the hole stays
//! one sequence wide — or to the end of the input, and the break is swallowed
//! in every one of those cases, terminated or not.
//!
//! Swallowing the break is not always a *merge*, though. A merge needs drawable
//! text on both sides of the break landing in one segment, so what decides is
//! whether drawable text follows the sequence in that same segment — however the
//! body ended. Where none does there is no far side to join to, and what the
//! terminal drew past the break comes back in no segment at all:
//!
//! ```text
//! rendered_lines("vllm\u{1b}]0;ti\ntle")          == ["vllm"]
//! rendered_lines("vllm\u{1b}]0;ti\ntle\u{7}")     == ["vllm"]
//! rendered_lines("vllm\u{1b}]0;ti\ntle\u{7}\nx")  == ["vllm", "x"]
//! ```
//!
//! The last two bodies are properly `BEL`-terminated, and the last is not near
//! the end of the input either; the `tle` is lost all the same.
//!
//! That is a loss, not a misattribution, so it belongs with the splitting
//! direction where nothing is promised: it costs a diagnosis the tool would
//! otherwise have made, and the canonical-symptom fallback covers it.
//!
//! Common terminals abort a control string on an embedded C0 byte and would
//! draw two rows there, so this is a real divergence rather than a technicality.
//! Nor does it take crafted input to reach, which an earlier wording of this
//! section claimed on the strength of a terminator being required. What it takes
//! is an *interleaved* capture: `rocm serve` hands the subprocess's stdout and
//! stderr the *same* file handle, so a title written on one stream can be cut by
//! the other stream's next line landing between the introducer and the
//! terminator, with neither writer doing anything unusual. A process killed
//! part-way through the same title is the other ordinary half of it, leaving a
//! body with no terminator at all — and killed output is the case this walk is
//! built for.
//!
//! Such a capture reaches [`rendered_lines`] whole when a user pastes it into
//! `rocm diagnose --symptom`. The vLLM engine's own path cannot carry an
//! embedded `\n` there: it pre-splits its log tail with `str::lines` and builds
//! its candidate symptom from a single line of it.
//!
//! Following the grammar is still the right call. The alternative is guessing
//! where an unterminated body ends, and the guess would have to be made on
//! exactly the truncated, interleaved output that is the normal case here. Guess
//! short and an `OSC` body is emitted as text — the leak this walk exists to
//! stop, and a second, quieter source of the very misattribution the line
//! boundaries are for, since a window title would then be scored as if the
//! process had printed it. The residual is therefore disclosed here rather than
//! papered over: a break inside a string body is the one way two rendered rows
//! come back as one segment, and — where no drawable text follows the sequence
//! in that same segment — the one way a rendered row comes back in none.
//!
//! This section is the single authoritative statement of the exception. The
//! code that creates it and the test that pins it both point here rather than
//! restating it, so that a change to the behaviour cannot leave a stale copy of
//! the guarantee behind in a doc a consumer reads.

/// What one step of the ECMA-48 walk found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Token {
    /// A character the terminal draws at the cursor.
    Text(char),
    /// Something that can move the cursor off the current row, so text either
    /// side of it was rendered on different lines.
    LineBreak,
    /// Something that draws nothing and cannot move the cursor, so it neither
    /// contributes text nor breaks a line.
    Ignorable,
}

/// Whether `c` is a character that carries no glyph of its own but changes how
/// the text around it renders.
///
/// `char::is_control` is Unicode category `Cc` only, so it misses the `Cf`
/// format characters — and `U+202E RIGHT-TO-LEFT OVERRIDE` in a log line
/// reverses how the rest of the printed `rocm diagnose --symptom '...'` command
/// renders in the user's terminal. That cannot escape the single quotes (no
/// ASCII `'` is involved), so it is display spoofing rather than shell
/// injection, but it is the same "untrusted subprocess output must not control
/// what the terminal shows" concern the escape stripping exists for, and the
/// answer has to be the same.
///
/// The ranges are the `Cf` category, enumerated rather than pulled from a
/// Unicode-tables dependency: the set is small, stable, and a new dependency for
/// one predicate is a worse trade. Over-inclusion is safe here — the only cost
/// of rejecting a character is falling back to the canonical symptom, which is
/// guaranteed to report a cause.
#[must_use]
pub fn is_control_or_format(c: char) -> bool {
    c.is_control()
        || matches!(c,
            '\u{00ad}'
            | '\u{0600}'..='\u{0605}'
            | '\u{061c}'
            | '\u{06dd}'
            | '\u{070f}'
            | '\u{0890}'..='\u{0891}'
            | '\u{08e2}'
            | '\u{180e}'
            | '\u{200b}'..='\u{200f}'
            | '\u{202a}'..='\u{202e}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206f}'
            | '\u{feff}'
            | '\u{fff9}'..='\u{fffb}'
            | '\u{110bd}'
            | '\u{110cd}'
            | '\u{13430}'..='\u{1343f}'
            | '\u{1bca0}'..='\u{1bca3}'
            | '\u{1d173}'..='\u{1d17a}'
            | '\u{e0001}'
            | '\u{e0020}'..='\u{e007f}')
}

/// Consumes one glyph, control character or escape sequence and says which of
/// the three it was.
///
/// The sequence grammar is ECMA-48's, followed exactly, and the reason is that
/// the input is a *killed* process's output: truncated and interleaved escapes
/// are the normal case here, not an exotic one. A scan that just ran to the
/// next byte in `0x40..=0x7E` mis-handled every one of them — it swallowed the
/// following sequence's introducer (`ESC [ 1 ; 2 ESC [ 0 m Killed` lost
/// `Killed`'s first six characters), ran straight past a multi-byte scalar
/// (which can never be in that ASCII range) and ate everything up to the next
/// byte that happened to land in it, emitted the body of an OSC title
/// sequence as text, and left a stray `ESC ESC` unrecoverable.
///
/// Consuming only what the grammar allows and then *stopping without consuming*
/// the offending byte bounds the damage to the malformed sequence itself: the
/// text after it survives, and a following well-formed sequence is still
/// recognised because its `ESC` is left for the main loop to re-read.
///
/// What this deliberately does not do is second-guess a well-formed sequence.
/// `ESC [ SP K` is a valid CSI (`SP` is an intermediate, `K` the final byte), so
/// it is consumed whole even though the `K` may have been the first letter of a
/// truncated process's "Killed": a well-formed sequence is consumed whole
/// because a real terminal consumes it too. That rule settles *how much* of the
/// byte stream one step eats, and only that — which [`Token`] the consumed
/// sequence is then labelled with is a separate question, answered below, and
/// answered deliberately unlike a terminal.
///
/// # Why [`Token::LineBreak`] is the default for anything unrecognised
///
/// The classification is deliberately lopsided, because the two ways of being
/// wrong do not cost the same. Calling a real boundary [`Token::Ignorable`]
/// merges two rendered lines into one, which is what lets a `llama.cpp` OOM be
/// scored against a `vllm` anchor from a different line — a confident, wrong
/// answer. Calling a non-boundary a [`Token::LineBreak`] splits one rendered
/// line in two, which at worst loses a diagnosis the tool would otherwise have
/// made — a silent miss, and one the canonical-symptom fallback already covers.
///
/// So only two classes are [`Token::Ignorable`], and only because neither can
/// move the cursor at all:
///
/// * `SGR` (`CSI ... m`) — colour and attributes, and the overwhelmingly common
///   thing to find *inside* a real log line. This is the one exception that has
///   to be right: treating it as a boundary would split
///   `ESC[31m vllm: ESC[1m torch.OutOfMemoryError` between its anchor and its
///   error token and lose a genuine OOM.
/// * The string-argument sequences (`OSC`, `DCS`, `SOS`, `PM`, `APC`) — their
///   bodies are arbitrary text, like a window title, that is never drawn at the
///   cursor. Consuming such a body whole is the one exception to the never-merge
///   property, stated in full in the module documentation because
///   [`rendered_lines`]'s public callers are the ones it binds.
///
/// Everything else with an `ESC` in it — `ESC E` (`NEL`), `ESC D` (`IND`),
/// `CSI n B` (`CUD`), and equally the sequences that do *not* move the cursor,
/// such as a charset designator — is a boundary. Enumerating the cursor-moving
/// finals exactly would be a terminal emulator, and a terminal emulator is not
/// decidable on a byte stream with no screen model: `CSI H` addresses a row that
/// depends on what came before, and overwriting text with `\r` makes "which line
/// was this on" a question about the whole capture. Over-splitting needs none of
/// that and cannot produce the misattribution.
fn next_token(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Option<Token> {
    let c = chars.next()?;
    if c != '\u{1b}' {
        return Some(classify_char(c));
    }
    match chars.peek() {
        // CSI — what a colourised logger emits.
        Some('[') => {
            chars.next();
            Some(if skip_csi_body(chars) {
                Token::Ignorable
            } else {
                Token::LineBreak
            })
        }
        // The string-argument sequences: OSC, DCS, SOS, PM, APC. Their bodies
        // are arbitrary text (a window title, say) and must not be emitted as if
        // the process had printed it, but they draw nothing at the cursor.
        //
        // Consuming the body whole is also the one hole in the never-merge
        // property, since a line break inside the body goes with it — and the
        // scan below ends on a bare `ESC` or on end of input as readily as on a
        // terminator, so a terminator is not what creates the hole.
        // The module documentation is where that exception is stated, and it is
        // stated there rather than here because a body comment in a private
        // function reaches no reader of the public docs. Any change to what this
        // arm consumes has to change that section, `rendered_lines`'s own
        // qualifier, and the test that pins the worked examples, in one commit.
        Some(']' | 'P' | 'X' | '^' | '_') => {
            chars.next();
            skip_string_sequence_body(chars);
            Some(Token::Ignorable)
        }
        // `ESC` with nothing after it, or `ESC ESC`: drop just this one and let
        // the loop re-read the next as a fresh introducer. A bare `ESC` draws
        // nothing and moves nothing on its own.
        None | Some('\u{1b}') => Some(Token::Ignorable),
        // Any other escape: optional intermediates, then one final byte. This is
        // where `ESC E` (NEL) and `ESC D` (IND) live.
        Some(_) => {
            skip_simple_escape_body(chars);
            Some(Token::LineBreak)
        }
    }
}

/// Classifies a character that is not an escape introducer.
///
/// `\t` is [`Token::Text`] rather than a boundary: it moves the cursor along the
/// row it is already on, so it is legitimate intra-line whitespace. Every other
/// `Cc` control is a boundary — `\n` and `\r` obviously, but equally `\x0b`
/// (`VT`) and `\x0c` (`FF`), which advance a line, `\u{85}` (`NEL`), and the
/// bytes that do nothing at all. The last group is the lopsidedness above: a
/// stray `\x01` between two rendered lines is far likelier to be a mangled line
/// advance than intra-line text, and guessing "boundary" costs a missed
/// diagnosis where guessing "text" costs a wrong one.
///
/// `U+2028`/`U+2029` are `Zl`/`Zp` rather than `Cc`, so `char::is_control` does
/// not cover them, but Unicode defines both as mandatory line breaks.
///
/// Those two scalars are the one place where moving this walk out of the vLLM
/// engine changed [`strip_terminal_control_sequences`] rather than merely
/// relocating it, and the commit that moved it says "behaviour is unchanged",
/// which is true of everything except this. The engine-local stripper tested
/// every non-escape character with [`is_control_or_format`] alone; that is
/// `false` for both (neither is `Cc`, neither is in the enumerated `Cf` set), so
/// both used to survive into the stripped message and now do not. The widening
/// is intentional — a mandatory line break is exactly the kind of non-drawing
/// character that stripper exists to remove — and the stripper table test in
/// `engines/vllm/src/lib.rs` pins both scalars so the next drift is caught.
fn classify_char(c: char) -> Token {
    if c == '\t' {
        return Token::Text(c);
    }
    if c.is_control() || matches!(c, '\u{2028}' | '\u{2029}') {
        return Token::LineBreak;
    }
    if is_control_or_format(c) {
        return Token::Ignorable;
    }
    Token::Text(c)
}

/// Consumes a CSI body — parameter bytes, then intermediate bytes, then one
/// final byte — from just after the `ESC [`. Stops without consuming anything
/// that does not belong, leaving it to be treated as text.
///
/// Returns whether the sequence was a well-formed `SGR` (final byte `m`), the
/// only CSI that cannot move the cursor. A CSI truncated before its final byte
/// returns `false`: an unterminated sequence is exactly the mangled output the
/// boundary question matters for, so it takes the conservative branch.
fn skip_csi_body(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> bool {
    while chars
        .next_if(|c| ('\u{30}'..='\u{3f}').contains(c))
        .is_some()
    {}
    while chars
        .next_if(|c| ('\u{20}'..='\u{2f}').contains(c))
        .is_some()
    {}
    chars.next_if(|c| ('\u{40}'..='\u{7e}').contains(c)) == Some('m')
}

/// Consumes a string-argument sequence's body and terminator (`BEL`, or `ST` =
/// `ESC \`) from just after the introducer. A body truncated by anything else —
/// including a bare `ESC` starting the next sequence — ends the scan with that
/// byte left in place.
fn skip_string_sequence_body(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    while let Some(&c) = chars.peek() {
        if c == '\u{7}' {
            chars.next();
            return;
        }
        if c == '\u{1b}' {
            let mut lookahead = chars.clone();
            lookahead.next();
            if lookahead.peek() == Some(&'\\') {
                chars.next();
                chars.next();
            }
            return;
        }
        chars.next();
    }
}

/// Consumes a non-CSI escape's body — optional intermediates, then one final
/// byte — from just after the `ESC`.
fn skip_simple_escape_body(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    while chars
        .next_if(|c| ('\u{20}'..='\u{2f}').contains(c))
        .is_some()
    {}
    chars.next_if(|c| ('\u{30}'..='\u{7e}').contains(c));
}

/// Removes ANSI escape sequences and any remaining control or format character.
///
/// A colourised or bell-bearing log line therefore cannot repaint the user's
/// terminal, or reorder how the message renders, from inside rocm-cli's own
/// error message.
///
/// Line boundaries are dropped rather than preserved: every caller here has
/// already split its input into lines and wants a single-line result.
#[must_use]
pub fn strip_terminal_control_sequences(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(token) = next_token(&mut chars) {
        if let Token::Text(c) = token {
            // `\t` is intra-line text to the boundary question but still a
            // control byte in a message this crate is about to print, and the
            // callers' contract is that nothing control-ish survives.
            if !is_control_or_format(c) {
                out.push(c);
            }
        }
    }
    out
}

/// Approximates the lines a terminal would have rendered `text` as, with the
/// escape sequences and non-drawing characters removed from each segment.
///
/// It is deliberately an *over*-approximation rather than a terminal emulator,
/// and the approximation is one-sided. Nothing at all is promised in the losing
/// direction: one rendered line may come back split in two, and text the
/// terminal drew may come back in no segment at all.
///
/// In the merging direction the promise holds with exactly one exception. Two
/// pieces of text the terminal drew on different rows share a returned segment
/// only when the break between them sits inside the body of a string-argument
/// sequence (`OSC`, `DCS`, `SOS`, `PM`, `APC`), which is consumed whole. See
/// [the module documentation](self) for the worked examples, for which ordinary
/// captures reach that shape, for why the exception is not worth closing
/// anyway, and for exactly which sequences are treated as boundaries.
///
/// Empty segments are returned as-is; callers that filter their lines drop them
/// for free.
#[must_use]
pub fn rendered_lines(text: &str) -> Vec<String> {
    let mut lines = vec![String::new()];
    let mut chars = text.chars().peekable();
    while let Some(token) = next_token(&mut chars) {
        match token {
            Token::Text(c) => {
                if let Some(last) = lines.last_mut() {
                    last.push(c);
                }
            }
            Token::LineBreak => lines.push(String::new()),
            Token::Ignorable => {}
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_advance_of_any_form_is_a_boundary_and_sgr_is_not() {
        // The table is the reviewer's reproduction on PR #251, generalised: the
        // fix that landed before it split on `['\n', '\r']`, an allowlist of two
        // characters behind a doc comment claiming the boundary was "the one the
        // terminal actually renders". Every other way a terminal is told to
        // start a new line collapsed the paste back into one line.
        //
        // `sgr` is the control in the other direction, and the reason this is a
        // grammar walk rather than the one-line `char::is_control` split that
        // was suggested: splitting at `ESC` fixes the rows below but cuts a
        // genuine colourised log line between its anchor and its error token.
        let cases: &[(&str, &str, &[&str])] = &[
            ("lf", "a\nb", &["a", "b"]),
            ("cr", "a\rb", &["a", "b"]),
            ("crlf", "a\r\nb", &["a", "", "b"]),
            ("nel esc-e", "a\u{1b}Eb", &["a", "b"]),
            ("ind esc-d", "a\u{1b}Db", &["a", "b"]),
            ("ri esc-m", "a\u{1b}Mb", &["a", "b"]),
            ("cud csi-1-b", "a\u{1b}[1Bb", &["a", "b"]),
            ("cuu csi-a", "a\u{1b}[Ab", &["a", "b"]),
            ("cup csi-h", "a\u{1b}[2;3Hb", &["a", "b"]),
            ("el csi-k", "a\u{1b}[Kb", &["a", "b"]),
            ("vt", "a\u{b}b", &["a", "b"]),
            ("ff", "a\u{c}b", &["a", "b"]),
            ("c1 nel", "a\u{85}b", &["a", "b"]),
            ("u2028 line sep", "a\u{2028}b", &["a", "b"]),
            ("u2029 para sep", "a\u{2029}b", &["a", "b"]),
            ("stray c0", "a\u{1}b", &["a", "b"]),
            ("bel", "a\u{7}b", &["a", "b"]),
            // `ESC [ 1 ; 2 b` is *complete*, not truncated -- `b` is a valid CSI
            // final byte (`REP`), so the grammar consumes it and it is not text.
            // A genuinely truncated CSI is one the next `ESC` interrupts; it
            // must not swallow that introducer, so the SGR after it still reads
            // as an SGR and only the truncated sequence is a boundary.
            ("csi with a letter final", "a\u{1b}[1;2b", &["a", ""]),
            ("truncated csi", "a\u{1b}[1;2\u{1b}[0mb", &["a", "b"]),
            // Not boundaries.
            ("sgr", "a\u{1b}[31mb", &["ab"]),
            ("sgr reset", "a\u{1b}[0mb", &["ab"]),
            ("sgr multi param", "a\u{1b}[1;38;5;9mb", &["ab"]),
            ("osc title", "a\u{1b}]0;title\u{7}b", &["ab"]),
            ("dcs", "a\u{1b}Pq\u{1b}\\b", &["ab"]),
            ("tab is intra-line", "a\tb", &["a\tb"]),
            ("zero-width format char", "a\u{200b}b", &["ab"]),
            ("bidi override", "a\u{202e}b", &["ab"]),
            ("bare esc", "a\u{1b}", &["a"]),
        ];
        for (label, input, want) in cases {
            let got = rendered_lines(input);
            assert_eq!(
                &got.iter().map(String::as_str).collect::<Vec<_>>(),
                want,
                "{label}"
            );
        }
    }

    #[test]
    fn a_lone_non_drawing_scalar_never_merges_two_rendered_lines() {
        // The property the over-approximation actually promises, over every
        // non-drawing scalar that could sit between two lines of a capture: the
        // whole C0 range, DEL, the whole C1 range, and the Unicode separators
        // and format characters. The printable range is excluded because those
        // are text by definition and merging is what they are for.
        //
        // The expectation is derived from `char::is_control` (std) and the
        // enumerated `Cf` set rather than from this module's own classifier, so
        // a change to `classify_char` cannot move the goalposts with it.
        for c in (0u32..=0x1f)
            .chain(0x7f..=0x9f)
            .filter_map(char::from_u32)
            .chain(['\u{2028}', '\u{2029}', '\u{200b}', '\u{feff}', '\u{202e}'])
        {
            let joined = format!("left{c}right");
            let lines = rendered_lines(&joined);
            let merged = lines
                .iter()
                .any(|l| l.contains("left") && l.contains("right"));
            // Exactly the characters a terminal draws on the same row may merge:
            // tab, and the non-drawing format characters that move no cursor.
            let may_merge = c == '\t' || (!c.is_control() && is_control_or_format(c));
            assert_eq!(merged, may_merge, "U+{:04X}", c as u32);
        }
    }

    #[test]
    fn a_line_break_inside_a_string_sequence_body_is_consumed_with_it() {
        // The one exception to the never-merge property, and the worked examples
        // the module documentation prints. They are pinned here so the prose and
        // the behaviour move together: if either expectation ever changes, the
        // module section and `rendered_lines`'s own qualifier are wrong and
        // have to change in the same commit. The sweep above cannot catch this
        // -- it feeds one scalar at a time, and this shape needs a
        // string-sequence introducer before the break.
        assert_eq!(
            rendered_lines("vllm\u{1b}]0;ti\ntle\u{7}llama.cpp OOM"),
            ["vllmllama.cpp OOM"],
        );
        // The same break outside a string body still splits, so the exception is
        // the sequence's framing and not the newline losing its meaning.
        assert_eq!(
            rendered_lines("vllm\u{1b}]0;title\u{7}\nllama.cpp OOM"),
            ["vllm", "llama.cpp OOM"],
        );
        // And an *unterminated* body is the same merge over again, not a
        // separate containment case: there is no `BEL` and no `ST` anywhere
        // below, and the break is still swallowed, so a terminator is not what
        // creates the exception. What the following `ESC` bounds is only how far
        // the body runs -- it is left in place -- so the hole does not widen to
        // swallow the rest of a capture.
        assert_eq!(
            rendered_lines("vllm\u{1b}]0;ti\ntle\u{1b}[0mllama.cpp OOM"),
            ["vllmllama.cpp OOM"],
        );
        // Swallowing the break merges only where drawable text follows the
        // sequence in the same segment. Where none does, the `tle` a terminal
        // would have drawn on its second row comes back in no segment at all --
        // and how the body ended is not what decides that. The second case below
        // is properly `BEL`-terminated and the third is terminated with more of
        // the capture still to come, only behind a break; both lose the row just
        // as the unterminated first one does. That is the losing direction, not
        // the misattributing one, and it is why the module section documents a
        // third behaviour rather than two. Pinned because prose has three times
        // now named a narrower trigger than the code has.
        assert_eq!(rendered_lines("vllm\u{1b}]0;ti\ntle"), ["vllm"]);
        assert_eq!(rendered_lines("vllm\u{1b}]0;ti\ntle\u{7}"), ["vllm"]);
        assert_eq!(
            rendered_lines("vllm\u{1b}]0;ti\ntle\u{7}\nx"),
            ["vllm", "x"]
        );
    }
}
