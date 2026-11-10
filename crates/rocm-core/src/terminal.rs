// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What rocm-cli does with untrusted terminal output: one ECMA-48 walk, and the
//! decisions that rest on it.
//!
//! Every caller here is handling the same input — a raw terminal capture from a
//! subprocess, pasted or quoted back at the user — and needs the same question
//! answered: *what would a terminal have rendered here?* They use the answer
//! differently.
//!
//! * The vLLM engine strips the sequences out, so a colourised or bell-bearing
//!   log line cannot repaint the user's terminal from inside rocm-cli's own
//!   error message ([`strip_terminal_control_sequences`]).
//! * The vLLM-OOM diagnostic needs the *line boundaries*, so that two things the
//!   terminal drew on separate rows are not scored as one line
//!   ([`rendered_lines`]) — with the single exception documented below.
//! * The surfaces that render such a line back inside a command the user is
//!   invited to paste reject it rather than rewrite it
//!   ([`quotable_in_single_quotes`]).
//! * `rocm chat` prints a model's reply, tool output and error text, which are
//!   untrusted too and span many lines; [`sanitize_terminal_text`] removes the
//!   sequences from them but keeps the line structure and shows any concealed
//!   text. Values a model chose for a command line are refused instead, for any
//!   [`is_display_altering`] character.
//!
//! A second, independent scan for the second caller would have been a second
//! grammar to get wrong, and the first one took three rounds to get right. So
//! the grammar lives here once, in one private stepping function, and each
//! caller interprets the classified tokens it yields. The third needs no walk
//! at all, only a character predicate ([`is_display_altering`]) that takes in
//! the ones the other two end on — which is why it lives here rather than
//! beside either of its two callers.
//!
//! # Line breaks inside a string body: merging and losing rows
//!
//! [`rendered_lines`] is an over-approximation of what a terminal would have
//! drawn: it may split one rendered line in two, or lose one or more
//! altogether, and in the other direction it merges two rendered rows in
//! exactly one shape. A line break *inside* the body of a string-argument
//! sequence (`OSC`, `DCS`, `SOS`, `PM`, `APC`) is consumed with that body, so
//! the text either side of the sequence joins up:
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
//! body scan runs to the next `BEL`, to `ST` (`ESC \` or U+009C), to an unrelated `ESC`
//! — which it leaves in place for the next step to re-read, so the hole stays
//! one sequence wide — or to the end of the input, and the break is swallowed
//! in every one of those cases, terminated or not.
//!
//! Swallowing the break is not always a *merge*, though. A merge needs drawable
//! text on both sides of the break landing in one segment — however the body
//! ended. Where either side is missing, what the terminal drew past the break
//! comes back in no segment at all:
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
//! draw both sides of the break, so this is a real divergence rather than a
//! technicality. Nor does it take crafted input to reach, which an earlier
//! wording of this section claimed on the strength of a terminator being
//! required. What it takes is an *interleaved* capture: `rocm serve` hands the
//! subprocess's stdout and stderr the *same* file handle, so a title written on
//! one stream can be cut by the other stream's next line landing between the
//! introducer and the terminator, with neither writer doing anything unusual.
//! A process killed part-way through the same title is the other ordinary half
//! of it, leaving a body with no terminator at all — and killed output is the
//! case this walk is built for.
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
//! come back as one segment, and the one way a rendered row comes back in none.
//!
//! This section states the rule normatively; the comments on the code that
//! creates it and on the test that pins it are local commentary, not the
//! statement of record. A change to the behaviour has to land here, and any
//! comment that has drifted from this section has to be brought back into line.

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

/// Whether `c` is a `Cc` control or one of the two Unicode separators that
/// mandate a line break.
///
/// `U+2028 LINE SEPARATOR` and `U+2029 PARAGRAPH SEPARATOR` are `Zl`/`Zp`, so
/// neither `char::is_control` nor [`is_control_or_format`]'s enumerated `Cf`
/// set covers them, yet a terminal draws the text after either on the next row.
/// This module's classifier ends a line on it, and [`is_display_altering`]
/// includes it, so [`quotable_in_single_quotes`], the guard on what may be
/// quoted into the `rocm diagnose --symptom '...'` command the OOM surfaces
/// print, refuses a line separator too.
#[must_use]
pub fn is_control_or_line_separator(c: char) -> bool {
    c.is_control() || matches!(c, '\u{2028}' | '\u{2029}')
}

/// Whether `c` has Unicode's `Default_Ignorable_Code_Point` property.
///
/// Those are the code points (`DerivedCoreProperties.txt`) a renderer shows
/// nothing for when it has no special handling for them, which is how a
/// terminal treats them.
///
/// Written out from the property rather than picked by hand, and wider than
/// [`is_control_or_format`]'s `Cf` set: the Hangul fillers (`Lo`) render as
/// blank space, the variation selectors (`Mn`) and the combining grapheme
/// joiner draw nothing, and the tag block holds a run that can spell out
/// arbitrary ASCII with no glyph at all. A value carrying any of them looks
/// identical to one without.
#[must_use]
pub const fn is_default_ignorable(c: char) -> bool {
    matches!(c,
        '\u{00ad}'                      // soft hyphen
        | '\u{034f}'                    // combining grapheme joiner
        | '\u{061c}'                    // Arabic letter mark
        | '\u{115f}'..='\u{1160}'       // Hangul choseong and jungseong fillers
        | '\u{17b4}'..='\u{17b5}'       // Khmer inherent vowels
        | '\u{180b}'..='\u{180f}'       // Mongolian variation selectors, vowel separator
        | '\u{200b}'..='\u{200f}'       // zero-width space and joiners, LRM, RLM
        | '\u{202a}'..='\u{202e}'       // bidi embeddings and overrides
        | '\u{2060}'..='\u{206f}'       // word joiner, invisible operators, bidi isolates
        | '\u{3164}'                    // Hangul filler
        | '\u{fe00}'..='\u{fe0f}'       // variation selectors
        | '\u{feff}'                    // zero-width no-break space (byte-order mark)
        | '\u{ffa0}'                    // halfwidth Hangul filler
        | '\u{fff0}'..='\u{fff8}'       // unassigned specials
        | '\u{1bca0}'..='\u{1bca3}'     // shorthand format controls
        | '\u{1d173}'..='\u{1d17a}'     // musical symbol format controls
        | '\u{e0000}'..='\u{e0fff}') // tags, variation selectors supplement, reserved
}

/// Whether `c` changes how text is displayed rather than being displayed:
/// a control, a `Cf` format character, a line or paragraph separator, or any
/// other default-ignorable code point.
///
/// The union of [`is_control_or_format`], [`is_control_or_line_separator`] and
/// [`is_default_ignorable`]: what [`sanitize_terminal_text`] removes from prose
/// apart from the layout characters and the single glyph-shaping characters it
/// keeps, and what [`hides_or_moves_text`] refuses in a value on the same terms.
#[must_use]
pub fn is_display_altering(c: char) -> bool {
    is_control_or_format(c) || is_control_or_line_separator(c) || is_default_ignorable(c)
}

/// The four invisible characters that may stay in printed text: the
/// zero-width non-joiner and joiner, and the text and emoji presentation
/// selectors (VS15, VS16). They pick how the visible character before them is
/// drawn — joining an emoji sequence, choosing the emoji form of `❤` — and
/// move nothing. The other variation selectors (VS1–VS14, the supplement at
/// U+E0100, the Mongolian ones) are not kept: almost no terminal text needs
/// them, and 256 of them can spell a payload with no glyph.
const fn shapes_the_preceding_glyph(c: char) -> bool {
    matches!(c, '\u{200c}' | '\u{200d}' | '\u{fe0e}' | '\u{fe0f}')
}

/// Whether `c`, coming after `previous`, is one of the glyph-shaping
/// characters kept in printed text: [`shapes_the_preceding_glyph`], directly
/// after a character that is itself drawn. A second one in a row, or one at
/// the start of the text or after a line break, shapes nothing and is dropped,
/// so a run of them cannot carry hidden text. One rule for prose and for
/// values, so a value is refused for exactly what printing would remove.
fn is_kept_glyph_shaper(previous: Option<char>, c: char) -> bool {
    shapes_the_preceding_glyph(c) && previous.is_some_and(|previous| !is_display_altering(previous))
}

/// Whether `value` holds anything a terminal would not show as it is.
///
/// That is a control (newline and tab included), an escape sequence, a format
/// character, a line separator, or a default-ignorable character other than a
/// single glyph-shaping one after a drawn character ([`is_kept_glyph_shaper`]).
///
/// That is exactly the set [`sanitize_terminal_text`] would remove or change,
/// plus `\n` and `\t`, which prose keeps but a one-line value must not carry.
/// So a value is refused for precisely what printing it as prose would not
/// show, and an emoji such as `❤️` (heart plus VS16) is not refused.
#[must_use]
pub fn hides_or_moves_text(value: &str) -> bool {
    let mut previous = None;
    for c in value.chars() {
        if is_display_altering(c) && !is_kept_glyph_shaper(previous, c) {
            return true;
        }
        previous = Some(c);
    }
    false
}

/// Whether `symptom` can be placed inside a `'...'` shell word verbatim.
///
/// The value is untrusted subprocess output (the vLLM startup log tail) and the
/// messages it lands in invite the user to paste the command into a shell, so a
/// bare `'` would close the quote and let text nobody vetted become shell
/// syntax. An apostrophe is routine in Python error text (`can't allocate`,
/// `model 'foo'`), and a line only has to look like an allocation failure to be
/// selected, so this is an ordinary case rather than an exotic one.
///
/// Rejecting instead of escaping (`'` -> `'\''`) is deliberate. Escaping keeps
/// the exact bytes but yields a command a reader cannot check by eye, and a
/// wrong escape is *runnable* and misleading rather than obviously broken;
/// control bytes would still reach the terminal. The canonical fallback is the
/// branch that already exists for "this line cannot be used", and it is
/// guaranteed to report a cause.
///
/// What the user keeps depends on which surface they are looking at, and only
/// one of the two shows them their own line. The vLLM engine's startup-failure
/// hint renders the selected row beside the command and prints the log tail just
/// above it, so there nothing is lost but the copy-paste convenience. The `rocm
/// serve` summary's OOM note shows neither — it is one sentence plus the
/// command — so on that surface the fallback really does withhold the failing
/// line, and the user has to reach for `rocm services logs <service-id>` (which
/// is what docs/manual-testing.md §6a tells them).
///
/// This is the counterpart of [`strip_terminal_control_sequences`], not a
/// duplicate of it: text rocm-cli merely *echoes* is stripped, while text it
/// renders into a command the user is told to run is rejected, because a
/// stripped line is a lookalike of the user's error that is also *runnable*.
///
/// It is shared rather than per-engine because two surfaces build that command
/// from the same untrusted text — the vLLM engine's startup-failure hint and the
/// `rocm serve` summary's OOM note — and a guard that protects only one of them
/// is the bug it was written to prevent.
///
/// The character test is [`is_display_altering`], which takes in both
/// [`is_control_or_format`] and [`is_control_or_line_separator`] (each covers
/// scalars the other does not) and the rest of the default-ignorable set,
/// because `symptom` is the *raw* log line, not the stripped one: a `Cf` bidi
/// override reordered how the printed command renders, a `Zl` line separator
/// broke it across two rows, and a tag character would sit in it unseen.
#[must_use]
pub fn quotable_in_single_quotes(symptom: &str) -> bool {
    !symptom.contains('\'') && !symptom.chars().any(is_display_altering)
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
    Some(match skip_escape(chars, Body::Grammar) {
        // CSI — what a colourised logger emits. Only a well-formed SGR is
        // ignorable; see above.
        Escape::Csi { sgr: true } => Token::Ignorable,
        Escape::Csi { sgr: false } => Token::LineBreak,
        // Consuming a string body whole is also the one hole in the
        // never-merge property, since a line break inside the body goes with
        // it — and the body scan ends on a bare `ESC` or on end of input as
        // readily as on a terminator, so a terminator is not what creates the
        // hole. The module documentation is where that exception is stated,
        // and it is stated there rather than here because a body comment in a
        // private function reaches no reader of the public docs. Any change to
        // what `skip_escape` consumes for a string sequence has to change that
        // section, `rendered_lines`'s own qualifier, and the test that pins the
        // worked examples, in one commit.
        Escape::String | Escape::Bare => Token::Ignorable,
        // This is where `ESC E` (NEL) and `ESC D` (IND) live.
        Escape::Other => Token::LineBreak,
    })
}

/// What [`skip_escape`] consumed after an `ESC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Escape {
    /// A CSI (`ESC [`), and whether it was a well-formed `SGR`.
    Csi { sgr: bool },
    /// One of the string-argument sequences: OSC, DCS, SOS, PM, APC. Their
    /// bodies are arbitrary text (a window title, say) and must not be emitted
    /// as if the process had printed it, but they draw nothing at the cursor.
    String,
    /// `ESC` with nothing after it, or `ESC ESC`: only this `ESC` goes, and the
    /// next one is left to be re-read as a fresh introducer. A bare `ESC` draws
    /// nothing and moves nothing on its own.
    Bare,
    /// Any other escape: optional intermediates, then one final byte.
    Other,
}

/// How far a string-argument sequence's body may run; see
/// [`skip_string_sequence_body`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Body {
    Grammar,
    ToLineEnd,
}

/// Consumes the rest of an escape sequence from just after its `ESC` and says
/// which kind it was. The one place the 7-bit introducers are told apart, so
/// [`next_token`] and [`sanitize_terminal_text`] read the same grammar.
fn skip_escape(chars: &mut std::iter::Peekable<std::str::Chars<'_>>, body: Body) -> Escape {
    match chars.peek() {
        Some('[') => {
            chars.next();
            Escape::Csi {
                sgr: skip_csi_body(chars),
            }
        }
        Some(']' | 'P' | 'X' | '^' | '_') => {
            chars.next();
            skip_string_sequence_body(chars, body);
            Escape::String
        }
        None | Some('\u{1b}') => Escape::Bare,
        Some(_) => {
            skip_simple_escape_body(chars);
            Escape::Other
        }
    }
}

/// Classifies a character that is not an escape introducer.
///
/// `\t` is [`Token::Text`] rather than a boundary: it moves the cursor along the
/// row it is already on, so it is legitimate intra-line whitespace. Everything
/// else [`is_control_or_line_separator`] accepts is a boundary — `\n` and `\r`
/// obviously, but equally `\x0b` (`VT`) and `\x0c` (`FF`), which advance a line,
/// `\u{85}` (`NEL`), and the bytes that do nothing at all. The last group is the
/// lopsidedness above: a stray `\x01` between two rendered lines is far likelier
/// to be a mangled line advance than intra-line text, and guessing "boundary"
/// costs a missed diagnosis where guessing "text" costs a wrong one.
fn classify_char(c: char) -> Token {
    if c == '\t' {
        return Token::Text(c);
    }
    if is_control_or_line_separator(c) {
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
/// `ESC \` or its 8-bit form U+009C) from just after the introducer. A body
/// truncated by anything else — including a bare `ESC` starting the next
/// sequence — ends the scan with that byte left in place.
///
/// `Body::ToLineEnd` also ends the scan, without consuming it, at a line break
/// (`\n`, `\r`, U+2028, U+2029). [`sanitize_terminal_text`] reads it that way,
/// so an unterminated sequence in one line of untrusted prose cannot swallow
/// the lines after it. [`next_token`] keeps `Body::Grammar`, whose consumption
/// of a break the module documentation states and its tests pin.
fn skip_string_sequence_body(chars: &mut std::iter::Peekable<std::str::Chars<'_>>, body: Body) {
    while let Some(&c) = chars.peek() {
        if body == Body::ToLineEnd && matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}') {
            return;
        }
        if c == '\u{7}' || c == '\u{9c}' {
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

/// Makes multi-line prose from an untrusted source safe to write to a terminal:
/// what is left is exactly what the reader will see.
///
/// For text that cannot be refused outright — a model's reply, a tool's
/// output, an error message quoting either. Escape sequences are removed whole,
/// introducer and payload alike, in their 7-bit and 8-bit (C1) forms: an SGR
/// "conceal" cannot hide the text after it, an OSC cannot retitle the window or
/// write the clipboard (OSC 52), a DCS or APC payload is never drawn, and
/// nothing can move the cursor. Every other [`is_display_altering`] character
/// is dropped, except:
///
/// * `\n` and `\t`, which prose needs and which only move to the next line or
///   column;
/// * a lone `\r`, and the line and paragraph separators, which become `\n`:
///   returning to column one would overwrite what was already printed, but the
///   text after it is still worth reading; `\r\n` becomes a plain `\n`;
/// * a single zero-width joiner or non-joiner, or emoji or text presentation
///   selector, directly after a drawn character ([`is_kept_glyph_shaper`]),
///   which shapes that glyph and moves nothing.
///
/// Concealed text therefore becomes *visible*, rather than disappearing with
/// its escape sequence: a reader should see what was sent, not a tidied
/// version of it.
///
/// A string sequence (OSC, DCS, SOS, PM, APC) with no terminator ends at the
/// next line break, which is kept: an unterminated one costs at most the rest
/// of its own line, never the lines after it.
///
/// Unlike [`strip_terminal_control_sequences`], which serves single log lines,
/// this keeps the line structure. A command line meant to be pasted is not
/// prose and is not passed through here: its values are refused instead (see
/// [`hides_or_moves_text`]), because cleaning them would make the line shown
/// differ from the one that runs.
#[must_use]
pub fn sanitize_terminal_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous = None;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        let shown = match c {
            '\n' | '\t' => Some(c),
            '\r' => (chars.peek() != Some(&'\n')).then_some('\n'),
            '\u{2028}' | '\u{2029}' => Some('\n'),
            '\u{1b}' => {
                skip_escape(&mut chars, Body::ToLineEnd);
                None
            }
            // The 8-bit CSI, and the 8-bit DCS, SOS, OSC, PM and APC.
            '\u{9b}' => {
                skip_csi_body(&mut chars);
                None
            }
            '\u{90}' | '\u{98}' | '\u{9d}' | '\u{9e}' | '\u{9f}' => {
                skip_string_sequence_body(&mut chars, Body::ToLineEnd);
                None
            }
            _ if is_kept_glyph_shaper(previous, c) => Some(c),
            _ if is_display_altering(c) => None,
            _ => Some(c),
        };
        if let Some(shown) = shown {
            out.push(shown);
        }
        // What came directly before in the input, removed or not, as in
        // `hides_or_moves_text`: a shaper separated from its base by anything
        // invisible shapes nothing.
        previous = Some(c);
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
    fn quotable_in_single_quotes_rejects_quote_and_control_bearing_symptoms() {
        // The ordinary case the guard exists for: Python error text with an
        // apostrophe, which would close the shell quote in the printed command.
        assert!(!quotable_in_single_quotes(
            "vllm: torch.OutOfMemoryError: GPU 0 can't allocate the model's weights"
        ));
        assert!(!quotable_in_single_quotes(
            "vllm: failed to load model 'foo/bar'"
        ));
        // Terminal control bytes from vLLM's colourised logger.
        assert!(!quotable_in_single_quotes(
            "vllm: \u{1b}[31mRuntimeError: HIP out of memory\u{1b}[0m"
        ));
        assert!(!quotable_in_single_quotes("vllm: out of memory\u{7}"));
        assert!(!quotable_in_single_quotes("vllm: out of\nmemory"));
        // The canonical fallback must always be usable, or the rejection branch
        // would have nowhere to go.
        assert!(quotable_in_single_quotes(crate::VLLM_OOM_CANONICAL_SYMPTOM));
        assert!(quotable_in_single_quotes(
            "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB."
        ));
        // Every scalar the stripper removes must also be inadmissible in the
        // quoted command, not merely stripped from the echoed sentence: the
        // candidate this guard sees is built from the *raw* line, not the
        // stripped one, so it needs its own pins. `Cf` is not `char::is_control`
        // and `Zl`/`Zp` are in neither, which is why the test below is two
        // predicates rather than one.
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
    fn strip_terminal_control_sequences_removes_whole_escape_sequences() {
        // Pin the exact rendering, not the absence of fragments: a stripper that
        // dropped only the escape byte and the `[` it introduces would leave
        // `31m`/`0m` behind, which carries no control byte and contains neither
        // `[31m` nor `[0m`, so every absence check would still pass.
        assert_eq!(
            strip_terminal_control_sequences(
                "\u{1b}[31mRuntimeError: HIP out of memory\u{1b}[0m\u{7}"
            ),
            "RuntimeError: HIP out of memory"
        );
        // A non-CSI escape takes the byte it introduces with it.
        assert_eq!(strip_terminal_control_sequences("a\u{1b}Bc"), "ac");
        // Text with nothing to strip is returned unchanged.
        assert_eq!(
            strip_terminal_control_sequences(crate::VLLM_OOM_CANONICAL_SYMPTOM),
            crate::VLLM_OOM_CANONICAL_SYMPTOM
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
            // go red, as do the quoting assertions above.
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
    }

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

    #[test]
    fn concealed_text_is_shown_instead_of_hidden() {
        // SGR 8 hides what follows on the terminal while the bytes stay in the
        // line; the reader must see them.
        let sent = "rocm serve qwen\u{1b}[8m --port 9\u{1b}[28m --managed";
        assert_eq!(
            sanitize_terminal_text(sent),
            "rocm serve qwen --port 9 --managed"
        );
    }

    #[test]
    fn escape_sequences_are_removed_whole_in_both_widths() {
        for (sent, shown) in [
            // OSC 52 clipboard write, BEL-terminated.
            ("a\u{1b}]52;c;aGVsbG8=\u{7}b", "ab"),
            // OSC 0 window title, ST-terminated.
            ("a\u{1b}]0;title\u{1b}\\b", "ab"),
            // 8-bit OSC and 8-bit ST.
            ("a\u{9d}0;title\u{9c}b", "ab"),
            // 7-bit OSC ended by the 8-bit ST.
            ("a\u{1b}]0;title\u{9c}b", "ab"),
            // DCS and APC, 7-bit and 8-bit.
            ("a\u{1b}Pq#0;2\u{1b}\\b", "ab"),
            ("a\u{1b}_payload\u{1b}\\b", "ab"),
            ("a\u{90}q#0\u{9c}b", "ab"),
            ("a\u{9f}payload\u{9c}b", "ab"),
            // Cursor up, erase line, colours.
            ("a\u{1b}[1A\u{1b}[2K\u{1b}[1;31mb\u{1b}[0m", "ab"),
            // 8-bit CSI: the parameters and final byte go with it.
            ("a\u{9b}2Kb", "ab"),
            // Two-character and nF escapes.
            ("a\u{1b}cb\u{1b}(Bc", "abc"),
            // An unterminated OSC swallows the rest of its line, as on a terminal.
            ("a\u{1b}]0;never closed", "a"),
            // A lone ESC at the end.
            ("ab\u{1b}", "ab"),
            // Every string introducer, 7-bit and 8-bit, takes its payload with
            // it: SOS and PM included, which nothing above reaches.
            ("a\u{1b}Xsos\u{1b}\\b", "ab"),
            ("a\u{1b}^pm\u{7}b", "ab"),
            ("a\u{98}sos\u{9c}b", "ab"),
            ("a\u{9e}pm\u{9c}b", "ab"),
            // A CSI's intermediate bytes (`SP`, `$`) go with it, so its final
            // byte is not left behind as text.
            ("a\u{1b}[2 qb", "ab"),
            ("a\u{1b}[1$}b", "ab"),
            ("a\u{9b}1 qb", "ab"),
        ] {
            assert_eq!(sanitize_terminal_text(sent), shown, "{sent:?}");
        }
    }

    /// A string sequence left open must not take the lines after it with it:
    /// those are often the CLI's own (a status line, the next tool result), and
    /// an open sequence anywhere in a reply would otherwise blank the rest of
    /// the output.
    #[test]
    fn an_unterminated_string_sequence_ends_at_its_line() {
        for open in [
            "\u{1b}]0;x", // OSC
            "\u{1b}Pq",   // DCS
            "\u{1b}_x",   // APC
            "\u{1b}Xx",   // SOS
            "\u{1b}^x",   // PM
            "\u{9d}0;x",  // 8-bit OSC
            "\u{90}q",    // 8-bit DCS
            "\u{98}x",    // 8-bit SOS
            "\u{9e}x",    // 8-bit PM
            "\u{9f}x",    // 8-bit APC
        ] {
            for (newline, shown) in [
                ("\n", "\n"),
                ("\r\n", "\n"),
                ("\r", "\n"),
                ("\u{2028}", "\n"),
            ] {
                let sent = format!("reply{open}{newline}CLI line one\nCLI line two");
                assert_eq!(
                    sanitize_terminal_text(&sent),
                    format!("reply{shown}CLI line one\nCLI line two"),
                    "{sent:?}"
                );
            }
        }
    }

    #[test]
    fn prose_keeps_its_layout_and_its_script() {
        let text =
            "line one\n\tindented 模型 é 👩\u{200d}💻 \u{2764}\u{fe0f} a\u{200c}b \u{2764}\u{fe0e}";
        assert_eq!(sanitize_terminal_text(text), text);
        assert_eq!(sanitize_terminal_text("a\r\nb"), "a\nb");
        assert_eq!(
            sanitize_terminal_text("shown\rhidden"),
            "shown\nhidden",
            "a carriage return must not overwrite what is already on the line"
        );
        assert_eq!(sanitize_terminal_text("a\u{2028}b"), "a\nb");
        assert_eq!(sanitize_terminal_text("a\u{202e}b\u{8}c\u{7}"), "abc");
        assert_eq!(
            sanitize_terminal_text("run\u{e0020}\u{e0069}\u{e0064} it\u{3164}"),
            "run it",
            "tag characters spell text the reader cannot see; it must not pass"
        );
    }

    /// The characters a value is refused for, beyond `Cc`: every one is
    /// invisible or reorders the line, and none is `char::is_control`, which is
    /// why a check written as `is_control` alone let them through.
    #[test]
    fn invisible_characters_are_display_altering() {
        for c in [
            '\u{202e}',
            '\u{202a}',
            '\u{2066}',
            '\u{2069}', // bidi
            '\u{200e}',
            '\u{061c}', // directional marks
            '\u{200b}',
            '\u{200d}',
            '\u{2060}',
            '\u{feff}', // zero-width
            '\u{2028}',
            '\u{2029}', // separators
            '\u{00ad}',
            '\u{034f}',
            '\u{fe0f}', // soft hyphen, CGJ, VS16
            '\u{115f}',
            '\u{3164}',
            '\u{ffa0}', // Hangul fillers
            '\u{e0001}',
            '\u{e0041}',
            '\u{e007f}', // tags
        ] {
            assert!(!c.is_control(), "U+{:04X}", c as u32);
            assert!(is_display_altering(c), "U+{:04X}", c as u32);
        }
        for c in ['a', 'é', '模', '\'', '&', ' ', '👩'] {
            assert!(!is_display_altering(c), "{c:?}");
        }
    }

    /// Both ends of every range of the default-ignorable table, and a point
    /// just outside each, so narrowing or widening a range is seen.
    #[test]
    fn the_default_ignorable_table_has_exact_edges() {
        let ranges: [(u32, u32); 17] = [
            (0x00ad, 0x00ad),
            (0x034f, 0x034f),
            (0x061c, 0x061c),
            (0x115f, 0x1160),
            (0x17b4, 0x17b5),
            (0x180b, 0x180f),
            (0x200b, 0x200f),
            (0x202a, 0x202e),
            (0x2060, 0x206f),
            (0x3164, 0x3164),
            (0xfe00, 0xfe0f),
            (0xfeff, 0xfeff),
            (0xffa0, 0xffa0),
            (0xfff0, 0xfff8),
            (0x1bca0, 0x1bca3),
            (0x1d173, 0x1d17a),
            (0xe0000, 0xe0fff),
        ];
        for (first, last) in ranges {
            for inside in [first, last] {
                let c = char::from_u32(inside).expect("scalar");
                assert!(is_default_ignorable(c), "U+{inside:04X} is in the set");
            }
            for outside in [first - 1, last + 1] {
                let in_another = ranges.iter().any(|&(a, b)| (a..=b).contains(&outside));
                if let Some(c) = char::from_u32(outside)
                    && !in_another
                {
                    assert!(
                        !is_default_ignorable(c),
                        "U+{outside:04X} is not in the set"
                    );
                }
            }
        }
    }

    /// The glyph-shaping characters stay only one at a time, directly after a
    /// character that is drawn; anything else, and the variation selectors
    /// beyond the presentation pair, is removed, so they cannot carry text.
    #[test]
    fn glyph_shapers_stay_only_singly_after_a_drawn_character() {
        for (sent, shown) in [
            ("a\u{200c}b", "a\u{200c}b"),
            ("a\u{200d}b", "a\u{200d}b"),
            ("\u{2764}\u{fe0f}", "\u{2764}\u{fe0f}"),
            ("\u{2764}\u{fe0e}", "\u{2764}\u{fe0e}"),
            // A run keeps its first member only.
            ("a\u{fe0f}\u{fe0f}\u{fe0f}b", "a\u{fe0f}b"),
            ("a\u{200d}\u{200c}b", "a\u{200d}b"),
            // At the start, or after a break or a tab, there is nothing to shape.
            ("\u{200d}a", "a"),
            ("a\n\u{fe0f}b", "a\nb"),
            ("a\t\u{200d}b", "a\tb"),
            // Anything invisible between the base and the shaper, an escape
            // sequence included, leaves it shaping nothing.
            ("a\u{1b}[0m\u{fe0f}", "a"),
            ("a\u{200b}\u{fe0f}b", "ab"),
            // The other variation selectors are never kept.
            ("a\u{e0100}\u{e0101}\u{fe0f}\u{200d}b", "ab"),
            ("a\u{fe00}b", "ab"),
            ("\u{1820}\u{180b}\u{180c}\u{180d}\u{180f}", "\u{1820}"),
        ] {
            assert_eq!(sanitize_terminal_text(sent), shown, "{sent:?}");
        }
    }

    /// A value is refused for exactly what printing would not show, plus the
    /// newline and tab a one-line value must not carry.
    #[test]
    fn a_value_hides_text_exactly_where_printing_would_change_it() {
        for value in [
            "C:\\Users\\me\\\u{2764}\u{fe0f}\\rocm",
            "👩\u{200d}💻",
            "/mnt/my folder",
            "模型",
        ] {
            assert!(!hides_or_moves_text(value), "{value:?}");
            assert_eq!(sanitize_terminal_text(value), value);
        }
        for value in [
            "a\nb",
            "a\tb",
            "a\rb",
            "a\u{1b}[2Kb",
            "a\u{9b}b",
            "a\u{202e}b",
            "a\u{200b}b",
            "\u{fe0f}a",
            "a\u{fe0f}\u{fe0f}",
            "a\u{e0100}",
            "a\u{3164}",
            "a\u{2028}b",
        ] {
            assert!(hides_or_moves_text(value), "{value:?}");
        }
    }

    proptest::proptest! {
        /// Whatever goes in, what comes out is what the refusal predicate
        /// accepts as a value, apart from the line breaks and tabs prose keeps:
        /// nothing display-altering survives except a single glyph shaper
        /// directly after a drawn character.
        #[test]
        fn nothing_display_altering_survives_sanitizing(text in proptest::prelude::any::<String>()) {
            let cleaned = sanitize_terminal_text(&text);
            for line in cleaned.split(['\n', '\t']) {
                proptest::prop_assert!(!hides_or_moves_text(line), "{:?} -> {:?}", text, cleaned);
            }
        }

        /// The same over input made of the characters that matter: every
        /// escape introducer, 7-bit and 8-bit, their terminators, line breaks,
        /// glyph shapers and ordinary text. The text after an introducer's
        /// line is never lost: every `Z` sent after the last line break in the
        /// input comes back.
        #[test]
        fn nothing_display_altering_survives_hostile_sequences(
            text in "[\u{1b}\u{9b}\u{9c}\u{9d}\u{90}\u{98}\u{9e}\u{9f}\u{7}\\[\\]\\\\P_^X;0-9a-y \r\n\u{202e}\u{e0041}\u{fe0f}\u{200d}]*"
        ) {
            let cleaned = sanitize_terminal_text(&text);
            for line in cleaned.split(['\n', '\t']) {
                proptest::prop_assert!(!hides_or_moves_text(line), "{:?} -> {:?}", text, cleaned);
            }
            let tail = format!("{text}\nZZZ");
            proptest::prop_assert!(
                sanitize_terminal_text(&tail).ends_with("\nZZZ"),
                "a line after {:?} was lost", text
            );
        }

        /// Text with nothing to remove is returned unchanged.
        #[test]
        fn plain_text_is_untouched(text in "[ -~\n\té模型]*") {
            proptest::prop_assert_eq!(sanitize_terminal_text(&text), text);
        }
    }
}
