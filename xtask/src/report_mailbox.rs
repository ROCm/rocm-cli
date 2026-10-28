// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Read Doctor reports out of mail exported from the report mailbox.
//!
//! The transport adapter for the rollup, and the only part of it that knows
//! reports arrive by mail. It turns an exported `.eml` file into the report
//! body and nothing else; [`rocm_core::read_report`] decides what the body
//! says. When reports move to a tracker, this file is what gets replaced.
//!
//! The mail envelope identifies the sender: the `From` address and display
//! name, the `Received` chain, the client in `X-Mailer`, the time zone in
//! `Date`, and often a signature under the body. The report itself carries
//! none of that, and this module keeps it that way downstream. The header
//! parser keeps the five headers it needs (`Subject`, `Message-ID`, the two
//! content headers, and `Date`) and drops every other one while reading. A
//! `Message-ID` is held only as a digest, `Date` only as a week (below), and
//! only the JSON value in the body is read. So no other header can leave this
//! module, whatever a later change prints. The subject does leave it, because it
//! is what tells a report from other mail; for a report the CLI wrote it, but
//! the subject of a reply or of unrelated mail is free text, which the caller
//! prints only on request.
//!
//! `Date` is the one kept header that is itself sender-identifying: its zone is.
//! An earlier version dropped it for that reason. It is kept now because it is
//! the only place a report's time is recorded, and the weekly trend needs a time
//! axis. It is reduced to the week it falls in inside [`week_of`], which
//! converts it to UTC, takes the day and then the week, and returns nothing
//! else. The time of day, the zone and the day itself never leave that function,
//! so none of them can reach a count or an output.
//!
//! Only `.eml` is read: one message per file, which is what Outlook on the web
//! and the Graph `$value` endpoint produce, and a damaged file is exactly one
//! unread report. mbox is not, because its `From `-line boundaries can merge or
//! split messages without any error, and a count that silently changes is the
//! failure this whole path exists to avoid.

use base64::Engine as _;
use chrono::DateTime;
use rocm_core::report_delivery::SUBJECT_TAG;
use rocm_core::{ReadOutcome, ReportWeek};
use sha2::{Digest, Sha256};

/// The earliest day a report can have been sent, as days since 1970-01-01:
/// 2026-10-05, the day the code that writes reports first landed in this
/// repository. No report exists from before it.
///
/// The report address takes mail from anyone, so a `Date` is what the sender's
/// client claims and not evidence of when a report was written. One before this
/// or one in the future reads as undated instead of inventing a week that never
/// happened. A date inside the range is still taken as given, so one sender can
/// still choose a week inside it; that is a limit of reading time from the mail,
/// and it matters once a rise is acted on without a person reading it.
pub(crate) const EARLIEST_PLAUSIBLE_DAY: i64 = 20_731;

/// One exported message, reduced to what the rollup may see.
///
/// There is deliberately no field here that could hold a sender, and none finer
/// than a week for when it was sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Message {
    /// The decoded subject, used only to tell report mail from other mail.
    pub subject: String,
    /// SHA-256 of the `Message-ID`, used only to drop a message exported
    /// twice. The raw value often names the sender's host, so it is not kept.
    pub id_digest: Option<String>,
    /// The week the message was sent in, or `None` when its `Date` is missing,
    /// cannot be read, or is not plausible. See [`week_of`].
    pub week: Option<ReportWeek>,
    /// The decoded text of the message, or `None` when no text part exists.
    pub body: Option<String>,
}

impl Message {
    /// Whether this message is a report, as opposed to a reply, an automatic
    /// reply, a bounce, or unrelated mail.
    ///
    /// The CLI writes the tag first in every subject, so anything that puts a
    /// prefix before it (`Re:`, `Automatic reply:`, `Undeliverable:`) is not a
    /// report. Forwards are the exception: a colleague forwarding a user's
    /// report has still delivered one report.
    pub fn is_report(&self) -> bool {
        self.without_forward_prefixes().starts_with(SUBJECT_TAG)
    }

    /// Whether the message arrived as a forward.
    ///
    /// A forward gets a new `Message-ID`, so it cannot be matched to the
    /// original, and a report forwarded to the mailbox after it was also sent
    /// there directly is counted twice. Identical report bodies from two
    /// people are legitimate, so the body cannot settle it either. The caller
    /// counts forwards visibly instead, so a reader can judge how much of a
    /// total they might account for.
    pub fn is_forward(&self) -> bool {
        self.without_forward_prefixes().len() != self.subject.trim_start().len()
    }

    fn without_forward_prefixes(&self) -> &str {
        let mut subject = self.subject.trim_start();
        loop {
            let lower = subject.to_ascii_lowercase();
            let Some(prefix) = ["fwd:", "fw:"].iter().find(|p| lower.starts_with(**p)) else {
                return subject;
            };
            subject = subject[prefix.len()..].trim_start();
        }
    }
}

/// Read a body as a report, tolerating text a person typed around it.
///
/// The whole body is tried first. When that is not even JSON, each `{` is
/// tried in turn, and the first complete JSON object there that carries a
/// `schema` is read. That is the report itself when a greeting sits above it,
/// even one with braces of its own such as "Hi {name}", or a signature below
/// it. This reads the structure, never the wording, and anything still
/// unreadable stays unread and is counted as such.
pub(crate) fn read_body(body: &str) -> ReadOutcome {
    let whole = rocm_core::read_report(body);
    if !matches!(whole, ReadOutcome::Unread { schema_seen: 0 }) {
        return whole;
    }
    body.match_indices('{')
        .find_map(|(start, _)| {
            match serde_json::Deserializer::from_str(&body[start..])
                .into_iter::<serde_json::Value>()
                .next()
            {
                Some(Ok(value)) if value.get("schema").is_some() => {
                    Some(rocm_core::read_report(&value.to_string()))
                }
                _ => None,
            }
        })
        .unwrap_or(whole)
}

/// Parse one `.eml` file. `today` is the current day in days since 1970-01-01,
/// passed in so the plausibility check below is the same on every run.
pub(crate) fn parse_eml(bytes: &[u8], today: i64) -> Message {
    let text = String::from_utf8_lossy(bytes).replace("\r\n", "\n");
    let (headers, body) = split_entity(&text);
    Message {
        subject: headers
            .subject
            .as_deref()
            .map(decode_words)
            .unwrap_or_default(),
        id_digest: headers
            .message_id
            .as_deref()
            .map(|id| format!("{:x}", Sha256::digest(id.trim().as_bytes()))),
        week: headers
            .date
            .as_deref()
            .and_then(|date| week_of(date, today)),
        body: entity_text(&headers, body),
    }
}

/// The week a `Date` header falls in, and nothing else about it.
///
/// The header is converted to UTC, so a message sent late on a Sunday in one
/// zone and early on a Monday in another lands in the same week. The zone is
/// used for that and then goes out of scope with the rest of the parsed date.
/// `None` for a header that is missing, malformed, before
/// [`EARLIEST_PLAUSIBLE_DAY`], or after `today` plus a day of slack for clocks
/// that disagree.
///
/// A trailing comment such as `(CEST)`, nested ones included, is part of the
/// format and `chrono` reads it. An unclosed one is malformed, so the header is
/// undated.
fn week_of(date: &str, today: i64) -> Option<ReportWeek> {
    let day = DateTime::parse_from_rfc2822(date.trim())
        .ok()?
        .timestamp()
        .div_euclid(86_400);
    (EARLIEST_PLAUSIBLE_DAY..=today.saturating_add(1))
        .contains(&day)
        .then(|| ReportWeek::from_unix_day(day))
}

/// The headers this module reads. Everything else is dropped while parsing.
#[derive(Debug, Default)]
struct Headers {
    subject: Option<String>,
    message_id: Option<String>,
    /// Held only until [`parse_eml`] reduces it to a week.
    date: Option<String>,
    content_type: Option<String>,
    transfer_encoding: Option<String>,
}

fn split_entity(text: &str) -> (Headers, &str) {
    let mut headers = Headers::default();
    // A part with no headers starts with the blank line that ends them.
    // Looking for the next blank line instead would take the body as headers.
    if let Some(body) = text.strip_prefix('\n') {
        return (headers, body);
    }
    let (head, body) = text.split_once("\n\n").unwrap_or((text, ""));
    // Unfold first: a header continues on every following line that starts
    // with whitespace.
    let mut unfolded: Vec<String> = Vec::new();
    for line in head.lines() {
        match unfolded.last_mut() {
            Some(last) if line.starts_with([' ', '\t']) => {
                last.push(' ');
                last.push_str(line.trim_start());
            }
            _ => unfolded.push(line.to_owned()),
        }
    }
    for line in unfolded {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().to_owned();
        match name.trim().to_ascii_lowercase().as_str() {
            "subject" => headers.subject = Some(value),
            "message-id" => headers.message_id = Some(value),
            "date" => headers.date = Some(value),
            "content-type" => headers.content_type = Some(value),
            "content-transfer-encoding" => headers.transfer_encoding = Some(value),
            _ => {}
        }
    }
    (headers, body)
}

/// A parameter of a structured header, e.g. `boundary` of `Content-Type`.
fn parameter(header: &str, name: &str) -> Option<String> {
    header.split(';').skip(1).find_map(|part| {
        let (key, value) = part.split_once('=')?;
        key.trim()
            .eq_ignore_ascii_case(name)
            .then(|| value.trim().trim_matches('"').to_owned())
    })
}

fn media_type(headers: &Headers) -> String {
    headers
        .content_type
        .as_deref()
        .and_then(|value| value.split(';').next())
        .unwrap_or("text/plain")
        .trim()
        .to_ascii_lowercase()
}

/// The readable text of an entity: the first `text/plain` part, else the first
/// `text/html` part with its markup removed, because some clients send HTML
/// only.
fn entity_text(headers: &Headers, body: &str) -> Option<String> {
    find_text(headers, body, "text/plain")
        .or_else(|| find_text(headers, body, "text/html").map(|html| strip_html(&html)))
}

fn find_text(headers: &Headers, body: &str, wanted: &str) -> Option<String> {
    let media = media_type(headers);
    if media.starts_with("multipart/") {
        let boundary = parameter(headers.content_type.as_deref()?, "boundary")?;
        return multipart_parts(body, &boundary)
            .into_iter()
            .find_map(|part| {
                let (part_headers, part_body) = split_entity(part);
                find_text(&part_headers, part_body, wanted)
            });
    }
    if media != wanted {
        return None;
    }
    let mut text = decode_transfer(headers.transfer_encoding.as_deref(), body);
    let content_type = headers.content_type.as_deref().unwrap_or_default();
    if parameter(content_type, "format").is_some_and(|f| f.eq_ignore_ascii_case("flowed")) {
        let delsp = parameter(content_type, "delsp").is_some_and(|d| d.eq_ignore_ascii_case("yes"));
        text = unflow(&text, delsp);
    }
    // An empty part is not the text: a client that sends HTML may leave the
    // plain alternative blank, and taking it would hide the HTML fallback.
    (!text.trim().is_empty()).then_some(text)
}

fn multipart_parts<'a>(body: &'a str, boundary: &str) -> Vec<&'a str> {
    let delimiter = format!("--{boundary}");
    let mut parts = Vec::new();
    let mut current: Option<usize> = None;
    let mut offset = 0;
    for line in body.split_inclusive('\n') {
        let trimmed = line.trim_end();
        if trimmed == delimiter || trimmed == format!("{delimiter}--") {
            if let Some(start) = current {
                parts.push(body[start..offset].trim_end_matches('\n'));
            }
            if trimmed.ends_with("--") && trimmed != delimiter {
                return parts;
            }
            current = Some(offset + line.len());
        }
        offset += line.len();
    }
    if let Some(start) = current {
        parts.push(&body[start..]);
    }
    parts
}

fn decode_transfer(encoding: Option<&str>, body: &str) -> String {
    let encoding = encoding.unwrap_or("7bit").trim().to_ascii_lowercase();
    let bytes = match encoding.as_str() {
        "quoted-printable" => decode_quoted_printable(body),
        "base64" => {
            let compact: String = body.chars().filter(|c| !c.is_whitespace()).collect();
            // An undecodable part reads as empty, so the report is counted as
            // unread rather than dropped.
            base64::engine::general_purpose::STANDARD
                .decode(compact)
                .unwrap_or_default()
        }
        _ => body.as_bytes().to_vec(),
    };
    String::from_utf8_lossy(&bytes).into_owned()
}

fn decode_quoted_printable(body: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len());
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'=' {
            // A soft line break: the encoder wrapped a line that was not
            // wrapped in the original.
            if bytes.get(i + 1) == Some(&b'\n') {
                i += 2;
                continue;
            }
            if let Some(byte) = bytes
                .get(i + 1..i + 3)
                .and_then(|hex| std::str::from_utf8(hex).ok())
                .and_then(|hex| u8::from_str_radix(hex, 16).ok())
            {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

/// Undo `format=flowed` (RFC 3676): a line ending in a space continues on the
/// next one, and a leading space added for stuffing is removed.
fn unflow(text: &str, delsp: bool) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let line = line.strip_prefix(' ').unwrap_or(line);
        if line.ends_with(' ') && line != "-- " {
            out.push_str(if delsp { &line[..line.len() - 1] } else { line });
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn strip_html(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut tag = String::new();
    for c in html.chars() {
        match (in_tag, c) {
            (false, '<') => {
                in_tag = true;
                tag.clear();
            }
            (true, '>') => {
                in_tag = false;
                let name = tag.trim_start_matches('/').to_ascii_lowercase();
                if name.starts_with("br") || name.starts_with('p') || name.starts_with("div") {
                    text.push('\n');
                }
            }
            (true, c) => tag.push(c),
            (false, c) => text.push(c),
        }
    }
    decode_entities(&text)
}

fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        let decoded = rest.find(';').filter(|end| *end <= 10).and_then(|end| {
            let name = &rest[1..end];
            let c = match name {
                "quot" => Some('"'),
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "nbsp" => Some(' '),
                "apos" => Some('\''),
                _ => name
                    .strip_prefix('#')
                    .and_then(|n| {
                        n.strip_prefix(['x', 'X'])
                            .map_or_else(|| n.parse().ok(), |hex| u32::from_str_radix(hex, 16).ok())
                    })
                    .and_then(char::from_u32),
            }?;
            Some((c, end))
        });
        if let Some((c, end)) = decoded {
            out.push(c);
            rest = &rest[end + 1..];
        } else {
            out.push('&');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

/// Decode RFC 2047 encoded words in a header, e.g. `=?utf-8?Q?=5Brocm-doctor=5D?=`,
/// so a client that encodes the subject does not hide the tag.
fn decode_words(header: &str) -> String {
    let mut out = String::new();
    let mut rest = header;
    let mut last_was_word = false;
    while let Some(start) = rest.find("=?") {
        let between = &rest[..start];
        // Whitespace between two adjacent encoded words is not part of the text.
        if !(last_was_word && between.trim().is_empty()) {
            out.push_str(between);
        }
        let word = &rest[start + 2..];
        // `charset?encoding?data?=`. The terminator is searched for only after
        // the first two fields, because Q data such as `=5B` follows a `?`.
        let decoded = (|| {
            let (_charset, after_charset) = word.split_once('?')?;
            let (encoding, after_encoding) = after_charset.split_once('?')?;
            let data_end = after_encoding.find("?=")?;
            let data = &after_encoding[..data_end];
            let end = word.len() - after_encoding.len() + data_end;
            let bytes = match encoding.to_ascii_lowercase().as_str() {
                "q" => decode_quoted_printable(&data.replace('_', " ")),
                "b" => base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .ok()?,
                _ => return None,
            };
            Some((String::from_utf8_lossy(&bytes).into_owned(), end))
        })();
        if let Some((text, end)) = decoded {
            out.push_str(&text);
            rest = &word[end + 2..];
            last_was_word = true;
        } else {
            out.push_str("=?");
            rest = word;
            last_was_word = false;
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2027-01-15, as days since 1970-01-01: the day these tests run "on".
    const TODAY: i64 = 20_833;

    fn parse(bytes: &[u8]) -> Message {
        parse_eml(bytes, TODAY)
    }

    /// The Monday of the week a message with this `Date` header lands in.
    fn week_for(date: &str) -> Option<String> {
        let headers = format!("Subject: [rocm-doctor] x\nDate: {date}");
        parse(&eml(&headers, REPORT))
            .week
            .map(|week| week.to_string())
    }

    const REPORT: &str = r#"{
  "schema": 1,
  "architecture": "gfx942",
  "architecture_matrix": "ROCm compatibility matrix, ROCm 7.1",
  "entry": "fix-6-path",
  "os_family": "linux",
  "os_major": "22",
  "distro": "ubuntu",
  "rocm": "7.1",
  "engine": "pytorch",
  "engine_version": "2.5",
  "cli_version": "0.1.0",
  "fix_offered": true
}"#;

    fn eml(headers: &str, body: &str) -> Vec<u8> {
        format!("{headers}\r\n\r\n{body}")
            .replace('\n', "\r\n")
            .replace("\r\r\n", "\r\n")
            .into_bytes()
    }

    fn understood(message: &Message) -> bool {
        matches!(
            read_body(message.body.as_deref().unwrap_or_default()),
            ReadOutcome::Understood(_)
        )
    }

    #[test]
    fn a_plain_message_reads_as_a_report() {
        let message = parse(&eml(
            "Subject: [rocm-doctor] fix-6-path on gfx942 / ubuntu-22\nContent-Type: text/plain; charset=utf-8",
            REPORT,
        ));
        assert!(message.is_report());
        assert!(understood(&message));
    }

    #[test]
    fn quoted_printable_soft_breaks_rejoin_the_report() {
        // An encoder may wrap any line, including mid-token, and may encode
        // characters that need no encoding.
        let encoded = REPORT.replacen("\"schema\"", "=22schema=22", 1).replace(
            "\"architecture_matrix\": \"ROCm",
            "\"architecture_matrix\": \"RO=\nCm",
        );
        let message = parse(&eml(
            "Subject: [rocm-doctor] x\nContent-Type: text/plain\nContent-Transfer-Encoding: quoted-printable",
            &encoded,
        ));
        assert!(understood(&message), "{:?}", message.body);
    }

    #[test]
    fn a_base64_part_inside_multipart_reads_as_a_report() {
        let encoded = base64::engine::general_purpose::STANDARD.encode(REPORT);
        let wrapped: Vec<&str> = encoded
            .as_bytes()
            .chunks(76)
            .map(|c| std::str::from_utf8(c).unwrap())
            .collect();
        let body = format!(
            "--b1\nContent-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: base64\n\n{}\n--b1\nContent-Type: text/html\n\n<p>ignored</p>\n--b1--\n",
            wrapped.join("\n")
        );
        let message = parse(&eml(
            "Subject: [rocm-doctor] x\nContent-Type: multipart/alternative;\n boundary=\"b1\"",
            &body,
        ));
        assert!(understood(&message), "{:?}", message.body);
    }

    /// Some clients send the prefilled body as HTML only.
    #[test]
    fn an_html_only_message_reads_as_a_report() {
        let html: String = REPORT
            .replace('"', "&quot;")
            .lines()
            .flat_map(|line| {
                [
                    "<div>".to_owned(),
                    line.replace(' ', "&nbsp;"),
                    "</div>".to_owned(),
                ]
            })
            .collect();
        let message = parse(&eml(
            "Subject: [rocm-doctor] x\nContent-Type: multipart/alternative; boundary=zz",
            &format!(
                "--zz\nContent-Type: text/html; charset=utf-8\n\n<html><body>{html}</body></html>\n--zz--\n"
            ),
        ));
        assert!(understood(&message), "{:?}", message.body);
    }

    #[test]
    fn a_flowed_body_is_unwrapped() {
        let flowed = REPORT.replace(
            "\"ROCm compatibility matrix, ",
            "\"ROCm compatibility matrix, \n",
        );
        let message = parse(&eml(
            "Subject: [rocm-doctor] x\nContent-Type: text/plain; format=flowed",
            &flowed,
        ));
        assert!(understood(&message), "{:?}", message.body);
    }

    #[test]
    fn an_encoded_subject_still_carries_the_tag() {
        let message = parse(&eml(
            "Subject: =?utf-8?Q?=5Brocm-doctor=5D_fix-6-path?= =?utf-8?B?IG9uIGdmeDk0Mg==?=",
            REPORT,
        ));
        assert_eq!(message.subject, "[rocm-doctor] fix-6-path on gfx942");
        assert!(message.is_report());
    }

    #[test]
    fn text_typed_around_the_report_does_not_stop_it_being_read() {
        // The signature carries braces on purpose: only the first complete
        // value is read, so they cannot spoil the report above them.
        let body = format!("Hi team,\n\nsee below.\n\n{REPORT}\n\n--\nA. Person\nTitle {{Team}}\n");
        assert!(matches!(read_body(&body), ReadOutcome::Understood(_)));
        // Braces above the report, in a greeting or a quoted line, are tried
        // and passed over rather than taken as the report.
        let body = format!("Hi {{name}},\n> {{quoted\n\n{REPORT}\n");
        assert!(
            matches!(read_body(&body), ReadOutcome::Understood(_)),
            "{body}"
        );
        // A report cut short is unread, and counted, rather than lost.
        let cut = format!("Hi team,\n{}", &REPORT[..REPORT.len() / 2]);
        assert!(matches!(
            read_body(&cut),
            ReadOutcome::Unread { schema_seen: 0 }
        ));
    }

    #[test]
    fn replies_automatic_replies_and_bounces_are_not_reports() {
        for subject in [
            "Re: [rocm-doctor] fix-6-path on gfx942 / ubuntu-22",
            "RE: [rocm-doctor] fix-6-path on gfx942 / ubuntu-22",
            "Automatic reply: [rocm-doctor] fix-6-path on gfx942 / ubuntu-22",
            "Undeliverable: [rocm-doctor] fix-6-path on gfx942 / ubuntu-22",
            "Lunch on Friday?",
        ] {
            let message = parse(&eml(&format!("Subject: {subject}"), REPORT));
            assert!(!message.is_report(), "{subject}");
        }
        for subject in ["Fw: [rocm-doctor] x", "FWD: Fw: [rocm-doctor] x"] {
            let message = parse(&eml(&format!("Subject: {subject}"), REPORT));
            assert!(message.is_report(), "{subject}");
        }
    }

    /// Nothing that identifies the sender survives parsing.
    #[test]
    fn the_sender_never_leaves_the_parser() {
        let message = parse(&eml(
            "From: Sender-Marker-Name <sender-marker@example.com>\n\
             Received: from host-marker.example.com (10.0.0.1)\n\
             X-Mailer: Mailer-Marker 1.0\n\
             Date: Mon, 26 Oct 2026 10:00:00 +0200\n\
             Message-ID: <id-marker@host-marker.example.com>\n\
             Subject: [rocm-doctor] fix-6-path on gfx942 / ubuntu-22",
            &format!("{REPORT}\n\n--\nSignature-Marker\n"),
        ));
        let kept = format!("{message:?}");
        // The `Date` is kept as a week and as nothing else: not its time of
        // day, its zone or its day.
        for marker in [
            "Sender-Marker",
            "sender-marker",
            "host-marker",
            "Mailer-Marker",
            "+0200",
            "10:00",
            "26 Oct",
            "id-marker",
        ] {
            assert!(!kept.contains(marker), "{marker} survived parsing: {kept}");
        }
        assert_eq!(
            message.week.map(|week| week.to_string()).as_deref(),
            Some("2026-10-26"),
            "the week is what survives of the Date"
        );
        // The signature is in the body, which is kept; it is never parsed.
        let ReadOutcome::Understood(report) = read_body(message.body.as_deref().unwrap()) else {
            panic!("the report under the headers must still read");
        };
        assert!(!format!("{report:?}").contains("Signature-Marker"));
        assert_eq!(message.id_digest.as_ref().map(String::len), Some(64));
    }

    /// A part may omit its headers entirely, which makes it `text/plain`.
    #[test]
    fn a_part_with_no_headers_is_read_as_text() {
        let message = parse(&eml(
            "Subject: [rocm-doctor] x\nContent-Type: multipart/mixed; boundary=q",
            &format!("--q\n\n{REPORT}\n--q--\n"),
        ));
        assert!(understood(&message), "{:?}", message.body);
    }

    /// A blank plain alternative must not hide the HTML one beside it.
    #[test]
    fn an_empty_plain_part_falls_back_to_the_html_part() {
        let html = REPORT.replace('"', "&quot;").replace('\n', "<br>");
        let message = parse(&eml(
            "Subject: [rocm-doctor] x\nContent-Type: multipart/alternative; boundary=zz",
            &format!(
                "--zz\nContent-Type: text/plain\n\n\n--zz\nContent-Type: text/html\n\n{html}\n--zz--\n"
            ),
        ));
        assert!(understood(&message), "{:?}", message.body);
    }

    #[test]
    fn forwards_are_recognised_and_other_reports_are_not() {
        let subject = |s: &str| parse(&eml(&format!("Subject: {s}"), REPORT));
        assert!(subject("Fw: [rocm-doctor] x").is_forward());
        assert!(subject("FWD: Fw: [rocm-doctor] x").is_forward());
        assert!(!subject("[rocm-doctor] x").is_forward());
    }

    /// Thursday 2026-10-29 is in the week that starts on Monday 2026-10-26,
    /// whatever zone it was written in and however the header is spelled.
    #[test]
    fn a_date_lands_in_the_week_of_its_utc_day_whatever_its_zone() {
        for date in [
            "Thu, 29 Oct 2026 10:00:00 +0200",
            "Thu, 29 Oct 2026 10:00:00 -0800",
            "Thu, 29 Oct 2026 10:00:00 GMT",
            "Thu, 29 Oct 2026 10:00:00 +0000 (UTC)",
            "Thu, 29 Oct 2026 10:00:00 +0200 (Central European Summer Time)",
            // A folded header is one header.
            "Thu, 29 Oct\n 2026 10:00:00 +0000",
        ] {
            assert_eq!(week_for(date).as_deref(), Some("2026-10-26"), "{date:?}");
        }
    }

    /// The week turns over at midnight in UTC, not at midnight where the mail
    /// was sent, so the same instant is the same week for every sender.
    #[test]
    fn the_week_turns_over_at_midnight_utc_not_where_it_was_sent() {
        for (date, monday) in [
            ("Sun, 25 Oct 2026 23:59:59 +0000", "2026-10-19"),
            ("Mon, 26 Oct 2026 00:00:00 +0000", "2026-10-26"),
            // Monday 00:30 in UTC, though it was still Sunday where it was sent.
            ("Sun, 25 Oct 2026 23:30:00 -0100", "2026-10-26"),
            // Sunday 23:30 in UTC, though it was already Monday where it was sent.
            ("Mon, 26 Oct 2026 00:30:00 +0100", "2026-10-19"),
        ] {
            assert_eq!(week_for(date).as_deref(), Some(monday), "{date}");
        }
    }

    /// A week named by its Monday reads the same either side of New Year.
    #[test]
    fn a_week_that_spans_new_year_is_one_week() {
        for (date, monday) in [
            ("Mon, 28 Dec 2026 00:00:00 +0000", "2026-12-28"),
            ("Thu, 31 Dec 2026 12:00:00 +0000", "2026-12-28"),
            ("Sun, 03 Jan 2027 23:59:59 +0000", "2026-12-28"),
            ("Mon, 04 Jan 2027 00:00:00 +0000", "2027-01-04"),
        ] {
            assert_eq!(week_for(date).as_deref(), Some(monday), "{date}");
        }
    }

    /// Mail from anyone can say anything in `Date`. A date that cannot be
    /// trusted to mean a week puts the report in no week, which the rollup then
    /// says, rather than in a week that never happened.
    #[test]
    fn a_date_that_is_missing_unreadable_or_implausible_is_undated() {
        assert_eq!(
            parse(&eml("Subject: [rocm-doctor] x", REPORT)).week,
            None,
            "no Date header"
        );
        for date in [
            "yesterday",
            "",
            "Thu, 29 Oct 2026 25:00:00 +0000", // not a time
            "Fri, 29 Oct 2026 10:00:00 +0000", // the weekday does not match the date
            "Mon, 01 Jan 2024 10:00:00 +0000", // before any report existed
            "Sun, 04 Oct 2026 12:00:00 +0000", // the day before the code first landed
            "Thu, 29 Oct 2099 10:00:00 +0000", // in the future
            // A comment that never closes is malformed, not a comment.
            "Thu, 29 Oct 2026 10:00:00 +0200 (unclosed",
        ] {
            assert_eq!(week_for(date), None, "{date:?}");
        }
        // The day the code first landed is the first plausible day.
        assert_eq!(
            week_for("Mon, 05 Oct 2026 00:00:00 +0000").as_deref(),
            Some("2026-10-05")
        );
        // One day ahead of today is allowed, for clocks that disagree. Two is not.
        assert!(week_for("Sat, 16 Jan 2027 10:00:00 +0000").is_some());
        assert_eq!(week_for("Sun, 17 Jan 2027 10:00:00 +0000"), None);
    }

    /// Only the top-level `Date` counts. A part of the message may carry its own
    /// headers, and those are not when the message was sent.
    #[test]
    fn a_date_inside_a_mime_part_is_not_the_date_of_the_message() {
        // The part is dated Thursday 5 November, in the week of 2026-11-02.
        let body = format!("--q\nDate: Thu, 05 Nov 2026 10:00:00 +0000\n\n{REPORT}\n--q--\n");
        let headers = "Subject: [rocm-doctor] x\nContent-Type: multipart/mixed; boundary=q";

        // With a top-level `Date` as well, the message is in the top-level week.
        let both = parse(&eml(
            &format!("{headers}\nDate: Thu, 29 Oct 2026 10:00:00 +0000"),
            &body,
        ));
        assert_eq!(
            both.week.map(|week| week.to_string()).as_deref(),
            Some("2026-10-26"),
            "the part's date must not stand in for the message's"
        );
        // With none at the top level, the part's does not fill the gap.
        assert_eq!(parse(&eml(headers, &body)).week, None);
    }

    #[test]
    fn a_message_with_no_text_part_has_no_body() {
        let message = parse(&eml(
            "Subject: [rocm-doctor] x\nContent-Type: multipart/mixed; boundary=q",
            "--q\nContent-Type: application/octet-stream\n\nAAAA\n--q--\n",
        ));
        assert_eq!(message.body, None);
    }
}
