// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! How a report leaves the machine, which is only ever by the user's own act.
//!
//! This module builds a link and decides whether a browser may be started. It
//! never sends anything, and there is deliberately no code here that could:
//! no HTTP client, no token, nothing to authenticate with. A report reaches a
//! tracker because a person read it and pressed a button.
//!
//! Kept apart from [`crate::report`], which decides what a report *contains*.
//! That module is pure by design and says so; this one is where the outside
//! world starts, so the boundary is worth keeping visible.

use crate::report::{Report, UNRECOGNISED};

/// Where a report is sent.
///
/// Fixed in code rather than configurable on purpose: an address a caller can
/// choose is an address an attacker can choose, and the user would be reading a
/// report they believe is going to AMD while it goes somewhere else.
///
/// A mailbox rather than an issue tracker. That choice costs the report its
/// anonymity, because a mail envelope carries the sender's address whatever the
/// body says, and it costs the ability to count reports, because a mailbox has
/// no query. Both are recorded where the decision was made rather than here.
pub const DESTINATION: &str = "ROCmCLI@amd.com";

/// The first word of every subject line, so a mail rule can route the whole set.
pub const SUBJECT_TAG: &str = "[rocm-doctor]";

/// What a subject says when the catalog recognised nothing.
pub const UNRECOGNISED_SUBJECT: &str = "unrecognised";

/// What should happen next, decided here and performed by the caller.
///
/// A value rather than an action, so the decision can be tested without a
/// browser, a network, or a display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delivery {
    /// Hand this to the user's mail client, then tell them what was opened.
    Open(String),
    /// Show this and start nothing.
    ///
    /// The headless case, and the honest default whenever there is doubt. A
    /// link on screen costs a user one paste, while a mail client started on a
    /// machine they are holding over SSH is a process they did not ask for on
    /// a display that is not theirs.
    ///
    /// This case matters more for mail than it did for a web link. Servers,
    /// lab machines and containers usually have no mail client at all, so the
    /// link would fail silently rather than open anything.
    Show(String),
}

/// The subject line: a routing tag, the cause, and the coarse description.
///
/// A mailbox has no labels, so the classification an issue would carry in
/// metadata has to live somewhere a mail rule and a human scanning an inbox can
/// both read. The subject is the only such place.
///
/// Ordered most stable first, so a sorted inbox groups by cause and then by
/// machine. Nothing is here that the body does not already carry: a subject is
/// as public as the body, and a field that is not approved for one is not
/// approved for the other.
#[must_use]
pub fn subject_for(report: &Report) -> String {
    let cause = if report.entry == UNRECOGNISED {
        UNRECOGNISED_SUBJECT
    } else {
        report.entry.as_str()
    };
    format!(
        "{SUBJECT_TAG} {cause} on {} / {}-{}",
        report.architecture, report.distro, report.os_major
    )
}

/// The prefilled mail link for a report.
///
/// The body is the report as the user was shown it. Nothing is added here: a
/// field that is not in [`Report`] has not been through the approved-field
/// check, and this is exactly the seam where "just one more useful detail"
/// would bypass it.
#[must_use]
pub fn report_url(report: &Report) -> String {
    mail_to(DESTINATION, report)
}

/// The link builder, separated from the destination so it can be exercised
/// against an address that is not the real mailbox. A test that sends to the
/// real one would be indistinguishable from a bug that does.
fn mail_to(destination: &str, report: &Report) -> String {
    // `expect` rather than a fallible return: every field of `Report` is a
    // `String`, a `u32` or a `bool`, so this has no failing case to handle,
    // and inventing one would add a branch no test could ever reach.
    let body = serde_json::to_string_pretty(report).expect("a report has no unserializable field");
    // The address is not escaped. RFC 6068 allows `@` and `.` unescaped in the
    // address part, and a percent-escaped `@` there is handled poorly by some
    // mail clients. This is safe because the address is a compile-time
    // constant this crate owns, never a value read from a machine. The query
    // values that follow are escaped, because they carry machine-derived text.
    format!(
        "mailto:{destination}?subject={}&body={}",
        percent_encode(&subject_for(report)),
        percent_encode(&body),
    )
}

/// Whether a browser may be started for this user.
///
/// Reads the environment rather than probing anything, and every unknown
/// answers no. Starting a browser is the one irreversible thing this module
/// can do, so it happens only where there is positive evidence of a desktop
/// the user is sitting at.
#[must_use]
pub fn may_open_browser(env: &dyn Fn(&str) -> Option<String>) -> bool {
    let set = |key: &str| env(key).is_some_and(|value| !value.trim().is_empty());

    // A session reached over SSH belongs to a display somewhere else. Opening
    // a browser here either fails or opens it on a machine the user is not
    // looking at.
    if set("SSH_CONNECTION") || set("SSH_CLIENT") || set("SSH_TTY") {
        return false;
    }
    // An explicit opt-out is honoured before any positive evidence: a user who
    // said no has said no.
    if set("ROCM_NO_BROWSER") {
        return false;
    }
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        return true;
    }
    // On Linux a desktop is not implied by anything except a display.
    set("DISPLAY") || set("WAYLAND_DISPLAY")
}

/// Decide what to do with a report.
///
/// `sending` is whether the user asked to be taken to a prefilled mail, rather
/// than only to read what it would say. Both conditions have to hold before a
/// mail client starts: the user asked, and this looks like a desktop they are
/// sitting at. Either one alone is not enough, and the user's is checked
/// first, because a machine that could open a mail client is not a reason to.
#[must_use]
pub fn deliver(report: &Report, sending: bool, env: &dyn Fn(&str) -> Option<String>) -> Delivery {
    choose(report_url(report), sending, env)
}

/// The choice itself, taking the link rather than building it, so the two
/// branches can be tested without reaching the real mailbox.
fn choose(url: String, sending: bool, env: &dyn Fn(&str) -> Option<String>) -> Delivery {
    if sending && may_open_browser(env) {
        Delivery::Open(url)
    } else {
        Delivery::Show(url)
    }
}

/// Percent-encode for a query-string value.
///
/// Written out rather than taken from a crate: this is the only encoding this
/// binary needs, and a signed artifact is not worth a dependency for fifteen
/// lines. Everything outside the unreserved set of RFC 3986 is escaped, which
/// is stricter than necessary and wrong in no case.
fn percent_encode(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(byte as char);
        } else {
            out.push('%');
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examine::{Examination, Gpu};
    use crate::report::prepare_report;

    /// A machine that produces a report, so the link under test is built from
    /// what the product actually emits rather than a hand-written `Report`.
    fn reportable_machine() -> Examination {
        Examination {
            os_family: "linux".to_owned(),
            distro_id: "ubuntu".to_owned(),
            distro_version: "22.04".to_owned(),
            os_version: "#1 SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC 2026".to_owned(),
            has_amd_gpu: true,
            // Markers rather than plausible values, so a leak into the subject
            // or the body is visible by eye in a failure message instead of
            // reading like a real machine.
            user_name: "SENTINEL-USER".to_owned(),
            rocm_path: "/SENTINEL-PATH/rocm".to_owned(),
            cpu_model: "SENTINEL-CPU".to_owned(),
            gpus: vec![Gpu {
                name: "SENTINEL-MARKETING-NAME".to_owned(),
                gfx_target: "gfx1100".to_owned(),
                pci_id: "SENTINEL-PCI".to_owned(),
                is_apu: Some(false),
                is_amd: true,
            }],
            ..Examination::default()
        }
    }

    fn report_of(entry: Option<&str>) -> Report {
        prepare_report(&reportable_machine(), entry, false)
            .expect("a released machine must produce a report")
    }

    /// No environment at all, which is the headless shape.
    ///
    /// Linux-only, like its two call sites: on Windows and macOS
    /// [`may_open_browser`] answers yes without consulting the environment, so
    /// there is no "no display" case to construct there, and an unguarded
    /// helper would be dead code on those targets -- which is exactly what
    /// failed the Windows lane here before this was guarded.
    #[cfg(target_os = "linux")]
    fn no_env() -> impl Fn(&str) -> Option<String> {
        |_| None
    }

    fn env_of(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |key| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| (*v).to_owned())
        }
    }

    /// The link carries the report and nothing else.
    ///
    /// Compared as JSON rather than as a `Report`, which is the whole point.
    /// Deserializing into `Report` discards fields the struct does not know,
    /// so a body carrying an extra `"hostname"` beside the approved fields
    /// round-trips to an identical `Report` and passes. Found by mutation: a
    /// planted leak survived the first version of this test. Comparing the
    /// parsed values keeps every key, including ones nothing agreed to.
    #[test]
    fn the_link_body_is_the_report_the_user_was_shown_and_nothing_more() {
        let report = report_of(None);
        let url = mail_to("nobody@example.invalid", &report);

        let body = query_value(&url, "body").expect("the link must carry a body");
        let carried: serde_json::Value =
            serde_json::from_str(body.trim()).expect("the body must be the report as JSON");
        let approved = serde_json::to_value(&report).expect("a report must serialize");

        assert_eq!(
            carried, approved,
            "the link body is not exactly the report the user approved. An added field has not \
             been through the approved-field check, and this seam is where one would be added"
        );
    }

    /// The link addresses the mailbox and carries the subject.
    #[test]
    fn the_link_addresses_the_destination_and_carries_the_subject() {
        let report = report_of(Some("fix-6-path"));
        let url = report_url(&report);

        assert!(
            url.starts_with(&format!("mailto:{DESTINATION}?")),
            "a report has to address the mailbox it is meant for: {url}"
        );
        assert_eq!(
            query_value(&url, "subject").as_deref(),
            Some(subject_for(&report).as_str()),
            "the subject is the only classification a mailbox can route on, so it has to \
             survive the link: {url}"
        );
    }

    /// The subject names the cause, because a mailbox has no labels.
    ///
    /// Paired, so "always says unrecognised" cannot satisfy the first half.
    #[test]
    fn the_subject_names_the_cause_so_a_mailbox_can_be_sorted_by_it() {
        let unrecognised = subject_for(&report_of(None));
        assert!(unrecognised.starts_with(SUBJECT_TAG));
        assert!(
            unrecognised.contains(UNRECOGNISED_SUBJECT),
            "a report with no matched cause has to say so in the subject: {unrecognised}"
        );

        let recognised = subject_for(&report_of(Some("fix-6-path")));
        assert!(recognised.starts_with(SUBJECT_TAG));
        assert!(
            recognised.contains("fix-6-path"),
            "premise failed: a matched entry has to reach the subject, or the case above is \
             satisfied by never naming a cause: {recognised}"
        );
        assert!(
            !recognised.contains(UNRECOGNISED_SUBJECT),
            "a matched entry must not also be called unrecognised: {recognised}"
        );
    }

    /// The subject carries nothing the body does not.
    ///
    /// A subject line is as public as the body and travels further, since it
    /// shows in an inbox list. Anything here that is not an approved field has
    /// bypassed the field check by a side door.
    #[test]
    fn the_subject_carries_no_field_the_report_does_not() {
        let machine = reportable_machine();
        let report = report_of(None);
        let subject = subject_for(&report);

        for planted in [
            machine.user_name.as_str(),
            machine.rocm_path.as_str(),
            machine.cpu_model.as_str(),
            machine.gpus[0].pci_id.as_str(),
            machine.gpus[0].name.as_str(),
        ] {
            if planted.is_empty() {
                continue;
            }
            assert!(
                !subject.contains(planted),
                "'{planted}' reached the subject line: {subject}"
            );
        }
    }

    /// A session reached over SSH is never given a browser, on any platform.
    ///
    /// Both refusals here are checked before the platform is consulted, so
    /// they hold everywhere and are asserted unconditionally. The rules that
    /// depend on a display live in the Linux-only test below, because Windows
    /// and macOS have no `DISPLAY` to reason about and
    /// [`may_open_browser`] treats them as a desktop outright.
    #[test]
    fn a_session_over_ssh_is_shown_the_link_rather_than_having_a_browser_started() {
        assert!(
            !may_open_browser(&env_of(&[
                ("DISPLAY", ":0"),
                ("SSH_CONNECTION", "10.0.0.1 22")
            ])),
            "a display variable does not make an SSH session local"
        );
        assert!(
            !may_open_browser(&env_of(&[("DISPLAY", ":0"), ("ROCM_NO_BROWSER", "1")])),
            "an explicit opt-out is not overridden by a display"
        );

        // The premise. Without it both assertions above are satisfied by a
        // function that refuses everything, which would take the feature with
        // it. Linux-only because it is the platform that needs evidence: see
        // the test below.
        #[cfg(target_os = "linux")]
        assert!(
            may_open_browser(&env_of(&[("DISPLAY", ":0")])),
            "premise failed: a plain local display must be allowed"
        );
    }

    /// On Linux, a desktop has to be evidenced, and an empty variable is not
    /// evidence.
    ///
    /// Linux-only, and that is the point rather than a convenience. Windows
    /// and macOS have no `DISPLAY`, so [`may_open_browser`] answers yes there
    /// without looking at the environment at all, and asserting the Linux rule
    /// on them tests nothing about either platform. The first version of this
    /// was not guarded and failed the Windows lane, having passed locally.
    #[cfg(target_os = "linux")]
    #[test]
    fn an_empty_display_variable_does_not_count_as_a_desktop_on_linux() {
        assert!(!may_open_browser(&env_of(&[("DISPLAY", "")])));
        assert!(!may_open_browser(&env_of(&[("DISPLAY", "   ")])));
        assert!(
            !may_open_browser(&no_env()),
            "no display at all is not a desktop either"
        );

        // Paired, so the three refusals above cannot be satisfied by refusing
        // everything.
        assert!(
            may_open_browser(&env_of(&[("WAYLAND_DISPLAY", "wayland-0")])),
            "premise failed: a Wayland display is evidence of a desktop"
        );
    }

    /// The destination is the agreed mailbox and nothing else.
    ///
    /// Pinned as a literal because it is the one value in this module that
    /// decides where a user's machine description goes. A typo here sends
    /// every report somewhere nobody is watching, or somewhere nobody should
    /// be watching, and no other test would notice.
    #[test]
    fn reports_address_the_agreed_mailbox() {
        assert_eq!(DESTINATION, "ROCmCLI@amd.com");
        assert!(
            report_url(&report_of(None)).contains(DESTINATION),
            "the destination has to survive into the link the user is handed"
        );
    }

    /// Reading a report is not asking to file one.
    ///
    /// Two conditions gate a browser, and this covers the one the machine
    /// cannot tell you: the user has to have asked. A desktop is permission
    /// from the environment, never from the person. Without this, adding a
    /// display to a machine would change what `--report` does.
    #[test]
    fn a_desktop_is_not_permission_to_open_anything_the_user_did_not_ask_for() {
        let url = "https://example.invalid/new".to_owned();
        let desktop = env_of(&[("DISPLAY", ":0")]);

        assert_eq!(
            choose(url.clone(), false, &desktop),
            Delivery::Show(url.clone()),
            "a report the user only asked to read must not open a browser, whatever the \
             machine looks like"
        );

        // The premise. Without this the assertion above is satisfied by never
        // opening anything, which would take the feature with it. Every
        // platform reaches this: a `DISPLAY` is evidence on Linux, and
        // Windows and macOS are a desktop regardless.
        assert_eq!(
            choose(url.clone(), true, &desktop),
            Delivery::Open(url.clone()),
            "premise failed: asking, on a desktop, has to open"
        );

        // The machine's half of the gate, which only Linux can express. On
        // Windows and macOS there is no environment that means "not a
        // desktop", so asserting this there would test nothing.
        #[cfg(target_os = "linux")]
        assert_eq!(
            choose(url.clone(), true, &no_env()),
            Delivery::Show(url.clone()),
            "asking does not override a machine with no desktop"
        );

        // The user's half, which every platform can express, because an
        // explicit opt-out is honoured before the platform is consulted.
        assert_eq!(
            choose(
                url.clone(),
                true,
                &env_of(&[("DISPLAY", ":0"), ("ROCM_NO_BROWSER", "1")])
            ),
            Delivery::Show(url),
            "an opt-out has to hold on every platform, not only where a display is read"
        );
    }

    /// Everything outside the unreserved set is escaped.
    #[test]
    fn a_value_is_escaped_so_it_cannot_end_the_query_or_start_a_new_field() {
        assert_eq!(percent_encode("a b"), "a%20b");
        assert_eq!(percent_encode("a&labels=x"), "a%26labels%3Dx");
        assert_eq!(percent_encode("a#b"), "a%23b");
        assert_eq!(percent_encode("-._~"), "-._~");
        assert_eq!(percent_encode("é"), "%C3%A9");
    }

    /// Read one query-string value back out of a link, decoded.
    fn query_value(url: &str, key: &str) -> Option<String> {
        let query = url.split_once('?')?.1;
        let raw = query
            .split('&')
            .find_map(|pair| pair.strip_prefix(&format!("{key}=")))?;
        let bytes = raw.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%' && i + 2 < bytes.len() {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
                out.push(u8::from_str_radix(hex, 16).ok()?);
                i += 3;
            } else {
                out.push(bytes[i]);
                i += 1;
            }
        }
        String::from_utf8(out).ok()
    }
}
