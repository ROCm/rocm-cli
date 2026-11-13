// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which engine protocol a request or an answer was written in, and whether
//! the other side can use it.
//!
//! Every request and every answer carries `protocol_version`, the
//! [`ENGINE_PROTOCOL_VERSION`] of the side that wrote it, as a
//! `MAJOR.MINOR.PATCH` string. A message without one was written before the
//! field existed and is read as [`PRE_VERSIONING_PROTOCOL`] (`0.1.0`), so an
//! engine or a daemon from an earlier release keeps working with this one.
//!
//! Two versions are compatible by the caret rule: while the major is `0`, the
//! major and the minor must both match (`0.1.0` and `0.1.9` are, `0.1.0` and
//! `0.2.0` are not); from `1.0.0` on, the major alone must match. The patch
//! never decides, in either direction. A version that is not three dot-separated
//! numbers cannot be compared with anything, and is treated as incompatible
//! and reported as unreadable rather than as a different version.
//!
//! Rules for raising [`ENGINE_PROTOCOL_VERSION`], following
//! `ENGINE_RECIPE_CONTRACT_VERSION` and the `rocm fix` catalog contract: an
//! **addition** — a new optional field, a new method an older peer never sends —
//! raises only the patch, which every peer on the same minor (major, from
//! `1.0.0`) still accepts. A **removal or a type change** — a field dropped,
//! renamed, made required or given another type — raises the minor while the
//! major is `0`, and the major from `1.0.0` on, so that a peer that would
//! misread the message refuses it instead of guessing.
//!
//! An engine refuses a request it cannot serve with an error answer whose code
//! is [`INCOMPATIBLE_PROTOCOL_CODE`] ([`incompatible_request_response`]), before
//! it reads the payload. The client checks an answer's version before anything
//! else in it (`decode_engine_response`), and before an `install` or a
//! `launch` it first asks the engine for its capabilities and checks the
//! version of that answer, so a request that changes the machine is never sent
//! to an engine that cannot read it.

use std::fmt;

use crate::{ENGINE_PROTOCOL_VERSION, EngineRequestEnvelope, EngineResponseEnvelope};

/// The version a request or an answer without `protocol_version` was written
/// in: the format from before the field existed.
pub const PRE_VERSIONING_PROTOCOL: &str = "0.1.0";

/// The error code an engine answers with when a request was made for a
/// protocol it cannot serve.
pub const INCOMPATIBLE_PROTOCOL_CODE: &str = "incompatible_protocol";

/// Why two engine protocol versions cannot be used together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolMismatch {
    /// Both versions are readable, and the caret rule says they differ.
    Incompatible { ours: String, theirs: String },
    /// The other side's version is not a `MAJOR.MINOR.PATCH` version.
    Unreadable { ours: String, theirs: String },
}

impl ProtocolMismatch {
    /// The version this side speaks.
    #[must_use]
    pub fn ours(&self) -> &str {
        match self {
            Self::Incompatible { ours, .. } | Self::Unreadable { ours, .. } => ours,
        }
    }

    /// The version the other side named, as it named it.
    #[must_use]
    pub fn theirs(&self) -> &str {
        match self {
            Self::Incompatible { theirs, .. } | Self::Unreadable { theirs, .. } => theirs,
        }
    }
}

impl fmt::Display for ProtocolMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incompatible { ours, theirs } => write!(
                f,
                "engine protocol {ours} cannot be used with engine protocol {theirs}"
            ),
            Self::Unreadable { ours, theirs } => write!(
                f,
                "engine protocol {ours} cannot be compared with protocol version {theirs:?}, \
                 which is not a MAJOR.MINOR.PATCH version"
            ),
        }
    }
}

impl std::error::Error for ProtocolMismatch {}

/// `version` as its three numbers, or `None` when it is not exactly
/// `MAJOR.MINOR.PATCH` with each part made of ASCII digits.
fn parse_version(version: &str) -> Option<(u64, u64, u64)> {
    let mut parts = version.split('.');
    let mut number = || {
        let part = parts.next()?;
        if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        part.parse::<u64>().ok()
    };
    let parsed = (number()?, number()?, number()?);
    if parts.next().is_some() {
        return None;
    }
    Some(parsed)
}

/// Whether a side that speaks `ours` can use a peer that speaks `theirs`.
///
/// The rule is the caret rule in the module documentation. `None` is a message
/// written before the field existed, read as [`PRE_VERSIONING_PROTOCOL`].
///
/// `ours` is this build's own constant; should it ever fail to parse, nothing
/// is compatible with it, which a test of [`ENGINE_PROTOCOL_VERSION`] rules out.
pub fn engine_protocol_compatible(
    ours: &str,
    theirs: Option<&str>,
) -> Result<(), ProtocolMismatch> {
    let theirs = theirs.unwrap_or(PRE_VERSIONING_PROTOCOL);
    let Some(their_version) = parse_version(theirs) else {
        return Err(ProtocolMismatch::Unreadable {
            ours: ours.to_owned(),
            theirs: theirs.to_owned(),
        });
    };
    let compatible = parse_version(ours).is_some_and(|our_version| {
        if our_version.0 == 0 || their_version.0 == 0 {
            our_version.0 == their_version.0 && our_version.1 == their_version.1
        } else {
            our_version.0 == their_version.0
        }
    });
    if compatible {
        Ok(())
    } else {
        Err(ProtocolMismatch::Incompatible {
            ours: ours.to_owned(),
            theirs: theirs.to_owned(),
        })
    }
}

/// The answer an engine gives instead of serving `request` when the request
/// was made for a protocol this build cannot serve, or `None` when it can.
///
/// Each engine calls this before it decodes the payload, so a request whose
/// payload means something else in another protocol is never acted on.
#[must_use]
pub fn incompatible_request_response(
    request: &EngineRequestEnvelope,
) -> Option<EngineResponseEnvelope> {
    let mismatch =
        engine_protocol_compatible(ENGINE_PROTOCOL_VERSION, request.protocol_version.as_deref())
            .err()?;
    let message = match &mismatch {
        ProtocolMismatch::Incompatible { ours, theirs } => format!(
            "this engine speaks engine protocol {ours}; the request was made for protocol \
             {theirs}, which it cannot serve"
        ),
        ProtocolMismatch::Unreadable { ours, theirs } => format!(
            "this engine speaks engine protocol {ours}; the request named protocol version \
             {theirs:?}, which is not a MAJOR.MINOR.PATCH version, so it cannot be served"
        ),
    };
    Some(EngineResponseEnvelope::failure(
        INCOMPATIBLE_PROTOCOL_CODE,
        message,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineMethod;
    use serde_json::json;

    /// The compatibility rule as a table, from this build's own version.
    /// Runs on every platform, Windows included.
    #[test]
    fn the_caret_rule_from_this_builds_version() {
        assert_eq!(
            ENGINE_PROTOCOL_VERSION, "0.1.0",
            "the table below is written for 0.1.0"
        );
        let incompatible = |theirs: &str| ProtocolMismatch::Incompatible {
            ours: "0.1.0".to_owned(),
            theirs: theirs.to_owned(),
        };
        let unreadable = |theirs: &str| ProtocolMismatch::Unreadable {
            ours: "0.1.0".to_owned(),
            theirs: theirs.to_owned(),
        };
        let table: &[(Option<&str>, Result<(), ProtocolMismatch>)] = &[
            (None, Ok(())),
            (Some("0.1.0"), Ok(())),
            (Some("0.1.9"), Ok(())),
            (Some("0.2.0"), Err(incompatible("0.2.0"))),
            (Some("0.0.1"), Err(incompatible("0.0.1"))),
            (Some("1.0.0"), Err(incompatible("1.0.0"))),
            (Some("9.0.0"), Err(incompatible("9.0.0"))),
            (Some(""), Err(unreadable(""))),
            (Some("banana"), Err(unreadable("banana"))),
            (Some("1.2"), Err(unreadable("1.2"))),
            (Some("1.2.3.4"), Err(unreadable("1.2.3.4"))),
            (Some("0.1.0-rc1"), Err(unreadable("0.1.0-rc1"))),
            (Some("v0.1.0"), Err(unreadable("v0.1.0"))),
            (Some("0..1"), Err(unreadable("0..1"))),
        ];
        for (theirs, expected) in table {
            assert_eq!(
                &engine_protocol_compatible(ENGINE_PROTOCOL_VERSION, *theirs),
                expected,
                "theirs: {theirs:?}"
            );
        }
    }

    /// From `1.0.0` on, only the major decides.
    #[test]
    fn past_one_the_major_alone_decides() {
        assert_eq!(engine_protocol_compatible("1.2.0", Some("1.9.4")), Ok(()));
        assert_eq!(engine_protocol_compatible("1.2.0", Some("1.0.0")), Ok(()));
        assert!(engine_protocol_compatible("1.2.0", Some("2.0.0")).is_err());
        assert!(engine_protocol_compatible("1.2.0", Some("0.2.0")).is_err());
        // A peer that sends nothing is 0.1.0, which a 1.x side cannot use.
        assert!(engine_protocol_compatible("1.0.0", None).is_err());
    }

    #[test]
    fn this_builds_version_is_readable() {
        assert!(parse_version(ENGINE_PROTOCOL_VERSION).is_some());
        assert!(parse_version(PRE_VERSIONING_PROTOCOL).is_some());
    }

    #[test]
    fn a_request_this_engine_can_serve_gets_no_refusal() {
        for version in [None, Some("0.1.0"), Some("0.1.7")] {
            let mut request = EngineRequestEnvelope::new(EngineMethod::Stop, json!({}));
            request.protocol_version = version.map(str::to_owned);
            assert!(
                incompatible_request_response(&request).is_none(),
                "{version:?}"
            );
        }
    }

    #[test]
    fn a_request_for_another_protocol_is_refused_naming_both_versions() {
        let mut request = EngineRequestEnvelope::new(EngineMethod::Install, json!({}));
        request.protocol_version = Some("0.2.0".to_owned());
        let response = incompatible_request_response(&request).expect("refused");
        assert!(!response.ok);
        assert_eq!(
            response.protocol_version.as_deref(),
            Some(ENGINE_PROTOCOL_VERSION)
        );
        let error = response.error.expect("an error");
        assert_eq!(error.code, INCOMPATIBLE_PROTOCOL_CODE);
        assert_eq!(
            error.message,
            "this engine speaks engine protocol 0.1.0; the request was made for protocol 0.2.0, \
             which it cannot serve"
        );

        request.protocol_version = Some("banana".to_owned());
        let error = incompatible_request_response(&request)
            .expect("refused")
            .error
            .expect("an error");
        assert_eq!(error.code, INCOMPATIBLE_PROTOCOL_CODE);
        assert_eq!(
            error.message,
            "this engine speaks engine protocol 0.1.0; the request named protocol version \
             \"banana\", which is not a MAJOR.MINOR.PATCH version, so it cannot be served"
        );
    }
}
