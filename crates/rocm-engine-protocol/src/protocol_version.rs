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
//! never decides, in either direction. A `MAJOR.MINOR.PATCH` version here is
//! three dot-separated plain decimal numbers, each without a leading zero
//! (`0` itself aside) and fitting in 64 bits. Anything else cannot be compared
//! with anything, and is treated as incompatible and reported as unreadable
//! rather than as a different version.
//!
//! [`ENGINE_PROTOCOL_VERSION`] `0.1.0` is the baseline: it is the release that
//! introduced the field, and it speaks the same format as the messages written
//! before it, which is why a missing field and `0.1.0` mean the same thing.
//!
//! Rules for raising [`ENGINE_PROTOCOL_VERSION`] from there. They are this
//! protocol's own: the recipe hint's `ENGINE_RECIPE_CONTRACT_VERSION` is
//! compared as an exact string, and the `rocm fix` catalog keeps its number on
//! an addition, so neither is the model here.
//! - An **addition** an older peer can ignore — a new optional field, a new
//!   method only an engine ever offers — raises the patch, which every peer on
//!   the same minor (major, from `1.0.0`) still accepts.
//! - A **new method a client may send** raises the minor while the major is
//!   `0`, and the major after that: an older engine cannot decode it and would
//!   answer `unknown_method` (see [`answer_engine_request`]). The exception is a
//!   method the client sends only to an engine whose capabilities say it has it.
//! - A **removal or a type change** — a field dropped, renamed, made required or
//!   given another type — raises the minor while the major is `0`, and the major
//!   from `1.0.0` on, so that a peer that would misread the message refuses it
//!   instead of guessing.
//!
//! An engine reads a request's version before anything else in it, its method
//! included, and refuses a request it cannot serve with an error answer whose
//! code is [`INCOMPATIBLE_PROTOCOL_CODE`] ([`answer_engine_request`] for a
//! request read from stdin, [`incompatible_request_response`] for one already
//! decoded). The client checks the version of an answer that is a single JSON
//! object before anything else in it, and before an `install` or a `launch`
//! it first asks the engine for its capabilities and checks the version of
//! that answer, so a request that installs or starts something is never sent
//! to an engine that cannot read it.

use serde_json::Value;

use crate::{ENGINE_PROTOCOL_VERSION, EngineMethod, EngineRequestEnvelope, EngineResponseEnvelope};

/// The version a request or an answer without `protocol_version` was written
/// in: the format from before the field existed.
pub const PRE_VERSIONING_PROTOCOL: &str = "0.1.0";

/// The error code an engine answers with when a request was made for a
/// protocol it cannot serve.
pub const INCOMPATIBLE_PROTOCOL_CODE: &str = "incompatible_protocol";

/// The error code an engine answers with when a request in a protocol it can
/// serve names a method it does not have.
pub const UNKNOWN_METHOD_CODE: &str = "unknown_method";

/// The error code an engine answers with when what it was sent is not a
/// request envelope at all.
pub const INVALID_REQUEST_CODE: &str = "invalid_request";

/// The error code an engine answers with when a request's payload does not
/// decode as that method's request.
pub const INVALID_PAYLOAD_CODE: &str = "invalid_payload";

/// Why two engine protocol versions cannot be used together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolMismatch {
    /// Both versions are readable, and the caret rule says they differ.
    Incompatible { ours: String, theirs: String },
    /// The other side's version is not a `MAJOR.MINOR.PATCH` version.
    Unreadable { ours: String, theirs: String },
}

/// `version` as its three numbers, or `None` when it is not exactly
/// `MAJOR.MINOR.PATCH` as the module documentation defines it.
fn parse_version(version: &str) -> Option<(u64, u64, u64)> {
    let mut parts = version.split('.');
    let mut number = || {
        let part = parts.next()?;
        if part.is_empty()
            || !part.bytes().all(|byte| byte.is_ascii_digit())
            || (part.len() > 1 && part.starts_with('0'))
        {
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
/// payload means something else in another protocol is never acted on. A
/// request read from text goes through [`answer_engine_request`], which judges
/// the version before the method is even decoded.
#[must_use]
pub fn incompatible_request_response(
    request: &EngineRequestEnvelope,
) -> Option<EngineResponseEnvelope> {
    refusal_for(request.protocol_version.as_deref())
}

/// The `incompatible_protocol` answer to a request that named `theirs`, or
/// `None` when this build can serve it.
fn refusal_for(theirs: Option<&str>) -> Option<EngineResponseEnvelope> {
    let mismatch = engine_protocol_compatible(ENGINE_PROTOCOL_VERSION, theirs).err()?;
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

/// The `protocol_version` a JSON object names, read without decoding the
/// rest of it: `None` when it is absent or `null`, the string itself, or, for
/// any other type, its JSON text (which is never a readable version). Both
/// sides use it to judge a message's version before decoding the message.
pub(crate) fn raw_protocol_version(value: &Value) -> Option<String> {
    match value.get("protocol_version") {
        None | Some(Value::Null) => None,
        Some(Value::String(version)) => Some(version.clone()),
        Some(other) => Some(other.to_string()),
    }
}

/// Answer the request an engine read from its stdin, as `text`: judge its
/// `protocol_version` first, then decode the rest and hand it to `handle`.
///
/// The version is read from the request as plain JSON before the envelope is
/// decoded, so a request made for another protocol is refused with
/// [`INCOMPATIBLE_PROTOCOL_CODE`] whatever its method is (a later protocol may
/// add methods this build cannot decode) and whatever type its version has (a
/// version that is present and neither a string nor `null` is unreadable, like
/// one that is not `MAJOR.MINOR.PATCH`; `null` is read like a missing version,
/// as [`PRE_VERSIONING_PROTOCOL`]). A request this build can serve that names a
/// method it does not have is answered with [`UNKNOWN_METHOD_CODE`], and text
/// that is not a request envelope at all, including JSON that is not an object,
/// with [`INVALID_REQUEST_CODE`]. Every outcome is an answer envelope. Input
/// that cannot be read as text (not valid UTF-8) never reaches this function:
/// the engine's stdio command fails reading it, and there is no answer.
pub fn answer_engine_request(
    text: &str,
    handle: impl FnOnce(EngineRequestEnvelope) -> EngineResponseEnvelope,
) -> EngineResponseEnvelope {
    let value: Value = match serde_json::from_str(text) {
        Ok(value) => value,
        Err(error) => {
            return EngineResponseEnvelope::failure(
                INVALID_REQUEST_CODE,
                format!("the request is not JSON: {error}"),
            );
        }
    };
    // A derived struct also decodes from a JSON array of its fields in order,
    // which no client writes; an array has no `protocol_version` to judge, so
    // reading one by position would serve a request whose version was never
    // checked.
    if !value.is_object() {
        return EngineResponseEnvelope::failure(
            INVALID_REQUEST_CODE,
            "the request is not an engine request envelope: it is not a JSON object",
        );
    }
    let version = raw_protocol_version(&value);
    if let Some(refusal) = refusal_for(version.as_deref()) {
        return refusal;
    }
    if let Some(Value::String(method)) = value.get("method")
        && serde_json::from_value::<EngineMethod>(Value::String(method.clone())).is_err()
    {
        return EngineResponseEnvelope::failure(
            UNKNOWN_METHOD_CODE,
            format!(
                "this engine (engine protocol {ENGINE_PROTOCOL_VERSION}) has no method {method:?}"
            ),
        );
    }
    match serde_json::from_value::<EngineRequestEnvelope>(value) {
        Ok(request) => handle(request),
        Err(error) => EngineResponseEnvelope::failure(
            INVALID_REQUEST_CODE,
            format!("the request is not an engine request envelope: {error}"),
        ),
    }
}

/// The protocol contract every engine holds, as one shared check.
///
/// Each engine's tests run it, so the three first-party engines (and `rocm`'s
/// built-in stdio verb) are held to the same list rather than to three
/// hand-copied ones.
#[cfg(any(test, feature = "test-support"))]
pub mod engine_contract {
    use super::{
        ENGINE_PROTOCOL_VERSION, EngineMethod, EngineRequestEnvelope, EngineResponseEnvelope,
        INCOMPATIBLE_PROTOCOL_CODE, INVALID_PAYLOAD_CODE, INVALID_REQUEST_CODE,
        UNKNOWN_METHOD_CODE,
    };

    fn code(answer: &EngineResponseEnvelope) -> Option<&str> {
        answer.error.as_ref().map(|error| error.code.as_str())
    }

    /// The first version the documented bump rule makes incompatible with
    /// `version`: the next minor while the major is `0`, the next major after.
    pub(super) fn breaking_successor(version: &str) -> String {
        let (major, minor, _) =
            super::parse_version(version).expect("a protocol version is MAJOR.MINOR.PATCH");
        if major == 0 {
            format!("0.{}.0", minor + 1)
        } else {
            format!("{}.0.0", major + 1)
        }
    }

    #[track_caller]
    fn assert_refused(engine: &str, request: &str, answer: &EngineResponseEnvelope, code_: &str) {
        assert!(!answer.ok, "{engine}: {request} was served");
        assert_eq!(
            code(answer),
            Some(code_),
            "{engine}: {request}: {:?}",
            answer.error
        );
        assert_eq!(
            answer.protocol_version.as_deref(),
            Some(ENGINE_PROTOCOL_VERSION),
            "{engine}: {request}: the answer must name this build's protocol"
        );
    }

    /// Check `engine` against the contract, through `answer_text` (what its
    /// stdio command does with the text it read) and `handle` (its decoded
    /// envelope handler, which `rocm` calls in-process):
    /// - a request for an incompatible or unreadable protocol is refused with
    ///   `incompatible_protocol` for every method on both paths, and, on the
    ///   stdio path, for a method this build does not know and for a version
    ///   that is not a string;
    /// - a compatible request naming an unknown method is `unknown_method`, and
    ///   text that is not a request envelope is `invalid_request`;
    /// - `capabilities` in this protocol, and with no version at all, is served,
    ///   and an `install` or `launch` with an empty payload reaches the engine's
    ///   own decoding (`invalid_payload`), with this build's version on every
    ///   answer.
    #[track_caller]
    pub fn assert_engine_holds_protocol_contract(
        engine: &str,
        answer_text: impl Fn(&str) -> EngineResponseEnvelope,
        handle: impl Fn(EngineRequestEnvelope) -> EngineResponseEnvelope,
    ) {
        let breaking = breaking_successor(ENGINE_PROTOCOL_VERSION);
        let breaking_text = format!("{breaking:?}");
        for theirs in [breaking_text.as_str(), r#""banana""#, "2", "[0,1,0]"] {
            for method in EngineMethod::ALL
                .iter()
                .map(EngineMethod::name)
                .chain(["frobnicate"])
            {
                let request = format!(
                    r#"{{"protocol_version":{theirs},"method":"{method}","payload":{{}}}}"#
                );
                assert_refused(
                    engine,
                    &request,
                    &answer_text(&request),
                    INCOMPATIBLE_PROTOCOL_CODE,
                );
            }
        }
        for method in EngineMethod::ALL {
            for theirs in [breaking.as_str(), "banana"] {
                let mut request = EngineRequestEnvelope::new(method.clone(), serde_json::json!({}));
                request.protocol_version = Some(theirs.to_owned());
                let label = format!("decoded {method:?} for {theirs}");
                assert_refused(engine, &label, &handle(request), INCOMPATIBLE_PROTOCOL_CODE);
            }
        }

        let unknown = format!(
            r#"{{"protocol_version":"{ENGINE_PROTOCOL_VERSION}","method":"frobnicate","payload":{{}}}}"#
        );
        assert_refused(
            engine,
            &unknown,
            &answer_text(&unknown),
            UNKNOWN_METHOD_CODE,
        );
        let no_method = format!(r#"{{"protocol_version":"{ENGINE_PROTOCOL_VERSION}"}}"#);
        let by_position = format!(r#"["{ENGINE_PROTOCOL_VERSION}","capabilities",{{}}]"#);
        for garbage in ["not json", no_method.as_str(), "[]", by_position.as_str()] {
            assert_refused(engine, garbage, &answer_text(garbage), INVALID_REQUEST_CODE);
        }

        let current = format!(
            r#"{{"protocol_version":"{ENGINE_PROTOCOL_VERSION}","method":"capabilities"}}"#
        );
        let mut served = vec![current.as_str()];
        // A request with no version is one written before the field existed,
        // `0.1.0`; it is served only while this build can serve that.
        if super::engine_protocol_compatible(ENGINE_PROTOCOL_VERSION, None).is_ok() {
            served.push(r#"{"method":"capabilities","payload":{}}"#);
        }
        for request in served {
            let answer = answer_text(request);
            assert!(answer.ok, "{engine}: {request}: {:?}", answer.error);
            assert_eq!(
                answer.protocol_version.as_deref(),
                Some(ENGINE_PROTOCOL_VERSION),
                "{engine}: {request}: a served answer must name this build's protocol"
            );
        }
        for method in [EngineMethod::Install, EngineMethod::Launch] {
            let label = format!("decoded {method:?} with an empty payload");
            let answer = handle(EngineRequestEnvelope::new(method, serde_json::json!({})));
            assert_refused(engine, &label, &answer, INVALID_PAYLOAD_CODE);
        }
    }
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
            (Some("+0.1.0"), Err(unreadable("+0.1.0"))),
            (Some("0.+1.0"), Err(unreadable("0.+1.0"))),
            (Some("0.01.0"), Err(unreadable("0.01.0"))),
            (Some("00.1.0"), Err(unreadable("00.1.0"))),
            (
                Some("99999999999999999999.0.0"),
                Err(unreadable("99999999999999999999.0.0")),
            ),
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
        for version in [None, Some(ENGINE_PROTOCOL_VERSION), Some("0.1.7")] {
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
            format!(
                "this engine speaks engine protocol {ENGINE_PROTOCOL_VERSION}; the request was made for protocol 0.2.0, \
                 which it cannot serve"
            )
        );

        request.protocol_version = Some("banana".to_owned());
        let error = incompatible_request_response(&request)
            .expect("refused")
            .error
            .expect("an error");
        assert_eq!(error.code, INCOMPATIBLE_PROTOCOL_CODE);
        assert_eq!(
            error.message,
            format!(
                "this engine speaks engine protocol {ENGINE_PROTOCOL_VERSION}; the request named protocol version \
                 \"banana\", which is not a MAJOR.MINOR.PATCH version, so it cannot be served"
            )
        );
    }

    /// The handler a request reaches once its version and method are judged:
    /// it serves `capabilities` and rejects every other payload, the shape of
    /// a first-party engine for the cases the contract check sends.
    fn toy_engine(request: EngineRequestEnvelope) -> EngineResponseEnvelope {
        if let Some(refusal) = incompatible_request_response(&request) {
            return refusal;
        }
        match request.method {
            EngineMethod::Capabilities => EngineResponseEnvelope::success(json!({})),
            _ => EngineResponseEnvelope::failure(INVALID_PAYLOAD_CODE, "toy"),
        }
    }

    /// A request is judged by its version before its method: one for another
    /// protocol is refused even when its method or its version's type is one
    /// this build cannot decode, and each other way a request can be unusable
    /// still gets an answer that says so.
    #[test]
    fn a_request_is_judged_by_its_version_before_its_method() {
        let answer = |text: &str| answer_engine_request(text, toy_engine);
        let error = |text: &str| answer(text).error.expect("an error answer");

        let refused = error(r#"{"protocol_version":"0.2.0","method":"frobnicate"}"#);
        assert_eq!(refused.code, INCOMPATIBLE_PROTOCOL_CODE);
        assert_eq!(
            refused.message,
            format!(
                "this engine speaks engine protocol {ENGINE_PROTOCOL_VERSION}; the request was made for protocol 0.2.0, \
                 which it cannot serve"
            )
        );
        let not_a_string = error(r#"{"protocol_version":2,"method":"stop","payload":{}}"#);
        assert_eq!(not_a_string.code, INCOMPATIBLE_PROTOCOL_CODE);
        assert_eq!(
            not_a_string.message,
            format!(
                "this engine speaks engine protocol {ENGINE_PROTOCOL_VERSION}; the request named protocol version \
                 \"2\", which is not a MAJOR.MINOR.PATCH version, so it cannot be served"
            )
        );

        let unknown = error(&format!(
            r#"{{"protocol_version":"{ENGINE_PROTOCOL_VERSION}","method":"frobnicate"}}"#
        ));
        assert_eq!(unknown.code, UNKNOWN_METHOD_CODE);
        assert_eq!(
            unknown.message,
            format!(
                "this engine (engine protocol {ENGINE_PROTOCOL_VERSION}) has no method \"frobnicate\""
            )
        );
        assert_eq!(
            error(r#"{"method":"frobnicate"}"#).code,
            UNKNOWN_METHOD_CODE
        );

        let not_json = error("not json");
        assert_eq!(not_json.code, INVALID_REQUEST_CODE);
        assert_eq!(
            not_json.message,
            "the request is not JSON: expected ident at line 1 column 2"
        );
        let no_method = error(&format!(
            r#"{{"protocol_version":"{ENGINE_PROTOCOL_VERSION}"}}"#
        ));
        assert_eq!(no_method.code, INVALID_REQUEST_CODE);
        assert_eq!(
            no_method.message,
            "the request is not an engine request envelope: missing field `method`"
        );

        assert!(answer(r#"{"method":"capabilities"}"#).ok);
        // A newer request on the same minor, with a field this build does not
        // know, is served.
        assert!(answer(r#"{"protocol_version":"0.1.9","method":"capabilities","hint":"new"}"#).ok);
        assert!(answer(r#"{"protocol_version":null,"method":"capabilities"}"#).ok);
    }

    /// The shared contract check passes an engine that holds the contract,
    /// so a red result in an engine's own test is that engine's.
    #[test]
    fn the_contract_check_passes_an_engine_that_holds_it() {
        engine_contract::assert_engine_holds_protocol_contract(
            "toy",
            |text| answer_engine_request(text, toy_engine),
            toy_engine,
        );
    }

    /// An engine that serves a request whatever protocol it names breaks the
    /// contract, and the shared check says so: a check that passed any engine
    /// would make every engine's contract test meaningless.
    #[test]
    #[should_panic(expected = "was served")]
    fn the_contract_check_fails_an_engine_that_ignores_the_version() {
        fn careless(request: EngineRequestEnvelope) -> EngineResponseEnvelope {
            match request.method {
                EngineMethod::Install | EngineMethod::Launch => {
                    EngineResponseEnvelope::failure(INVALID_PAYLOAD_CODE, "toy")
                }
                _ => EngineResponseEnvelope::success(json!({})),
            }
        }
        engine_contract::assert_engine_holds_protocol_contract(
            "careless",
            |text| match serde_json::from_str::<EngineRequestEnvelope>(text) {
                Ok(request) => careless(request),
                Err(_) => EngineResponseEnvelope::failure(INVALID_REQUEST_CODE, "toy"),
            },
            careless,
        );
    }

    /// An engine whose decoded path refuses but whose stdio path does not read
    /// the version first fails the check too.
    #[test]
    #[should_panic(expected = "incompatible_protocol")]
    fn the_contract_check_fails_an_engine_whose_stdio_skips_the_version() {
        engine_contract::assert_engine_holds_protocol_contract(
            "stdio-skips-version",
            |text| match serde_json::from_str::<EngineRequestEnvelope>(text) {
                Ok(mut request) => {
                    request.protocol_version = None;
                    toy_engine(request)
                }
                Err(_) => EngineResponseEnvelope::failure(INVALID_REQUEST_CODE, "toy"),
            },
            toy_engine,
        );
    }

    /// The version the contract check uses as "incompatible" is the first one
    /// the bump rule makes so, for both branches of the rule, and it is
    /// incompatible with this build's.
    #[test]
    fn the_contract_checks_breaking_version_is_the_next_incompatible_one() {
        assert_eq!(engine_contract::breaking_successor("0.1.0"), "0.2.0");
        assert_eq!(engine_contract::breaking_successor("0.3.7"), "0.4.0");
        assert_eq!(engine_contract::breaking_successor("1.4.2"), "2.0.0");
        assert_eq!(engine_contract::breaking_successor("2.0.0"), "3.0.0");
        let breaking = engine_contract::breaking_successor(ENGINE_PROTOCOL_VERSION);
        assert!(engine_protocol_compatible(ENGINE_PROTOCOL_VERSION, Some(&breaking)).is_err());
    }

    /// The panic message of `check`, or `None` when it did not panic.
    fn panic_message(check: impl FnOnce() + std::panic::UnwindSafe) -> Option<String> {
        let payload = std::panic::catch_unwind(check).err()?;
        Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|text| (*text).to_owned())
                })
                .unwrap_or_default(),
        )
    }

    /// Six rules of the shared contract check, each failing an engine that
    /// breaks that rule alone: a refusal names this build's protocol; a served
    /// answer names it; an unknown method in another protocol is refused for
    /// the protocol on the stdio path; the decoded path refuses another
    /// protocol; a request with no version is served; an empty-payload install
    /// reaches the engine's own decoding. Every toy below is the conforming toy
    /// with one of these wrong, and the check must panic naming it. The check's
    /// other rules have no toy here.
    #[test]
    fn the_contract_check_fails_an_engine_that_breaks_any_one_rule() {
        type Answer = fn(&str) -> EngineResponseEnvelope;
        type Handle = fn(EngineRequestEnvelope) -> EngineResponseEnvelope;
        fn conforming_text(text: &str) -> EngineResponseEnvelope {
            answer_engine_request(text, toy_engine)
        }
        fn unversioned(mut answer: EngineResponseEnvelope) -> EngineResponseEnvelope {
            answer.protocol_version = None;
            answer
        }
        // Rule: a refusal names this build's protocol.
        fn refuses_unversioned(text: &str) -> EngineResponseEnvelope {
            let answer = conforming_text(text);
            if answer.ok {
                answer
            } else {
                unversioned(answer)
            }
        }
        // Rule: a served answer names this build's protocol.
        fn serves_unversioned(text: &str) -> EngineResponseEnvelope {
            let answer = conforming_text(text);
            if answer.ok {
                unversioned(answer)
            } else {
                answer
            }
        }
        // Rule: an unknown method in another protocol is refused for the
        // protocol, not for the method.
        fn method_before_version(text: &str) -> EngineResponseEnvelope {
            if text.contains(r#""method":"frobnicate""#) {
                return EngineResponseEnvelope::failure(UNKNOWN_METHOD_CODE, "toy");
            }
            conforming_text(text)
        }
        // Rule: the decoded path refuses another protocol too.
        fn decoded_ignores_version(request: EngineRequestEnvelope) -> EngineResponseEnvelope {
            let mut request = request;
            request.protocol_version = None;
            toy_engine(request)
        }
        // Rule: a request with no version is served as `0.1.0`.
        fn refuses_unversioned_requests(text: &str) -> EngineResponseEnvelope {
            let value: serde_json::Value = serde_json::from_str(text).unwrap_or_default();
            if value.get("method").is_some() && value.get("protocol_version").is_none() {
                return refusal_for(Some("banana")).expect("refused");
            }
            conforming_text(text)
        }
        // Rule: an install or launch with an empty payload reaches the
        // engine's own decoding.
        fn installs_anything(request: EngineRequestEnvelope) -> EngineResponseEnvelope {
            if let Some(refusal) = incompatible_request_response(&request) {
                return refusal;
            }
            EngineResponseEnvelope::success(json!({}))
        }
        let breaking = engine_contract::breaking_successor(ENGINE_PROTOCOL_VERSION);
        let cases: [(&str, Answer, Handle, String); 6] = [
            (
                "refusal-version",
                refuses_unversioned,
                toy_engine,
                "the answer must name this build's protocol".to_owned(),
            ),
            (
                "served-version",
                serves_unversioned,
                toy_engine,
                "a served answer must name this build's protocol".to_owned(),
            ),
            (
                "method-first",
                method_before_version,
                toy_engine,
                format!(r#"{{"protocol_version":"{breaking}","method":"frobnicate""#),
            ),
            (
                "decoded-version",
                conforming_text,
                decoded_ignores_version,
                format!("decoded Detect for {breaking}: "),
            ),
            (
                "legacy-request",
                refuses_unversioned_requests,
                toy_engine,
                r#"{"method":"capabilities","payload":{}}"#.to_owned(),
            ),
            (
                "empty-install",
                conforming_text,
                installs_anything,
                "decoded Install with an empty payload was served".to_owned(),
            ),
        ];
        for (label, answer_text, handle, expected) in cases {
            let message = panic_message(|| {
                engine_contract::assert_engine_holds_protocol_contract(label, answer_text, handle);
            })
            .unwrap_or_else(|| panic!("{label}: the check passed an engine that breaks it"));
            assert!(message.contains(&expected), "{label}: {message}");
        }
    }
}
