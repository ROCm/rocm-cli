// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for the engine protocol: the types a plugin's stdout is
//! decoded into, and the helpers that turn untrusted filesystem and CLI input
//! into identifiers the CLI then acts on.
//!
//! Decoding of arbitrary bytes, arbitrary text and byte-level mutations of
//! valid encodings is smoke-tested for panics only: every type here uses
//! derived `Deserialize`, so those tests guard against a future hand-written
//! decoder or an unwrap on the decode path, not against wrong values. What a
//! decode must produce is pinned by the round-trip, version-skew and golden
//! document tests.

use proptest::prelude::*;
use rocm_engine_protocol::{
    DetectResponse, DevicePolicy, EngineCapabilities, EngineDeviceAvailability, EngineMethod,
    EngineRecipeEndpointHint, EngineRecipeHint, EngineRecipeUnsupportedCombinationHint,
    EngineRequestEnvelope, EngineResponseEnvelope, GpuSelection, HealthcheckResponse,
    LaunchRequest, LogsRequest, LogsResponse, ResolveModelRequest,
    engine_id_from_plugin_binary_name, gpu_indices_to_csv, is_public_bind_host, launch_gpu_indices,
};
use std::collections::BTreeMap;

/// Decode `bytes` as `T`, asserting only that it does not panic.
macro_rules! decode_without_panic {
    ($type:ty, $bytes:expr) => {
        let _ = serde_json::from_slice::<$type>($bytes);
    };
}

/// Every envelope/payload type a plugin's output is decoded into.
fn decode_all_without_panic(bytes: &[u8]) {
    decode_without_panic!(EngineRequestEnvelope, bytes);
    decode_without_panic!(EngineResponseEnvelope, bytes);
    decode_without_panic!(DetectResponse, bytes);
    decode_without_panic!(HealthcheckResponse, bytes);
    decode_without_panic!(LaunchRequest, bytes);
    decode_without_panic!(LogsRequest, bytes);
    decode_without_panic!(LogsResponse, bytes);
    decode_without_panic!(ResolveModelRequest, bytes);
    decode_without_panic!(EngineRecipeHint, bytes);
    decode_without_panic!(GpuSelection, bytes);
}

/// `any::<f32>()` deliberately excludes NaN and the infinities, so a naive
/// generator never samples the values JSON has no representation for — exactly
/// the region where a float round trip can break. Sample them explicitly.
fn tokens_per_sec_strategy() -> impl Strategy<Value = Option<f32>> {
    proptest::option::of(prop_oneof![
        6 => any::<f32>(),
        1 => Just(f32::NAN),
        1 => Just(f32::INFINITY),
        1 => Just(f32::NEG_INFINITY),
        1 => Just(f32::MIN_POSITIVE),
        1 => Just(-0.0_f32),
    ])
}

fn healthcheck_strategy() -> impl Strategy<Value = HealthcheckResponse> {
    (
        "[ -~]{0,20}",
        any::<bool>(),
        "[ -~]{0,20}",
        any::<u64>(),
        any::<u32>(),
        proptest::option::of("[ -~]{0,30}"),
        tokens_per_sec_strategy(),
    )
        .prop_map(
            |(
                status,
                model_loaded,
                device,
                uptime_sec,
                queue_depth,
                last_error,
                tokens_per_sec,
            )| {
                HealthcheckResponse {
                    status,
                    model_loaded,
                    device,
                    uptime_sec,
                    queue_depth,
                    last_error,
                    tokens_per_sec,
                }
            },
        )
}

fn recipe_hint_strategy() -> impl Strategy<Value = EngineRecipeHint> {
    (
        "[ -~]{0,10}",
        "[ -~]{0,10}",
        prop::collection::vec("[ -~]{0,12}", 0..4),
        prop::collection::btree_map("[ -~]{0,8}", "[ -~]{0,8}", 0..4),
        // Often empty: an empty map is omitted on the wire, so both the
        // omitted and the present spelling have to round-trip.
        prop::collection::btree_map("[ -~]{0,8}", "[ -~]{0,8}", 0..3),
        proptest::option::of((
            "[ -~]{0,8}",
            prop::collection::btree_map("[ -~]{0,8}", "[ -~]{0,8}", 0..3),
        )),
        prop::collection::vec(("[ -~]{0,8}", "[ -~]{0,8}"), 0..3),
        prop::collection::vec("[ -~]{0,8}", 0..3),
    )
        .prop_map(
            |(
                contract_version,
                engine,
                required_flags,
                parser_settings,
                extra_env,
                preferred_endpoint,
                unsupported,
                notes,
            )| EngineRecipeHint {
                contract_version,
                engine,
                required_flags,
                parser_settings,
                extra_env,
                preferred_endpoint: preferred_endpoint.map(|(endpoint_mode, settings)| {
                    EngineRecipeEndpointHint {
                        endpoint_mode,
                        settings,
                    }
                }),
                unsupported_combinations: unsupported
                    .into_iter()
                    .map(
                        |(combination, reason)| EngineRecipeUnsupportedCombinationHint {
                            combination,
                            reason,
                        },
                    )
                    .collect(),
                notes,
            },
        )
}

fn optional_text() -> impl Strategy<Value = Option<String>> {
    proptest::option::of("[ -~]{0,10}")
}

fn detect_response_strategy() -> impl Strategy<Value = DetectResponse> {
    (
        (
            any::<bool>(),
            optional_text(),
            optional_text(),
            optional_text(),
            proptest::option::of(any::<bool>()),
            optional_text(),
            optional_text(),
            optional_text(),
        ),
        prop::collection::vec(
            (
                "[ -~]{0,8}",
                any::<bool>(),
                proptest::option::of("[ -~]{0,8}"),
            ),
            0..4,
        ),
        (
            any::<[bool; 5]>(),
            "[ -~]{0,8}",
            prop::collection::vec("[ -~]{0,8}", 0..4),
        ),
    )
        .prop_map(
            |(
                (
                    installed,
                    env_id,
                    runtime_kind,
                    runtime_executable,
                    managed_env,
                    python_version,
                    torch_version,
                    transformers_version,
                ),
                devices,
                (
                    [
                        cpu,
                        rocm_gpu,
                        openai_compatible,
                        tool_calling,
                        reasoning_parser,
                    ],
                    quantized_models,
                    notes,
                ),
            )| DetectResponse {
                installed,
                env_id,
                runtime_kind,
                runtime_executable,
                managed_env,
                python_version,
                torch_version,
                transformers_version,
                available_devices: devices
                    .into_iter()
                    .map(|(kind, available, reason)| EngineDeviceAvailability {
                        kind,
                        available,
                        reason,
                    })
                    .collect(),
                capabilities: EngineCapabilities {
                    cpu,
                    rocm_gpu,
                    openai_compatible,
                    tool_calling,
                    quantized_models,
                    reasoning_parser,
                },
                notes,
            },
        )
}

/// Byte-level mutations of a valid encoding: the only generator that reliably
/// reaches field-level decode logic.
fn mutate(json: &str, position: usize, kind: u8) -> Vec<u8> {
    let mut bytes = json.as_bytes().to_vec();
    if bytes.is_empty() {
        return bytes;
    }
    let position = position % bytes.len();
    match kind % 6 {
        0 => bytes[position] = b'"',
        1 => bytes[position] = b'{',
        2 => bytes[position] = 0xff,
        3 => {
            bytes.truncate(position);
        }
        4 => {
            bytes.remove(position);
        }
        _ => bytes.insert(position, b'\\'),
    }
    bytes
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    /// Encode then decode must be the identity for every value the type holds.
    #[test]
    fn healthcheck_response_roundtrips(response in healthcheck_strategy()) {
        let encoded = serde_json::to_string(&response).expect("encode");
        let decoded: HealthcheckResponse = serde_json::from_str(&encoded).expect("decode");
        prop_assert_eq!(decoded.status, response.status);
        prop_assert_eq!(decoded.model_loaded, response.model_loaded);
        prop_assert_eq!(decoded.device, response.device);
        prop_assert_eq!(decoded.uptime_sec, response.uptime_sec);
        prop_assert_eq!(decoded.queue_depth, response.queue_depth);
        prop_assert_eq!(decoded.last_error, response.last_error);
        // JSON has no representation for NaN or the infinities, so `serde_json`
        // encodes them as `null` and they come back as `None`. Every *finite*
        // value must round-trip exactly; a non-finite one is documented here as
        // collapsing to "unknown" rather than surviving or erroring.
        match response.tokens_per_sec {
            Some(value) if !value.is_finite() => prop_assert_eq!(
                decoded.tokens_per_sec,
                None,
                "a non-finite rate has no JSON representation and reads back as unknown"
            ),
            expected => prop_assert_eq!(
                decoded.tokens_per_sec.map(f32::to_bits),
                expected.map(f32::to_bits),
                "a finite tokens_per_sec must survive an encode/decode round trip"
            ),
        }
    }

    #[test]
    fn recipe_hint_roundtrips(hint in recipe_hint_strategy()) {
        let encoded = serde_json::to_string(&hint).expect("encode");
        let decoded: EngineRecipeHint = serde_json::from_str(&encoded).expect("decode");
        prop_assert_eq!(&decoded, &hint);
        // Duplicate-serve reuse compares a recipe hint's JSON as a string, so an
        // empty `extra_env` must stay off the wire: a hint encodes exactly as it
        // did before the field existed.
        let value = serde_json::to_value(&hint).expect("encode");
        prop_assert_eq!(
            value.get("extra_env").is_some(),
            !hint.extra_env.is_empty(),
            "extra_env must be omitted exactly when it is empty"
        );
    }

    #[test]
    fn detect_response_roundtrips(response in detect_response_strategy()) {
        let encoded = serde_json::to_string(&response).expect("encode");
        let decoded: DetectResponse = serde_json::from_str(&encoded).expect("decode");
        // `DetectResponse` has no `PartialEq`; compare the whole struct through
        // its JSON value instead of a hand-picked subset of fields. (Being a
        // round trip, this cannot see a renamed field.)
        prop_assert_eq!(
            serde_json::to_value(&decoded).expect("re-encode"),
            serde_json::to_value(&response).expect("encode")
        );
    }

    /// `EngineResponseEnvelope::success` serializes caller data with `expect`;
    /// it must not panic for the response types the engines actually send, and
    /// the data must arrive intact on the other side of the wire.
    #[test]
    fn response_envelope_success_carries_the_data_intact(response in healthcheck_strategy()) {
        let envelope = EngineResponseEnvelope::success(&response);
        prop_assert!(envelope.ok);
        let encoded = serde_json::to_string(&envelope).expect("encode");
        let decoded: EngineResponseEnvelope = serde_json::from_str(&encoded).expect("decode");
        prop_assert!(decoded.ok);
        prop_assert!(decoded.error.is_none());
        // Compared as the response it decodes back into, re-encoded: `serde_json`
        // parses floats to the nearest f64 only approximately, so comparing the
        // raw `Value`s would report a parser rounding as a lost field.
        let data = decoded.data.expect("a success envelope carries data");
        let carried: HealthcheckResponse = serde_json::from_value(data).expect("data decodes");
        prop_assert_eq!(
            serde_json::to_string(&carried).expect("re-encode"),
            serde_json::to_string(&response).expect("encode"),
            "the envelope must carry the caller's data unchanged"
        );
    }

    /// Smoke test: mutating a valid encoding must yield either a clean decode
    /// or a clean error — never a panic. This can only fail on a panic.
    #[test]
    fn mutated_valid_encodings_never_panic(
        response in healthcheck_strategy(),
        hint in recipe_hint_strategy(),
        position in any::<usize>(),
        kind in any::<u8>(),
    ) {
        for json in [
            serde_json::to_string(&response).expect("encode"),
            serde_json::to_string(&hint).expect("encode"),
            serde_json::to_string(&EngineResponseEnvelope::success(&response)).expect("encode"),
        ] {
            decode_all_without_panic(&mutate(&json, position, kind));
        }
    }

    /// Fully arbitrary bytes must never panic a decoder.
    #[test]
    fn arbitrary_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..256)) {
        decode_all_without_panic(&bytes);
    }

    #[test]
    fn arbitrary_text_never_panics(text in "[ -~\\n\\t]{0,256}") {
        decode_all_without_panic(text.as_bytes());
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 1024, ..ProptestConfig::default() })]

    /// A plugin binary name is attacker-influenced (anyone who can drop a file
    /// into a plugin directory). The derived engine id is later used to build
    /// paths and command lines, so it must never carry a separator or a
    /// `.`/`..` traversal component, and the helper must never panic on a
    /// multi-byte name.
    #[test]
    fn engine_id_is_never_a_traversal_component(
        name in prop_oneof![
            "[ -~]{0,24}",
            "rocm-engine-[ -~]{0,16}",
            "rocm-engine-[A-Za-z0-9._/\\\\-]{0,12}(\\.exe|\\.EXE|)",
            "\\PC{0,16}",
            // The traversal names themselves, which no random strategy above
            // reaches: an id of exactly `.` or `..`, with and without `.exe`.
            prop::sample::select(vec![
                "rocm-engine-.",
                "rocm-engine-..",
                "rocm-engine-..exe",
                "rocm-engine-...exe",
                "rocm-engine-.EXE",
            ])
            .prop_map(str::to_owned),
        ],
    ) {
        if let Some(id) = engine_id_from_plugin_binary_name(&name) {
            prop_assert!(!id.is_empty());
            prop_assert!(!id.contains('/'), "engine id must not contain `/`: {id:?}");
            prop_assert!(!id.contains('\\'), "engine id must not contain `\\`: {id:?}");
            prop_assert!(id != "." && id != "..", "engine id must not traverse: {id:?}");
            prop_assert!(
                id.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-')),
                "engine id must stay in the documented character set: {id:?}"
            );
        }
    }

    /// `--gpu` is direct user input. Parsing must never panic, and a successful
    /// parse must round-trip through the CLI spelling it reports.
    #[test]
    fn gpu_selection_parse_never_panics_and_roundtrips(value in "[ -~]{0,16}") {
        match GpuSelection::parse_cli_value(&value) {
            Ok(GpuSelection::Auto) => {
                prop_assert!(launch_gpu_indices(Some(&GpuSelection::Auto)).is_empty());
            }
            Ok(GpuSelection::Index(index)) => {
                prop_assert_eq!(
                    GpuSelection::parse_cli_value(&index.to_string()),
                    Ok(GpuSelection::Index(index))
                );
                prop_assert_eq!(launch_gpu_indices(Some(&GpuSelection::Index(index))), vec![index]);
                prop_assert_eq!(
                    gpu_indices_to_csv(&[index]),
                    Some(index.to_string())
                );
            }
            Err(_) => {}
        }
    }

    /// A GPU ordinal must never reach `HIP_VISIBLE_DEVICES` as anything but a
    /// plain comma-separated list of decimal integers — the value is
    /// interpolated into a child process environment.
    #[test]
    fn gpu_csv_is_always_plain_decimal(indices in prop::collection::vec(any::<u32>(), 0..8)) {
        match gpu_indices_to_csv(&indices) {
            None => prop_assert!(indices.is_empty()),
            Some(csv) => {
                prop_assert!(!indices.is_empty());
                let parsed: Result<Vec<u32>, _> =
                    csv.split(',').map(str::parse::<u32>).collect();
                prop_assert_eq!(parsed.as_deref(), Ok(indices.as_slice()));
            }
        }
    }

    /// The loopback allowlist decides whether an endpoint gets an API key.
    /// Anything that is not one of the three exact accepted spellings must be
    /// classified public (key required) — misclassifying a public bind as
    /// loopback opens an unauthenticated listener.
    ///
    /// A random host string essentially never lands near those spellings, so
    /// most cases are drawn from the near-miss region: each loopback spelling
    /// padded, re-cased, bracketed or extended, plus other addresses that
    /// reach this host but are not on the list.
    #[test]
    fn only_exact_loopback_spellings_skip_the_api_key(
        host in prop_oneof![
            3 => (
                prop::sample::select(LOOPBACK_SPELLINGS.to_vec()),
                0u8..8,
            )
                .prop_map(|(spelling, kind)| near_loopback(spelling, kind)),
            1 => prop::sample::select(OTHER_LOCAL_ADDRESSES.to_vec()).prop_map(str::to_owned),
            1 => "[ -~]{0,24}",
        ],
    ) {
        let loopback = !is_public_bind_host(&host);
        let allowlisted = LOOPBACK_SPELLINGS.contains(&host.as_str());
        prop_assert!(
            loopback == allowlisted,
            "host {host:?} was classified {} but only the exact spellings {LOOPBACK_SPELLINGS:?} \
             may skip the key",
            if loopback { "loopback" } else { "public" }
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// A `LogsRequest` carries a `tail_lines` count straight from the caller into
    /// the engines' tail logic; it must round-trip exactly, including the
    /// extremes an engine could legitimately be asked for.
    #[test]
    fn logs_request_roundtrips(
        service_id in "[ -~]{0,24}",
        tail_lines in proptest::option::of(prop_oneof![
            Just(0usize),
            Just(usize::MAX),
            0usize..10_000,
        ]),
    ) {
        let request = LogsRequest { service_id: service_id.clone(), tail_lines };
        let encoded = serde_json::to_string(&request).expect("encode");
        let decoded: LogsRequest = serde_json::from_str(&encoded).expect("decode");
        prop_assert_eq!(decoded.service_id, service_id);
        prop_assert_eq!(decoded.tail_lines, tail_lines);
    }

    #[test]
    fn logs_response_roundtrips(
        log_path in "[ -~]{0,24}",
        lines in prop::collection::vec("\\PC{0,40}", 0..8),
    ) {
        let response = LogsResponse {
            log_path: log_path.clone(),
            recent_lines: lines.clone(),
        };
        let encoded = serde_json::to_string(&response).expect("encode");
        let decoded: LogsResponse = serde_json::from_str(&encoded).expect("decode");
        prop_assert_eq!(decoded.log_path, log_path);
        prop_assert_eq!(decoded.recent_lines, lines);
    }

    /// `host` and `port` from a launch request are what the CLI later classifies
    /// for API-key enforcement, so they must survive the wire unchanged.
    #[test]
    fn launch_request_roundtrips_host_and_port(
        host in "[ -~]{0,24}",
        port in any::<u16>(),
        selection in proptest::option::of(prop_oneof![
            Just(GpuSelection::Auto),
            any::<u32>().prop_map(GpuSelection::Index),
        ]),
        device_policy in proptest::option::of(prop_oneof![
            Just(DevicePolicy::GpuRequired),
            Just(DevicePolicy::GpuPreferred),
            Just(DevicePolicy::CpuOnly),
        ]),
    ) {
        let request = LaunchRequest {
            service_id: "svc".to_owned(),
            env_id: None,
            runtime_id: None,
            model_ref: "model".to_owned(),
            host: host.clone(),
            port,
            device_policy: device_policy.clone(),
            endpoint_mode: None,
            engine_recipe: None,
            gpu_selection: selection.clone(),
        };
        let encoded = serde_json::to_string(&request).expect("encode");
        let decoded: LaunchRequest = serde_json::from_str(&encoded).expect("decode");
        prop_assert_eq!(&decoded.host, &host);
        prop_assert_eq!(decoded.port, port);
        prop_assert_eq!(decoded.gpu_selection, selection);
        prop_assert_eq!(decoded.device_policy, device_policy);
        prop_assert_eq!(
            is_public_bind_host(&decoded.host),
            is_public_bind_host(&host),
            "the bind classification must not change across the wire"
        );
    }

    /// A request envelope decodes exactly when its method is one of the
    /// documented wire names — an unknown method is rejected rather than
    /// falling back to some default — and a decoded method re-encodes to the
    /// spelling it arrived as. Half the cases use a real method name, the rest
    /// near misses of one or arbitrary words.
    #[test]
    fn request_envelope_accepts_exactly_the_known_methods(
        method in prop_oneof![
            2 => prop::sample::select(METHOD_WIRE_NAMES.to_vec()).prop_map(str::to_owned),
            1 => (prop::sample::select(METHOD_WIRE_NAMES.to_vec()), 0u8..4).prop_map(
                |(name, kind)| match kind {
                    0 => name.to_ascii_uppercase(),
                    1 => format!("{name}s"),
                    2 => name.replace('_', ""),
                    _ => format!(" {name}"),
                }
            ),
            1 => "[a-z_]{0,16}",
        ],
        payload in "[ -~]{0,24}",
    ) {
        let json = serde_json::json!({ "method": method, "payload": payload }).to_string();
        let parsed = serde_json::from_str::<EngineRequestEnvelope>(&json);
        let known = METHOD_WIRE_NAMES.contains(&method.as_str());
        prop_assert_eq!(
            parsed.is_ok(),
            known,
            "method {:?} decoded: {}",
            method,
            parsed.is_ok()
        );
        if let Ok(envelope) = parsed {
            let reencoded = serde_json::to_value(&envelope).expect("encode");
            prop_assert_eq!(
                reencoded.get("method").and_then(serde_json::Value::as_str),
                Some(method.as_str()),
                "a decoded method must re-encode to the same wire spelling"
            );
        }
    }
}

/// The documented wire name of every engine method, written out rather than
/// derived from the enum.
const METHOD_WIRE_NAMES: [&str; 9] = [
    "detect",
    "install",
    "capabilities",
    "resolve_model",
    "launch",
    "healthcheck",
    "endpoint",
    "stop",
    "logs",
];

/// The three spellings the CLI accepts as loopback, written out here rather than
/// shared with the code under test.
const LOOPBACK_SPELLINGS: [&str; 3] = ["127.0.0.1", "localhost", "::1"];

/// Addresses that reach this host, or every interface, without being one of the
/// accepted spellings: all must be treated as public.
const OTHER_LOCAL_ADDRESSES: [&str; 9] = [
    "0.0.0.0",
    "::",
    "[::1]",
    "::ffff:127.0.0.1",
    "127.0.0.2",
    "127.000.0.1",
    "0:0:0:0:0:0:0:1",
    "localhost.localdomain",
    "",
];

/// One near miss of a loopback spelling; kind 0 is the spelling itself.
fn near_loopback(spelling: &str, kind: u8) -> String {
    match kind {
        0 => spelling.to_owned(),
        1 => format!(" {spelling}"),
        2 => format!("{spelling}\n"),
        3 => spelling.to_ascii_uppercase(),
        4 => {
            let mut chars = spelling.chars();
            chars
                .next()
                .map(|first| first.to_ascii_uppercase().to_string() + chars.as_str())
                .unwrap_or_default()
        }
        5 => format!("[{spelling}]"),
        6 => format!("{spelling}:8080"),
        _ => format!("{spelling}."),
    }
}

/// Present four times in five, so that a meaningful share of documents carry
/// every required field and take the decode-succeeds path.
fn mostly_present<T: std::fmt::Debug + Clone + 'static>(
    value: impl Strategy<Value = T> + 'static,
) -> impl Strategy<Value = Option<T>> {
    proptest::option::weighted(0.8, value)
}

/// A map strategy over the fields of a valid response, so that fields can be
/// dropped individually — version skew as an engine would produce it, not as
/// random bytes would.
fn field_skew_strategy() -> impl Strategy<Value = serde_json::Value> {
    (
        mostly_present("[ -~]{0,8}"),
        mostly_present(any::<bool>()),
        mostly_present("[ -~]{0,8}"),
        mostly_present(any::<u64>()),
        mostly_present(any::<u32>()),
        "[ -~]{0,8}",
    )
        .prop_map(
            |(status, model_loaded, device, uptime, queue_depth, extra)| {
                let mut object = serde_json::Map::new();
                if let Some(status) = status {
                    object.insert("status".to_owned(), serde_json::Value::String(status));
                }
                if let Some(model_loaded) = model_loaded {
                    object.insert(
                        "model_loaded".to_owned(),
                        serde_json::Value::Bool(model_loaded),
                    );
                }
                if let Some(device) = device {
                    object.insert("device".to_owned(), serde_json::Value::String(device));
                }
                if let Some(uptime) = uptime {
                    object.insert("uptime_sec".to_owned(), serde_json::Value::from(uptime));
                }
                if let Some(queue_depth) = queue_depth {
                    object.insert(
                        "queue_depth".to_owned(),
                        serde_json::Value::from(queue_depth),
                    );
                }
                object.insert(
                    "unknown_future_field".to_owned(),
                    serde_json::Value::String(extra),
                );
                serde_json::Value::Object(object)
            },
        )
}

/// The fields `HealthcheckResponse` cannot decode without.
const HEALTHCHECK_REQUIRED_FIELDS: [&str; 5] = [
    "status",
    "model_loaded",
    "device",
    "uptime_sec",
    "queue_depth",
];

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// Version skew: a newer engine sends extra fields and may omit ones this
    /// build requires. The document decodes exactly when every required field
    /// is present — the unknown field it always carries must be ignored — and
    /// a decoded response carries the values that were sent.
    #[test]
    fn version_skew_decodes_exactly_when_required_fields_are_present(value in field_skew_strategy()) {
        let bytes = serde_json::to_vec(&value).expect("encode");
        let all_required_present = HEALTHCHECK_REQUIRED_FIELDS
            .iter()
            .all(|field| value.get(field).is_some());
        let decoded = serde_json::from_slice::<HealthcheckResponse>(&bytes);
        prop_assert_eq!(
            decoded.is_ok(),
            all_required_present,
            "decode result {:?} for {}",
            decoded.as_ref().map(|_| ()).map_err(ToString::to_string),
            value
        );
        if let Ok(response) = decoded {
            prop_assert_eq!(Some(response.status.as_str()), value["status"].as_str());
            prop_assert_eq!(Some(response.model_loaded), value["model_loaded"].as_bool());
            prop_assert_eq!(Some(response.device.as_str()), value["device"].as_str());
            prop_assert_eq!(Some(response.uptime_sec), value["uptime_sec"].as_u64());
            prop_assert_eq!(
                Some(u64::from(response.queue_depth)),
                value["queue_depth"].as_u64()
            );
            prop_assert_eq!(response.last_error, None);
        }
        decode_all_without_panic(&bytes);
    }
}

/// Golden documents for the wire spellings the round trips cannot see: device
/// policy and GPU selection values, and a literal healthcheck response as an
/// engine binary built from another revision would send it.
#[test]
fn protocol_values_use_their_documented_wire_spellings() {
    for (policy, wire) in [
        (DevicePolicy::GpuRequired, "gpu_required"),
        (DevicePolicy::GpuPreferred, "gpu_preferred"),
        (DevicePolicy::CpuOnly, "cpu_only"),
    ] {
        assert_eq!(
            serde_json::to_value(&policy).expect("encode"),
            serde_json::json!(wire)
        );
    }
    assert_eq!(
        serde_json::to_value(GpuSelection::Auto).expect("encode"),
        serde_json::json!("auto")
    );
    assert_eq!(
        serde_json::to_value(GpuSelection::Index(1)).expect("encode"),
        serde_json::json!({ "index": 1 })
    );

    let response: HealthcheckResponse = serde_json::from_value(serde_json::json!({
        "status": "ready",
        "model_loaded": true,
        "device": "gpu",
        "uptime_sec": 12,
        "queue_depth": 3,
        "last_error": "boom",
        "tokens_per_sec": 1.5,
    }))
    .expect("a healthcheck document decodes");
    assert_eq!(response.status, "ready");
    assert!(response.model_loaded);
    assert_eq!(response.device, "gpu");
    assert_eq!(response.uptime_sec, 12);
    assert_eq!(response.queue_depth, 3);
    assert_eq!(response.last_error.as_deref(), Some("boom"));
    assert_eq!(response.tokens_per_sec, Some(1.5));
}

/// Non-finite token rates have no JSON representation: they encode as `null`
/// and read back as unknown, rather than failing the whole healthcheck. Pinned
/// deterministically, since the round-trip property only samples them.
#[test]
fn non_finite_token_rates_read_back_as_unknown() {
    for rate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let response: HealthcheckResponse = serde_json::from_value(serde_json::json!({
            "status": "ready",
            "model_loaded": true,
            "device": "gpu",
            "uptime_sec": 1,
            "queue_depth": 0,
        }))
        .expect("decode");
        let response = HealthcheckResponse {
            tokens_per_sec: Some(rate),
            ..response
        };
        let encoded = serde_json::to_string(&response).expect("a non-finite rate still encodes");
        let decoded: HealthcheckResponse = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(
            decoded.tokens_per_sec, None,
            "{rate} read back as {:?}",
            decoded.tokens_per_sec
        );
    }
}

/// `--gpu` values as a user types them, with the selection each must produce.
#[test]
fn gpu_cli_values_parse_as_documented() {
    for (value, expected) in [
        ("auto", Ok(GpuSelection::Auto)),
        ("AUTO", Ok(GpuSelection::Auto)),
        ("", Ok(GpuSelection::Auto)),
        ("  ", Ok(GpuSelection::Auto)),
        ("0", Ok(GpuSelection::Index(0))),
        (" 1 ", Ok(GpuSelection::Index(1))),
        ("4294967295", Ok(GpuSelection::Index(u32::MAX))),
    ] {
        assert_eq!(GpuSelection::parse_cli_value(value), expected, "{value:?}");
    }
    let multi = GpuSelection::parse_cli_value("1,2").expect_err("multiple GPUs are rejected");
    assert!(
        multi.contains("multiple GPUs is not supported"),
        "a list of GPUs must say why it is rejected, got {multi:?}"
    );
    for value in ["-1", "4294967296", "gpu0", "1.0", "auto1"] {
        assert!(
            GpuSelection::parse_cli_value(value).is_err(),
            "{value:?} must be rejected"
        );
    }
}

/// Literal request documents as the CLI sends them, and a detect response
/// without the optional runtime fields: each must decode to the values it
/// spells. This catches a renamed or removed field; it cannot catch a lost
/// `#[serde(default)]` on an `Option`, which serde treats as optional anyway.
#[test]
fn request_and_detect_documents_have_their_documented_wire_shape() {
    let launch: LaunchRequest = serde_json::from_value(serde_json::json!({
        "service_id": "svc",
        "env_id": null,
        "runtime_id": null,
        "model_ref": "org/model",
        "host": "127.0.0.1",
        "port": 8080,
        "device_policy": "gpu_required",
        "endpoint_mode": null,
        "gpu_selection": { "index": 1 },
    }))
    .expect("a launch request decodes");
    assert_eq!(launch.service_id, "svc");
    assert_eq!(launch.model_ref, "org/model");
    assert_eq!(launch.host, "127.0.0.1");
    assert_eq!(launch.port, 8080);
    assert_eq!(launch.device_policy, Some(DevicePolicy::GpuRequired));
    assert_eq!(launch.gpu_selection, Some(GpuSelection::Index(1)));
    assert!(launch.engine_recipe.is_none());

    let logs: LogsRequest =
        serde_json::from_value(serde_json::json!({ "service_id": "svc", "tail_lines": 50 }))
            .expect("a logs request decodes");
    assert_eq!(logs.service_id, "svc");
    assert_eq!(logs.tail_lines, Some(50));

    let detect: DetectResponse = serde_json::from_value(serde_json::json!({
        "installed": true,
        "env_id": "env",
        "python_version": null,
        "torch_version": null,
        "transformers_version": null,
        "available_devices": [{ "kind": "gpu", "available": true }],
        "capabilities": {
            "cpu": false,
            "rocm_gpu": true,
            "openai_compatible": true,
            "tool_calling": false,
            "quantized_models": "none",
            "reasoning_parser": false,
        },
        "notes": ["n"],
    }))
    .expect("a detect response without the optional runtime fields decodes");
    assert!(detect.installed);
    assert_eq!(detect.env_id.as_deref(), Some("env"));
    assert!(detect.runtime_kind.is_none() && detect.managed_env.is_none());
    assert_eq!(detect.available_devices.len(), 1);
    assert!(detect.available_devices[0].available);
    assert!(detect.capabilities.rocm_gpu && !detect.capabilities.cpu);
    assert_eq!(detect.notes, vec!["n".to_owned()]);
}

/// The error envelope an engine sends, as a literal document: a swapped code
/// and message, or a failure that reads as success, must fail here.
#[test]
fn failure_envelope_has_its_documented_wire_shape() {
    assert_eq!(
        serde_json::to_value(EngineResponseEnvelope::failure("c", "m")).expect("encode"),
        serde_json::json!({ "ok": false, "error": { "code": "c", "message": "m" } })
    );
}

/// The plugin binary naming rule, positively: the `rocm-engine-` prefix with
/// an optional `.exe` in either case yields the id, and anything else none.
#[test]
fn plugin_binary_names_yield_their_engine_id() {
    for name in [
        "rocm-engine-foo",
        "rocm-engine-foo.exe",
        "rocm-engine-foo.EXE",
    ] {
        assert_eq!(
            engine_id_from_plugin_binary_name(name).as_deref(),
            Some("foo"),
            "{name}"
        );
    }
    for name in ["rocm-engin-foo", "engine-foo", "foo", "rocm-engine-"] {
        assert_eq!(engine_id_from_plugin_binary_name(name), None, "{name}");
    }
}

/// The documented wire name of each method. The `match` is exhaustive, so a
/// new method fails to compile until it is listed here; the variant list in
/// the test below and [`METHOD_WIRE_NAMES`] are maintained by hand and are not
/// compile-enforced.
const fn method_wire_name(method: &EngineMethod) -> &'static str {
    match method {
        EngineMethod::Detect => "detect",
        EngineMethod::Install => "install",
        EngineMethod::Capabilities => "capabilities",
        EngineMethod::ResolveModel => "resolve_model",
        EngineMethod::Launch => "launch",
        EngineMethod::Healthcheck => "healthcheck",
        EngineMethod::Endpoint => "endpoint",
        EngineMethod::Stop => "stop",
        EngineMethod::Logs => "logs",
    }
}

/// Every method's wire spelling, written out: the plugin protocol is a
/// contract with separately built engine binaries, so a renamed method must
/// fail here rather than at runtime. (The round-trip properties cannot see a
/// rename, since encoding and decoding change together; the golden documents
/// above pin a few more spellings, not the whole wire format.)
#[test]
fn engine_methods_use_their_documented_wire_names() {
    for wire in METHOD_WIRE_NAMES {
        let envelope: EngineRequestEnvelope =
            serde_json::from_value(serde_json::json!({ "method": wire }))
                .unwrap_or_else(|error| panic!("{wire:?} must decode: {error}"));
        assert_eq!(method_wire_name(&envelope.method), wire);
    }
    for method in [
        EngineMethod::Detect,
        EngineMethod::Install,
        EngineMethod::Capabilities,
        EngineMethod::ResolveModel,
        EngineMethod::Launch,
        EngineMethod::Healthcheck,
        EngineMethod::Endpoint,
        EngineMethod::Stop,
        EngineMethod::Logs,
    ] {
        let wire = method_wire_name(&method);
        assert!(
            METHOD_WIRE_NAMES.contains(&wire),
            "{wire:?} is missing from METHOD_WIRE_NAMES"
        );
        assert_eq!(
            serde_json::to_value(&method).expect("encode"),
            serde_json::Value::from(wire)
        );
        let envelope: EngineRequestEnvelope =
            serde_json::from_value(serde_json::json!({ "method": wire }))
                .unwrap_or_else(|error| panic!("{wire:?} must decode: {error}"));
        assert_eq!(envelope.method, method);
    }
}

// `BTreeMap` keys in a recipe hint come off the wire; duplicates in the JSON
// must resolve deterministically rather than depending on iteration order.
proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[test]
    fn duplicate_parser_setting_keys_resolve_last_wins(
        key in "[a-z]{1,6}",
        first in "[ -~]{0,8}",
        second in "[ -~]{0,8}",
    ) {
        let json = format!(
            r#"{{"contract_version":"0.1.0","engine":"vllm","parser_settings":{{{key}:{first},{key}:{second}}}}}"#,
            key = serde_json::to_string(&key).expect("key"),
            first = serde_json::to_string(&first).expect("first"),
            second = serde_json::to_string(&second).expect("second"),
        );
        let decoded: EngineRecipeHint = serde_json::from_str(&json).expect("decode");
        let expected: BTreeMap<String, String> = BTreeMap::from([(key, second)]);
        prop_assert_eq!(decoded.parser_settings, expected);
    }
}
