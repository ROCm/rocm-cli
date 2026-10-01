// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property tests for model selection and for the readiness decision the CLI
//! makes from an engine's `/v1/models` response.
//!
//! Two separate risks live here. Selection must be *deterministic and
//! unambiguous*: `resolve_model_recipe` takes the first recipe whose
//! `matches_ref` returns true, so two recipes answering to the same selector
//! would make the winner an accident of catalog order. And readiness is decided
//! from a JSON document served by the engine — untrusted input — so it must
//! never panic and must not report a *different* model as the one the user
//! asked for.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use proptest::prelude::*;
use rocm_core::{ModelRecipeRecord, builtin_model_recipes, openai_models_endpoint_has_model};

fn catalog() -> &'static [ModelRecipeRecord] {
    use std::sync::OnceLock;
    static CATALOG: OnceLock<Vec<ModelRecipeRecord>> = OnceLock::new();
    CATALOG.get_or_init(builtin_model_recipes)
}

/// Every selector the shipped catalog advertises: canonical ids and aliases.
fn advertised_selectors() -> Vec<(String, String)> {
    let mut selectors = Vec::new();
    for recipe in catalog() {
        selectors.push((
            recipe.canonical_model_id.clone(),
            recipe.canonical_model_id.clone(),
        ));
        for alias in &recipe.aliases {
            selectors.push((alias.clone(), recipe.canonical_model_id.clone()));
        }
    }
    selectors
}

/// Serve one HTTP response carrying `body` on an ephemeral loopback port.
fn serve_once(body: String) -> (u16, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let port = listener.local_addr().expect("local addr").port();
    let handle = thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut scratch = [0_u8; 4096];
            let _ = stream.read(&mut scratch);
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(body.as_bytes());
            let _ = stream.flush();
        }
    });
    (port, handle)
}

/// Ask the readiness check what it makes of `body` when expecting `expected`.
fn endpoint_reports(body: String, expected: Option<&str>) -> Result<bool, String> {
    let (port, server) = serve_once(body);
    let url = format!("http://127.0.0.1:{port}");
    let outcome = openai_models_endpoint_has_model(&url, expected, None, Duration::from_secs(10))
        .map_err(|error| error.to_string());
    let _ = server.join();
    outcome
}

#[test]
fn the_shipped_catalog_is_not_empty() {
    assert!(
        !catalog().is_empty(),
        "these properties are vacuous without a catalog to check"
    );
}

/// No selector the catalog advertises may resolve to two different recipes.
/// `resolve_model_recipe` returns the first match, so an overlap means the
/// winner depends on JSON ordering rather than on the user's intent.
#[test]
fn no_advertised_selector_matches_two_recipes() {
    let mut ambiguous: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for (selector, _) in advertised_selectors() {
        let matches: Vec<&str> = catalog()
            .iter()
            .filter(|recipe| recipe.matches_ref(&selector))
            .map(|recipe| recipe.canonical_model_id.as_str())
            .collect();
        if matches.len() > 1 {
            ambiguous.insert(selector, matches);
        }
    }
    assert!(
        ambiguous.is_empty(),
        "these selectors resolve to more than one recipe, so the winner is catalog order: \
         {ambiguous:#?}"
    );
}

/// Every advertised selector must resolve to the recipe that advertises it.
#[test]
fn every_advertised_selector_resolves_to_its_own_recipe() {
    for (selector, owner) in advertised_selectors() {
        let resolved: Vec<&str> = catalog()
            .iter()
            .filter(|recipe| recipe.matches_ref(&selector))
            .map(|recipe| recipe.canonical_model_id.as_str())
            .collect();
        assert_eq!(
            resolved,
            vec![owner.as_str()],
            "selector {selector:?} must resolve to exactly its own recipe"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    /// Selector matching ignores surrounding whitespace and ASCII case: any
    /// re-casing of an advertised selector, padded with spaces, tabs or
    /// newlines, still selects exactly its own recipe. (That other edits do
    /// not match is the next property's job.)
    #[test]
    fn selector_matching_tolerates_only_case_and_surrounding_whitespace(
        index in any::<prop::sample::Index>(),
        upper in prop::collection::vec(any::<bool>(), 0..80),
        lead in "[ \\t\\n]{0,3}",
        trail in "[ \\t\\n]{0,3}",
    ) {
        let selectors = advertised_selectors();
        let (selector, owner) = &selectors[index.index(selectors.len())];
        let recased: String = selector
            .chars()
            .enumerate()
            .map(|(position, ch)| {
                if upper.get(position).copied().unwrap_or(false) {
                    ch.to_ascii_uppercase()
                } else {
                    ch
                }
            })
            .collect();
        let decorated = format!("{lead}{recased}{trail}");
        let resolved: Vec<&str> = catalog()
            .iter()
            .filter(|recipe| recipe.matches_ref(&decorated))
            .map(|recipe| recipe.canonical_model_id.as_str())
            .collect();
        prop_assert_eq!(
            resolved,
            vec![owner.as_str()],
            "case/whitespace decoration of {:?} changed which recipe it selects",
            selector
        );
    }

    /// Selection is otherwise exact: an advertised selector with one character
    /// inserted (interior whitespace included) or deleted selects no recipe at
    /// all — neither its own (no fuzzy matching) nor another — unless the edit
    /// happens to spell some other advertised selector, up to ASCII case.
    #[test]
    fn a_mistyped_selector_selects_nothing(
        index in any::<prop::sample::Index>(),
        position in any::<prop::sample::Index>(),
        delete in any::<bool>(),
        inserted in prop::sample::select(
            "abcdefghijklmnopqrstuvwxyz0123456789-_./ \t".chars().collect::<Vec<_>>()
        ),
    ) {
        let selectors = advertised_selectors();
        let (selector, _) = &selectors[index.index(selectors.len())];
        let mut mistyped: Vec<char> = selector.chars().collect();
        if delete {
            let at = position.index(mistyped.len());
            mistyped.remove(at);
        } else {
            let at = position.index(mistyped.len() + 1);
            // Whitespace at either end is surrounding whitespace, which the
            // previous property covers and selection deliberately ignores.
            prop_assume!(
                !(inserted.is_whitespace() && (at == 0 || at == mistyped.len()))
            );
            mistyped.insert(at, inserted);
        }
        let mistyped: String = mistyped.into_iter().collect();
        prop_assume!(
            !selectors
                .iter()
                .any(|(other, _)| other.eq_ignore_ascii_case(&mistyped))
        );

        let selected: Vec<&str> = catalog()
            .iter()
            .filter(|recipe| recipe.matches_ref(&mistyped))
            .map(|recipe| recipe.canonical_model_id.as_str())
            .collect();
        prop_assert!(
            selected.is_empty(),
            "mistyped selector {:?} (from {:?}) selected {:?}",
            mistyped,
            selector,
            selected
        );
    }
}

/// `/v1/models` bodies shaped the way a real engine shapes them, plus the
/// degenerate shapes a broken or hostile one could send.
fn models_body_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => prop::collection::vec("[ -~]{0,24}", 0..4).prop_map(|ids| {
            serde_json::json!({
                "object": "list",
                "data": ids
                    .into_iter()
                    .map(|id| serde_json::json!({ "id": id, "object": "model" }))
                    .collect::<Vec<_>>(),
            })
            .to_string()
        }),
        1 => Just("{}".to_owned()),
        1 => Just(r#"{"data": []}"#.to_owned()),
        1 => Just(r#"{"data": null}"#.to_owned()),
        1 => Just(r#"{"data": [{}]}"#.to_owned()),
        1 => Just(r#"{"data": [{"id": null}]}"#.to_owned()),
        1 => Just(r#"{"data": [[]]}"#.to_owned()),
        1 => Just(r#"{"data": "not-an-array"}"#.to_owned()),
        1 => Just("[]".to_owned()),
        1 => Just("null".to_owned()),
        1 => "[ -~]{0,64}",
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    /// A malformed or hostile `/v1/models` document must produce a clean
    /// `false` or a clean error — never a panic, and never a spurious "ready".
    #[test]
    fn a_models_document_never_reports_ready_without_naming_a_model(
        body in models_body_strategy(),
        expected in prop::sample::select(vec![
            "Qwen/Qwen3-4B-Instruct-2507",
            "meta-llama/Llama-3.1-8B-Instruct",
            "nonexistent/model",
        ]),
    ) {
        // The call must return, not unwind. Any `Ok(true)` has to be justified
        // by the document actually naming a model that matches.
        if endpoint_reports(body.clone(), Some(expected)) == Ok(true) {
            let value: serde_json::Value = serde_json::from_str(body.trim())
                .expect("a ready verdict requires a parseable document");
            let names_a_model = value
                .get("data")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .any(|item| {
                    ["id", "model", "name"]
                        .into_iter()
                        .filter_map(|field| item.get(field).and_then(serde_json::Value::as_str))
                        .any(|value| !value.trim().is_empty())
                });
            prop_assert!(
                names_a_model,
                "reported ready for {expected:?} from a document naming no model at all: {body}"
            );
        }
    }

    /// Positive control for the property above: a document that does list the
    /// expected model — exactly, re-cased, padded, or by its basename alone, as
    /// engines commonly report it — among other entries reads as ready. Without
    /// this, a check that never reported ready would pass every negative
    /// property here.
    /// It pins the outcome, not which matching rule produces it: several rules
    /// accept these spellings, so it does not stand guard over any one of them.
    #[test]
    fn a_models_document_listing_the_expected_model_reports_ready(
        others in prop::collection::vec("[ -~]{0,24}", 0..3),
        at in any::<prop::sample::Index>(),
        spelling in 0u8..4,
        expected in prop::sample::select(vec![
            "Qwen/Qwen3-4B-Instruct-2507",
            "meta-llama/Llama-3.1-8B-Instruct",
            "nonexistent/model",
        ]),
    ) {
        let listed = match spelling {
            0 => expected.to_owned(),
            1 => expected.to_ascii_uppercase(),
            2 => format!(" {expected}\t"),
            _ => expected.rsplit('/').next().unwrap_or(expected).to_owned(),
        };
        let mut ids = others;
        let position = at.index(ids.len() + 1);
        ids.insert(position, listed.clone());
        let body = serde_json::json!({
            "object": "list",
            "data": ids
                .iter()
                .map(|id| serde_json::json!({ "id": id, "object": "model" }))
                .collect::<Vec<_>>(),
        })
        .to_string();
        prop_assert_eq!(
            endpoint_reports(body, Some(expected)),
            Ok(true),
            "{:?} listed as {:?} was not reported ready",
            expected,
            listed
        );
    }
}

/// Whether `name` is a bare *family* word — a single alphabetic token of three
/// or more characters with no digit, like `qwen`, `llama` or `gemma`.
///
/// Approximates the shape `rocm_core`'s private family matcher treats as a
/// family: readiness matches such a word against any model with a name token
/// that starts with it, so these names are excluded from the strict property
/// below and pinned separately. (Being an approximation, it only decides which
/// catalog names that property skips; the near-miss names that property also
/// checks never pass through it.)
fn is_bare_family_word(name: &str) -> bool {
    let normalized: String = name
        .trim()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect();
    let single_token = name
        .trim()
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .count()
        == 1;
    single_token && normalized.len() >= 3 && !normalized.chars().any(|ch| ch.is_ascii_digit())
}

/// An engine that lists only a *different* catalog model must not satisfy a
/// request for the one the user asked for. This is the "silently serving an
/// unrelated model" case: readiness is what gates the traffic the user then
/// sends, so a false positive routes inference to the wrong weights.
///
/// Checked exhaustively over every ordered pair of catalog entries — including
/// each entry's aliases as the served id, since an engine echoes back whatever
/// name it was launched with — rather than sampled. The catalog is small, the
/// confusable pairs in it are specific (several `Qwen*` variants differing only
/// in size and quantization), and a sampling strategy can miss exactly those.
///
/// Each wanted model is also checked against near misses of its own name that
/// it does not answer to: the name with a quantization or variant suffix, the
/// first name token containing a digit (`qwen3`, or just `3` for a Llama), a
/// two-letter stub, and the leading word with its first letter dropped
/// (`wen`). Those are what separate an exact comparison from a substring one,
/// and a family match from one that has lost its guards or matches inside a
/// word rather than at its start.
///
/// Bare family words are excluded here and pinned by
/// [`a_bare_family_word_satisfies_readiness_for_the_whole_family`], which
/// documents the one shape that *does* cross-match today.
#[test]
fn listing_a_different_catalog_model_is_never_readiness() {
    let recipes = catalog();
    assert!(recipes.len() > 1, "need at least two recipes to confuse");
    let mut false_positives = Vec::new();

    for wanted in recipes {
        for served in recipes {
            if served.canonical_model_id == wanted.canonical_model_id {
                continue;
            }
            let served_names = std::iter::once(&served.canonical_model_id).chain(&served.aliases);
            for served_name in served_names {
                // Skip a name the *wanted* recipe also answers to: that is a
                // genuine shared alias, not a confusion.
                if wanted.matches_ref(served_name) || is_bare_family_word(served_name) {
                    continue;
                }
                let body = serde_json::json!({
                    "object": "list",
                    "data": [{ "id": served_name, "object": "model" }],
                })
                .to_string();
                if endpoint_reports(body, Some(&wanted.canonical_model_id)) == Ok(true) {
                    false_positives.push(format!(
                        "wanted {:?} but an endpoint serving only {:?} reported ready",
                        wanted.canonical_model_id, served_name
                    ));
                }
            }
        }

        for served_name in near_miss_names(&wanted.canonical_model_id) {
            if wanted.matches_ref(&served_name) {
                continue;
            }
            let body = serde_json::json!({
                "object": "list",
                "data": [{ "id": served_name, "object": "model" }],
            })
            .to_string();
            if endpoint_reports(body, Some(&wanted.canonical_model_id)) == Ok(true) {
                false_positives.push(format!(
                    "wanted {:?} but an endpoint serving only the near miss {:?} reported ready",
                    wanted.canonical_model_id, served_name
                ));
            }
        }
    }

    assert!(
        false_positives.is_empty(),
        "readiness matched an unrelated model:\n{}",
        false_positives.join("\n")
    );
}

/// Readiness accepts a catalog model under any of its names: an engine that
/// reports an alias satisfies a request for the canonical id, and the reverse.
/// Checked for every alias in the shipped catalog.
#[test]
fn readiness_accepts_a_catalog_model_under_any_of_its_names() {
    let mut misses = Vec::new();
    for recipe in catalog() {
        for alias in &recipe.aliases {
            for (served, wanted) in [
                (alias.as_str(), recipe.canonical_model_id.as_str()),
                (recipe.canonical_model_id.as_str(), alias.as_str()),
            ] {
                let body = serde_json::json!({
                    "object": "list",
                    "data": [{ "id": served, "object": "model" }],
                })
                .to_string();
                if endpoint_reports(body, Some(wanted)) != Ok(true) {
                    misses.push(format!("serving {served:?} did not satisfy {wanted:?}"));
                }
            }
        }
    }
    assert!(misses.is_empty(), "{}", misses.join("\n"));
}

/// Names that sit next to `id` without being it: `id` with a variant suffix,
/// the first token of its basename that contains a digit, a two-letter stub of
/// its basename, and the basename's leading letters minus the first one.
fn near_miss_names(id: &str) -> Vec<String> {
    let basename = id.rsplit('/').next().unwrap_or(id);
    let mut names: Vec<String> = ["-FP8", "-GGUF", "-Instruct", "-v2"]
        .iter()
        .map(|suffix| format!("{id}{suffix}"))
        .collect();
    if let Some(first_token) = basename
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .find(|token| token.chars().any(|ch| ch.is_ascii_digit()))
    {
        names.push(first_token.to_ascii_lowercase());
    }
    let stub: String = basename
        .chars()
        .filter(char::is_ascii_alphabetic)
        .take(2)
        .collect();
    if stub.len() == 2 {
        names.push(stub.to_ascii_lowercase());
    }
    let leading_word: String = basename
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect();
    if leading_word.len() >= 4 {
        names.push(leading_word[1..].to_ascii_lowercase());
    }
    names
}

/// Characterization test for a known false-positive surface, pinned so it is
/// visible rather than merely absent from the suite.
///
/// Readiness matches a bare *family* word against any model with a name token
/// that starts with it, and does so in both directions. One direction is
/// exercised by an existing unit test in `net.rs`: asking for `qwen` is
/// satisfied by an engine serving `Qwen3-0.6B-GGUF`. This test pins the other: an endpoint
/// whose `/v1/models` reports only the bare id `qwen` — itself a catalog alias
/// — is reported ready for other `Qwen*` entries in the catalog, not just the
/// one it actually loaded. The test asserts that such cross-matches exist and
/// that they stay inside the family.
///
/// So when an engine reports nothing more specific than the family word,
/// readiness cannot tell the models of that family apart: a request for one
/// `Qwen*` model reads as ready whichever `Qwen*` model the engine has
/// resident. Note the asymmetry — *selection* (`matches_ref`) is exact up to
/// case and whitespace, while the readiness check that confirms the engine is
/// serving the selected model is looser.
///
/// If readiness is ever tightened, this test is expected to fail and should be
/// updated along with the behavior.
#[test]
fn a_bare_family_word_satisfies_readiness_for_the_whole_family() {
    let body = serde_json::json!({
        "object": "list",
        "data": [{ "id": "qwen", "object": "model" }],
    })
    .to_string();

    let cross_matched: Vec<&str> = catalog()
        .iter()
        .map(|recipe| recipe.canonical_model_id.as_str())
        // Every catalog entry except the one that genuinely owns the `qwen`
        // alias.
        .filter(|id| {
            !catalog()
                .iter()
                .find(|recipe| recipe.canonical_model_id == *id)
                .is_some_and(|recipe| recipe.matches_ref("qwen"))
        })
        .filter(|id| endpoint_reports(body.clone(), Some(id)) == Ok(true))
        .collect();

    assert!(
        !cross_matched.is_empty(),
        "expected the documented family-word cross-match to still be present"
    );
    assert!(
        cross_matched
            .iter()
            .all(|id| id.to_ascii_lowercase().contains("qwen")),
        "the cross-match must at least stay inside the family: {cross_matched:?}"
    );

    // The other direction, which the `net.rs` unit test also exercises: asking
    // for the bare family word is satisfied by an engine serving any catalog
    // model of that family.
    for recipe in catalog().iter().filter(|recipe| {
        recipe
            .canonical_model_id
            .to_ascii_lowercase()
            .contains("qwen")
    }) {
        let body = serde_json::json!({
            "object": "list",
            "data": [{ "id": recipe.canonical_model_id, "object": "model" }],
        })
        .to_string();
        assert_eq!(
            endpoint_reports(body, Some("qwen")),
            Ok(true),
            "asking for `qwen` was not satisfied by a served {:?}",
            recipe.canonical_model_id
        );
    }
}

/// The same behavior outside the catalog: a user can `rocm serve` any Hugging
/// Face ref, and readiness compares the engine's reported id against it. A bare
/// family word therefore satisfies a request for any model of that family,
/// whatever its size.
#[test]
fn a_bare_family_word_satisfies_readiness_for_an_uncatalogued_model() {
    for (served, wanted) in [
        ("qwen", "Qwen/Qwen3-235B-A22B-Instruct"),
        ("llama", "meta-llama/Llama-3.1-70B-Instruct"),
        ("gemma", "google/gemma-2-27b-it"),
    ] {
        let body = serde_json::json!({
            "object": "list",
            "data": [{ "id": served, "object": "model" }],
        })
        .to_string();
        assert_eq!(
            endpoint_reports(body, Some(wanted)),
            Ok(true),
            "pinning the current behavior: serving {served:?} reads as ready for {wanted:?}"
        );
    }
}
