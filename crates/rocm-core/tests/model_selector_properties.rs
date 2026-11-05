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

#[path = "common/loopback.rs"]
mod loopback;
#[path = "common/proptest_cases.rs"]
mod proptest_cases;

use std::time::Duration;

use loopback::LoopbackServer;
use proptest::prelude::*;
use proptest_cases::proptest_cases;
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

/// A `/v1/models` document listing exactly `ids`.
fn models_document(ids: &[&str]) -> String {
    serde_json::json!({
        "object": "list",
        "data": ids
            .iter()
            .map(|id| serde_json::json!({ "id": id, "object": "model" }))
            .collect::<Vec<_>>(),
    })
    .to_string()
}

/// Ask the readiness check what it makes of the document `server` serves when
/// expecting `expected`. Panics if the request never reached the server, so a
/// transport failure cannot pass as a "not ready" verdict.
fn readiness_from(server: &LoopbackServer, expected: Option<&str>) -> Result<bool, String> {
    let before = server.requests_served();
    let outcome =
        openai_models_endpoint_has_model(&server.url(), expected, None, Duration::from_secs(10))
            .map_err(|error| error.to_string());
    assert!(
        server.requests_served() > before,
        "the readiness check never reached the test server: {outcome:?}"
    );
    outcome
}

/// Ask the readiness check what it makes of `body` when expecting `expected`.
fn endpoint_reports(body: String, expected: Option<&str>) -> Result<bool, String> {
    let server = LoopbackServer::start(Some("application/json"), body.into_bytes());
    readiness_from(&server, expected)
}

#[test]
fn the_shipped_catalog_is_not_empty() {
    assert!(
        !catalog().is_empty(),
        "these properties are vacuous without a catalog to check"
    );
}

/// Every advertised selector must resolve to the recipe that advertises it.
///
/// That no selector is advertised twice is the catalog schema's own check
/// (`ModelRecipeIndexDocument::validate`, which normalizes refs the same way
/// `matches_ref` does), run on the shipped catalog by its unit tests.
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
    #![proptest_config(ProptestConfig {
        cases: proptest_cases(512),
        ..ProptestConfig::default()
    })]

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

/// `/v1/models` bodies that name no model matching any of the expectations
/// below, shaped the way a real engine shapes them, plus the degenerate shapes
/// a broken or hostile one could send.
///
/// The listed ids carry no letter. Every expected name below does, and every
/// rule readiness matches by — equality, basename, catalog alias, family word —
/// needs a letter in common, so none of these documents may read as ready.
fn models_body_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => prop::collection::vec("[0-9 ._:#/@-]{0,24}", 0..4).prop_map(|ids| {
            let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
            models_document(&ids)
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
    #![proptest_config(ProptestConfig {
        cases: proptest_cases(48),
        ..ProptestConfig::default()
    })]

    /// A `/v1/models` document that names no matching model — malformed,
    /// hostile, or listing only unrelated ids — must produce a clean `false` or
    /// a clean error: never a panic, and never "ready". A check that reported
    /// ready for any non-empty listing, ignoring the expected model, fails here.
    #[test]
    fn a_models_document_naming_no_matching_model_never_reports_ready(
        body in models_body_strategy(),
        expected in prop::sample::select(vec![
            "Qwen/Qwen3-4B-Instruct-2507",
            "meta-llama/Llama-3.1-8B-Instruct",
            "nonexistent/model",
        ]),
    ) {
        let verdict = endpoint_reports(body.clone(), Some(expected));
        prop_assert_ne!(
            verdict,
            Ok(true),
            "reported ready for {:?} from a document naming no matching model: {}",
            expected,
            body
        );
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
        let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
        let body = models_document(&ids);
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
/// Two shapes that *do* cross-match today are excluded here and pinned
/// separately, so they stay visible: bare family words, by
/// [`a_bare_family_word_satisfies_readiness_for_the_whole_family`], and one
/// recipe's alias spelling another recipe's basename, by
/// [`an_alias_spelling_another_models_basename_reads_as_ready_for_it`].
///
/// The request count grows with the square of the catalog (every served name
/// against every wanted model), so each served name gets one loopback server
/// that answers all of its requests, rather than a server per request.
#[test]
fn listing_a_different_catalog_model_is_never_readiness() {
    let recipes = catalog();
    assert!(recipes.len() > 1, "need at least two recipes to confuse");
    let mut false_positives = Vec::new();

    for served in recipes {
        let served_names = std::iter::once(&served.canonical_model_id).chain(&served.aliases);
        for served_name in served_names {
            if is_bare_family_word(served_name) {
                continue;
            }
            let server = LoopbackServer::start(
                Some("application/json"),
                models_document(&[served_name.as_str()]).into_bytes(),
            );
            for wanted in recipes {
                // Skip the served recipe itself, a name the *wanted* recipe
                // also answers to (a genuine shared alias, not a confusion),
                // and the pinned alias/basename collision.
                if wanted.canonical_model_id == served.canonical_model_id
                    || wanted.matches_ref(served_name)
                    || spells_the_basename_of(served_name, wanted)
                {
                    continue;
                }
                if readiness_from(&server, Some(&wanted.canonical_model_id)) == Ok(true) {
                    false_positives.push(format!(
                        "wanted {:?} but an endpoint serving only {:?} reported ready",
                        wanted.canonical_model_id, served_name
                    ));
                }
            }
        }
    }

    for wanted in recipes {
        for served_name in near_miss_names(&wanted.canonical_model_id) {
            if wanted.matches_ref(&served_name) {
                continue;
            }
            let body = models_document(&[served_name.as_str()]);
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

/// Whether `name`, as an engine would report it, is exactly the basename of
/// `recipe`'s canonical id (up to ASCII case) — the part after the last `/`,
/// which readiness accepts on its own because engines commonly report a model
/// by its basename.
fn spells_the_basename_of(name: &str, recipe: &ModelRecipeRecord) -> bool {
    let basename = recipe
        .canonical_model_id
        .rsplit('/')
        .next()
        .unwrap_or(&recipe.canonical_model_id);
    name.trim().eq_ignore_ascii_case(basename)
}

/// Characterization test for a known false-positive surface, pinned so it is
/// visible rather than merely absent from the suite.
///
/// Readiness accepts a model's basename for the model (an engine launched on
/// `Qwen/Qwen3-32B` commonly reports `Qwen3-32B`). Some catalog aliases spell
/// the basename of a *different* catalog model: `qwen3-32b` is an alias of
/// `Qwen/Qwen3-32B-FP8` and also the basename of `Qwen/Qwen3-32B`. An engine
/// launched on the FP8 recipe under that alias reports `qwen3-32b`, and that
/// reads as ready for `Qwen/Qwen3-32B` — readiness cannot tell the two apart
/// from the name alone.
///
/// The collisions are derived from the catalog and compared with the list
/// below, so a catalog change that adds one fails here and has to be looked
/// at rather than being silently excluded from
/// [`listing_a_different_catalog_model_is_never_readiness`]. If readiness or
/// the catalog aliases are changed to remove a collision, update the list
/// along with them.
#[test]
fn an_alias_spelling_another_models_basename_reads_as_ready_for_it() {
    let mut collisions = Vec::new();
    for owner in catalog() {
        for alias in &owner.aliases {
            for other in catalog() {
                if other.canonical_model_id != owner.canonical_model_id
                    && !other.matches_ref(alias)
                    && spells_the_basename_of(alias, other)
                {
                    collisions.push((
                        alias.as_str(),
                        owner.canonical_model_id.as_str(),
                        other.canonical_model_id.as_str(),
                    ));
                }
            }
        }
    }
    collisions.sort_unstable();
    assert_eq!(
        collisions,
        vec![
            ("qwen3-32b", "Qwen/Qwen3-32B-FP8", "Qwen/Qwen3-32B"),
            (
                "qwen3.6-35b-a3b",
                "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
                "Qwen/Qwen3.6-35B-A3B"
            ),
        ],
        "(alias, recipe it belongs to, recipe whose basename it spells) — the catalog's \
         alias/basename collisions changed"
    );

    for (alias, owner, other) in collisions {
        assert_eq!(
            endpoint_reports(models_document(&[alias]), Some(other)),
            Ok(true),
            "pinning the current behavior: {owner:?} served as {alias:?} reads as ready for \
             {other:?}"
        );
    }
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
                let body = models_document(&[served]);
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
/// exercised by an existing unit test in `endpoint_readiness.rs`: asking for `qwen` is
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
    let body = models_document(&["qwen"]);

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

    // The other direction, which the `endpoint_readiness.rs` unit test also exercises: asking
    // for the bare family word is satisfied by an engine serving any catalog
    // model of that family.
    for recipe in catalog().iter().filter(|recipe| {
        recipe
            .canonical_model_id
            .to_ascii_lowercase()
            .contains("qwen")
    }) {
        let body = models_document(&[recipe.canonical_model_id.as_str()]);
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
        let body = models_document(&[served]);
        assert_eq!(
            endpoint_reports(body, Some(wanted)),
            Ok(true),
            "pinning the current behavior: serving {served:?} reads as ready for {wanted:?}"
        );
    }
}
