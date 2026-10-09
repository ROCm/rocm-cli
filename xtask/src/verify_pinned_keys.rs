// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Assert the release/metadata public keys pinned in the installers and
//! `apps/rocm/src/therock.rs` match the canonical published keys under
//! `docs/keys/`.
//!
//! This closes the gap where CI verifies release artifacts against the public key
//! configured in its environment (see `scripts/release_readiness.py`) while
//! installers and the binary trust a *separately embedded* copy. If those diverge,
//! CI can bless a release that installers then reject once default-on verification
//! lands. This check enforces a single source of truth: the committed
//! `docs/keys/*.pem` files.
//!
//! Design notes:
//!
//! - **Dormant by default.** Until a canonical `docs/keys/` file exists and is
//!   non-empty, its key is skipped. So this passes as a no-op today (empty pinned
//!   sentinels, no canonical files) and only starts enforcing once real keys are
//!   published.
//! - **Per-constant equality.** Each pinned key is embedded in a specific named
//!   constant (shell `NAME="…"`, PowerShell string/here-string, Rust `const`). We
//!   isolate *that constant's own value span*, reduce it to its base64 body, and
//!   compare it for equality to the canonical key. Comparing the whole file (rather
//!   than the specific constant) would let a wrong/tampered constant pass whenever
//!   the canonical body appears anywhere else in the file — e.g. in the `NEXT` slot
//!   mid-rotation or a stale comment.
//! - **CI cross-check.** When a CI signing public key is configured — the file
//!   named by `ROCM_CLI_SIGNING_PUBLIC_KEY_PATH`, else the inline
//!   `ROCM_CLI_SIGNING_PUBLIC_KEY_PEM` release/nightly CI wires from the secret —
//!   it must equal the canonical *current* release key, so the key CI verifies
//!   against is exactly the one users pin. Both environment sources are resolved
//!   in the same order `scripts/release_readiness.py` uses (path before PEM). An
//!   explicit `--public-key` passed to the gate is not visible here, and this
//!   runs as its own workflow step, so a *relative* path would resolve against
//!   each process's own working directory. No workflow does either today.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

const PEM_BEGIN: &str = "-----BEGIN PUBLIC KEY-----";
const PEM_END: &str = "-----END PUBLIC KEY-----";

const CURRENT_RELEASE_KEY: &str = "release-current";

/// The key sources, in the order `scripts/release_readiness.py` resolves them.
const CI_PUBLIC_KEY_PATH_ENV: &str = "ROCM_CLI_SIGNING_PUBLIC_KEY_PATH";
const CI_PUBLIC_KEY_PEM_ENV: &str = "ROCM_CLI_SIGNING_PUBLIC_KEY_PEM";

/// How a pinned constant embeds its PEM string, so its value span can be isolated.
#[derive(Clone, Copy)]
enum Embedding {
    /// Shell double-quoted assignment: `NAME="…"` (value runs to the next `"`).
    Shell,
    /// Rust string const: `const NAME: &str = "…";` (value runs to the next `"`).
    Rust,
    /// PowerShell string or here-string: `$Name = "…"` or `$Name = @"…"@`.
    PowerShell,
}

/// A source file plus the specific constant within it that must embed the key.
struct Source {
    path: &'static str,
    /// Identifier that holds the pinned PEM (includes the leading `$` for PowerShell).
    token: &'static str,
    embedding: Embedding,
}

/// A canonical published key and every source that must embed the identical bytes.
struct PinnedKey {
    name: &'static str,
    canonical: &'static str,
    sources: &'static [Source],
}

const PINNED_KEYS: &[PinnedKey] = &[
    PinnedKey {
        name: CURRENT_RELEASE_KEY,
        canonical: "docs/keys/rocm-cli-release-current-public.pem",
        sources: &[
            Source {
                path: "install.sh",
                token: "PINNED_RELEASE_PUBLIC_KEY_CURRENT",
                embedding: Embedding::Shell,
            },
            Source {
                path: "install.ps1",
                token: "$PinnedReleasePublicKeyCurrent",
                embedding: Embedding::PowerShell,
            },
        ],
    },
    PinnedKey {
        name: "release-next",
        canonical: "docs/keys/rocm-cli-release-next-public.pem",
        sources: &[
            Source {
                path: "install.sh",
                token: "PINNED_RELEASE_PUBLIC_KEY_NEXT",
                embedding: Embedding::Shell,
            },
            Source {
                path: "install.ps1",
                token: "$PinnedReleasePublicKeyNext",
                embedding: Embedding::PowerShell,
            },
        ],
    },
    PinnedKey {
        name: "metadata",
        canonical: "docs/keys/rocm-cli-metadata-public.pem",
        sources: &[Source {
            path: "apps/rocm/src/therock.rs",
            token: "PINNED_METADATA_PUBLIC_KEY_PEM",
            embedding: Embedding::Rust,
        }],
    },
];

/// Reduce arbitrary text to its base64 alphabet, dropping everything else.
///
/// Newline escape sequences are removed first: a literal `\n` (shell/Rust) or
/// `` `n `` (PowerShell) would otherwise leave a stray `n`/`r`/`t` — all
/// base64-alphabet letters — and corrupt the payload. Real newlines are plain
/// whitespace and drop out with everything else non-base64.
fn base64_only(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if (c == '\\' || c == '`') && i + 1 < bytes.len() {
            let next = bytes[i + 1] as char;
            if next == 'n' || next == 'r' || next == 't' {
                i += 2;
                continue;
            }
        }
        if c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=' {
            out.push(c);
        }
        i += 1;
    }
    out
}

/// Return the base64 body of the first PUBLIC KEY PEM in `text`, or `None`.
fn pem_body(text: &str) -> Option<String> {
    let begin = text.find(PEM_BEGIN)?;
    let start = begin + PEM_BEGIN.len();
    let end = text[start..].find(PEM_END)? + start;
    let body = base64_only(&text[start..end]);
    if body.is_empty() { None } else { Some(body) }
}

/// Isolate the raw text span assigned to `token` under `embedding`, or `None` if the
/// assignment is not found. PEM payloads contain no `"`, so the first quote after the
/// assignment reliably closes the value.
fn extract_constant<'a>(source: &'a str, token: &str, embedding: Embedding) -> Option<&'a str> {
    let key = source.find(token)?;
    let after = &source[key + token.len()..];
    match embedding {
        Embedding::Shell => {
            // `NAME="…"` — the token is immediately followed by `="`.
            let open = after.find('"')?;
            let rest = &after[open + 1..];
            let close = rest.find('"')?;
            Some(&rest[..close])
        }
        Embedding::Rust => {
            // `const NAME: &str = "…";` — skip to the `=`, then the opening quote.
            let eq = after.find('=')?;
            let open = after[eq..].find('"')? + eq;
            let rest = &after[open + 1..];
            let close = rest.find('"')?;
            Some(&rest[..close])
        }
        Embedding::PowerShell => {
            let here = after.find("@\"");
            let quote = after.find('"');
            match (here, quote) {
                // Here-string `= @"…"@`: the `@"` quote is the first quote seen.
                (Some(h), Some(q)) if q == h + 1 => {
                    let rest = &after[h + 2..];
                    let close = rest.find("\"@")?;
                    Some(&rest[..close])
                }
                // Plain `= "…"`.
                (_, Some(q)) => {
                    let rest = &after[q + 1..];
                    let close = rest.find('"')?;
                    Some(&rest[..close])
                }
                _ => None,
            }
        }
    }
}

/// Check every populated canonical key against its pinned sources. `ci_public_key`
/// is the PEM text CI would verify against, already resolved by
/// [`resolve_ci_public_key`] from either key source, or `None`; it is passed in
/// rather than read here so tests need not mutate the process environment.
fn check_pinned_keys(root: &Path, ci_public_key: Option<&str>) -> Result<Vec<String>> {
    let mut messages = Vec::new();
    let mut current_release_body: Option<String> = None;

    for entry in PINNED_KEYS {
        let canonical_path = root.join(entry.canonical);
        let canonical_text = match std::fs::read_to_string(&canonical_path) {
            Ok(text) if !text.trim().is_empty() => text,
            _ => {
                messages.push(format!(
                    "{}: no canonical key yet — skipped (dormant)",
                    entry.name
                ));
                continue;
            }
        };
        let canonical_body = pem_body(&canonical_text).with_context(|| {
            format!(
                "{}: {} is not a valid PUBLIC KEY PEM",
                entry.name, entry.canonical
            )
        })?;
        if entry.name == CURRENT_RELEASE_KEY {
            current_release_body = Some(canonical_body.clone());
        }

        for source in entry.sources {
            let source_path = root.join(source.path);
            let source_text = std::fs::read_to_string(&source_path).with_context(|| {
                format!("{}: source file is missing: {}", entry.name, source.path)
            })?;
            let embedded =
                extract_constant(&source_text, source.token, source.embedding).and_then(pem_body);
            if embedded.as_deref() != Some(canonical_body.as_str()) {
                bail!(
                    "{}: {} does not embed the canonical key {} in {} — the pinned \
                     constant and the published key have diverged",
                    entry.name,
                    source.path,
                    entry.canonical,
                    source.token
                );
            }
        }
        messages.push(format!(
            "{}: pinned in {}",
            entry.name,
            entry
                .sources
                .iter()
                .map(|s| s.path)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    messages.extend(check_ci_public_key(
        ci_public_key,
        current_release_body.as_deref(),
    )?);
    Ok(messages)
}

/// When a CI signing public key is configured, it must equal the canonical current
/// release key — the key CI verifies release artifacts against.
///
/// An absent key is reported rather than passed over in silence. This check cannot
/// *demand* one: `ci.yml` never exposes the signing secret, so requiring it would
/// fail every PR. But "no key configured" and "key matches" must not look alike in
/// the log, or a secret that was rotated away reads as a clean run.
fn check_ci_public_key(
    ci_public_key: Option<&str>,
    current_release_body: Option<&str>,
) -> Result<Vec<String>> {
    let env_pem = ci_public_key.unwrap_or_default();
    if env_pem.trim().is_empty() {
        return Ok(vec![
            "ci public key: not configured — cross-check skipped".to_owned(),
        ]);
    }
    let env_body = pem_body(env_pem)
        .context("the configured CI signing public key is not a valid PUBLIC KEY PEM")?;
    let Some(current) = current_release_body else {
        return Ok(vec![
            "ci public key: configured, but no canonical current release key to compare \
             against yet — skipped"
                .to_owned(),
        ]);
    };
    if env_body != current {
        bail!(
            "the CI signing public key does not match the canonical current release key \
             ({}); CI would verify against a different key than installers pin",
            PINNED_KEYS[0].canonical
        );
    }
    Ok(vec![
        "ci public key: matches the canonical current release key".to_owned(),
    ])
}

/// Resolve the CI signing public key the way `scripts/release_readiness.py`'s
/// `resolve_signing_key` does: the file named by `ROCM_CLI_SIGNING_PUBLIC_KEY_PATH`
/// first, then the inline `ROCM_CLI_SIGNING_PUBLIC_KEY_PEM`.
///
/// Reading only the PEM would make the cross-check skip silently whenever the key
/// was supplied by path, while the readiness gate verified release artifacts
/// against that very key — a trust root nobody compared with `docs/keys/`, and
/// reported in the log as "not configured".
///
/// An empty variable counts as absent; a variable naming a missing or empty file
/// is an error. The blank case matches the Python gate exactly: `env_path` and
/// `env_text` both `strip()` the value and treat whatever is left of nothing as
/// unset, which is the shape an unset GitHub secret takes when it expands to the
/// empty string. Erroring on a whitespace-only path here while the gate fell
/// through to the PEM would make the two disagree about which key is in use.
///
/// Once a path *is* named, a file that cannot be read or that holds nothing is an
/// error rather than a fall-through: the operator named a key, so reporting "not
/// configured" and exiting 0 would be the exact confusion the doc on
/// [`check_ci_public_key`] forbids.
///
/// `lookup` is injected so the variable names themselves are covered: reading
/// them directly here left callers free to query a misspelled name with every
/// test still green.
fn resolve_ci_public_key(lookup: impl Fn(&str) -> Option<OsString>) -> Result<Option<String>> {
    if let Some(raw) = lookup(CI_PUBLIC_KEY_PATH_ENV) {
        // `OsString`, not `String`: a non-UTF-8 path must still name the file the
        // operator meant, not silently fall through to the inline PEM. Lossy
        // replacement characters are not whitespace, so such a path still counts
        // as named.
        if !raw.to_string_lossy().trim().is_empty() {
            let path = PathBuf::from(&raw);
            let pem = std::fs::read_to_string(&path).with_context(|| {
                format!(
                    "failed to read the CI signing public key named by \
                     {CI_PUBLIC_KEY_PATH_ENV}: {}",
                    path.display()
                )
            })?;
            if pem.trim().is_empty() {
                bail!(
                    "the CI signing public key named by {CI_PUBLIC_KEY_PATH_ENV} is \
                     empty: {}; a configured key that holds nothing must not be \
                     reported as 'not configured'",
                    path.display()
                );
            }
            return Ok(Some(pem));
        }
    }
    Ok(lookup(CI_PUBLIC_KEY_PEM_ENV)
        .map(|raw| raw.to_string_lossy().into_owned())
        .filter(|value| !value.trim().is_empty()))
}

fn repo_root() -> PathBuf {
    // `CARGO_MANIFEST_DIR` is the `xtask/` crate dir; its parent is the repo root,
    // so this is correct regardless of the working directory CI invokes us from.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask crate has a parent directory")
        .to_path_buf()
}

/// A named shim rather than a closure in [`run`]: the generic `std::env::var_os`
/// fn item cannot satisfy the higher-ranked `Fn(&str)` bound, and a closure
/// spelled out at the call site is a place a wrong variable name could live
/// where no test could reach it. Here it is a pass-through with nothing to get
/// wrong, and `env_lookup_passes_the_name_through` pins that it stays one.
fn env_lookup(name: &str) -> Option<OsString> {
    std::env::var_os(name)
}

/// Resolve the CI key through `lookup`, then run every pinned-key check against
/// `root`, returning the lines the caller should print.
///
/// Split out of [`run`] so the resolution-to-comparison handoff is testable: with
/// it inlined there, a `run` that queried the wrong variable, resolved nothing,
/// or simply handed `check_pinned_keys` a `None` left every test green while the
/// cross-check silently degraded to "not configured".
fn run_with(root: &Path, lookup: impl Fn(&str) -> Option<OsString>) -> Result<Vec<String>> {
    let ci_public_key = resolve_ci_public_key(lookup)?;
    check_pinned_keys(root, ci_public_key.as_deref())
}

pub fn run() -> Result<()> {
    for message in run_with(&repo_root(), env_lookup)? {
        println!("pinned key check: {message}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "-----BEGIN PUBLIC KEY-----\n\
        MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAtesttesttesttest+/\n\
        abcDEF0123456789testtesttesttesttesttesttesttesttesttestABCD==\n\
        -----END PUBLIC KEY-----\n";

    #[test]
    fn base64_only_strips_escapes_and_non_alphabet() {
        assert_eq!(base64_only("AB\\nCD `n EF-=+/"), "ABCDEF=+/");
    }

    #[test]
    fn pem_body_extracts_inner_base64() {
        let body = pem_body(SAMPLE).expect("valid pem");
        assert!(body.starts_with("MIIBIjAN"));
        assert!(!body.contains("BEGIN"));
        assert_eq!(pem_body("not a pem"), None);
    }

    #[test]
    fn extract_constant_isolates_each_syntax() {
        let shell = "PINNED_RELEASE_PUBLIC_KEY_CURRENT=\"the-current\"\n\
                     PINNED_RELEASE_PUBLIC_KEY_NEXT=\"the-next\"\n";
        assert_eq!(
            extract_constant(shell, "PINNED_RELEASE_PUBLIC_KEY_CURRENT", Embedding::Shell),
            Some("the-current")
        );
        assert_eq!(
            extract_constant(shell, "PINNED_RELEASE_PUBLIC_KEY_NEXT", Embedding::Shell),
            Some("the-next")
        );

        let rust = "const PINNED_METADATA_PUBLIC_KEY_PEM: &str = \"the-meta\";\n";
        assert_eq!(
            extract_constant(rust, "PINNED_METADATA_PUBLIC_KEY_PEM", Embedding::Rust),
            Some("the-meta")
        );

        let ps = "$PinnedReleasePublicKeyCurrent = @\"\nthe-here\n\"@\n\
                  $PinnedReleasePublicKeyNext = \"the-plain\"\n";
        assert_eq!(
            extract_constant(ps, "$PinnedReleasePublicKeyCurrent", Embedding::PowerShell),
            Some("\nthe-here\n")
        );
        assert_eq!(
            extract_constant(ps, "$PinnedReleasePublicKeyNext", Embedding::PowerShell),
            Some("the-plain")
        );
    }

    #[test]
    fn extract_targets_the_named_constant_not_the_whole_file() {
        // The bug: a wrong CURRENT passes if the real key is anywhere in the file.
        // Here CURRENT holds a wrong key while NEXT holds the real one; extracting
        // CURRENT must return the wrong value, not the file's real key.
        let shell = format!(
            "PINNED_RELEASE_PUBLIC_KEY_CURRENT=\"{}\"\nPINNED_RELEASE_PUBLIC_KEY_NEXT=\"{}\"\n",
            SAMPLE.replace("test", "wrong"),
            SAMPLE
        );
        let current = extract_constant(
            &shell,
            "PINNED_RELEASE_PUBLIC_KEY_CURRENT",
            Embedding::Shell,
        )
        .and_then(pem_body)
        .unwrap();
        assert_ne!(current, pem_body(SAMPLE).unwrap());
    }

    #[test]
    fn ci_public_key_cross_check() {
        let current = pem_body(SAMPLE).unwrap();

        // No CI key configured -> nothing to assert, but the skip is reported so an
        // unset/rotated-away secret does not read as a passing cross-check.
        for absent in [None, Some("   "), Some("")] {
            let reported = check_ci_public_key(absent, Some(&current)).unwrap();
            assert!(
                reported.iter().any(|m| m.contains("not configured")),
                "absent CI key should report a skip, got {reported:?}"
            );
        }

        // Configured but no canonical current key yet -> skipped, not an error.
        let skipped = check_ci_public_key(Some(SAMPLE), None).unwrap();
        assert!(skipped.iter().any(|m| m.contains("skipped")));

        // Matching CI key -> accepted.
        assert!(check_ci_public_key(Some(SAMPLE), Some(&current)).is_ok());

        // Mismatched CI key -> rejected (CI would verify against a different key).
        let other = SAMPLE.replace("test", "diff");
        assert!(check_ci_public_key(Some(&other), Some(&current)).is_err());

        // Malformed CI key -> rejected.
        assert!(check_ci_public_key(Some("not a pem"), Some(&current)).is_err());
    }

    /// Each variable name must be the one the release gate reads *for that role*.
    ///
    /// The lookup tests below match on these same constants, so they cannot see a
    /// wrong *value*: misspelling one leaves every case green while `run` queries
    /// a variable nothing ever sets, and the cross-check silently skips.
    ///
    /// Matching the gate's assignment lines, not merely the name somewhere in the
    /// file, is what makes this bite. Both literals also appear in
    /// `PRODUCTION_TRUST_ENV_NAMES` and `validate_production_trust`, so a
    /// name-anywhere search passed three real regressions: swapping the two
    /// constants (release CI would read the PEM text as a file path), and pointing
    /// either at `ROCM_CLI_METADATA_*` (the cross-check would quietly skip).
    /// Pinning `BINDING = "NAME"` ties each name to the role the gate gives it.
    #[test]
    fn ci_public_key_env_names_match_the_release_gate() {
        let gate = std::fs::read_to_string(repo_root().join("scripts/release_readiness.py"))
            .expect("read scripts/release_readiness.py");
        for (binding, name) in [
            ("SIGNING_PUBLIC_KEY_PATH_ENV", CI_PUBLIC_KEY_PATH_ENV),
            ("SIGNING_PUBLIC_KEY_ENV", CI_PUBLIC_KEY_PEM_ENV),
        ] {
            let assignment = format!("\n{binding} = \"{name}\"\n");
            assert!(
                gate.contains(&assignment),
                "release_readiness.py does not bind {binding} to {name}; this \
                 cross-check would resolve a different key than the gate verifies \
                 with. Expected the line {assignment:?}"
            );
        }
    }

    /// The cross-check must see a key supplied by path, not just the inline PEM.
    ///
    /// `release_readiness.py` prefers `ROCM_CLI_SIGNING_PUBLIC_KEY_PATH` and
    /// verifies artifacts against it. Resolving only the PEM here meant that key
    /// was never compared with `docs/keys/`, and the run still logged
    /// "not configured — cross-check skipped" while a key was in use.
    #[test]
    fn ci_public_key_resolves_the_path_variable_before_the_inline_pem() {
        let dir = tempfile::tempdir().expect("tempdir");
        let key_file = dir.path().join("ci-public-key.pem");
        std::fs::write(&key_file, SAMPLE).expect("write key");
        let other = SAMPLE.replace("test", "diff");

        // A lookup over a fixed map, so the *names* queried are covered too: with
        // the variables read directly, `run()` could query a misspelled name and
        // every case here would still pass.
        let env = |path: Option<&str>, pem: Option<&str>| {
            let path = path.map(OsString::from);
            let pem = pem.map(OsString::from);
            move |name: &str| match name {
                CI_PUBLIC_KEY_PATH_ENV => path.clone(),
                CI_PUBLIC_KEY_PEM_ENV => pem.clone(),
                other => panic!("unexpected variable queried: {other}"),
            }
        };
        let named = key_file.to_str().expect("utf-8 path");

        // Path wins over the PEM, so the key cross-checked is the key verified with.
        let resolved = resolve_ci_public_key(env(Some(named), Some(&other)))
            .expect("readable key")
            .expect("a key");
        assert_eq!(pem_body(&resolved).unwrap(), pem_body(SAMPLE).unwrap());

        // ...and that resolved key is what reaches the comparison: a path key that
        // disagrees with the canonical one must fail rather than skip.
        let canonical = pem_body(&other).unwrap();
        assert!(check_ci_public_key(Some(&resolved), Some(&canonical)).is_err());

        // The PEM is still used when no path is named.
        assert_eq!(
            resolve_ci_public_key(env(None, Some(SAMPLE)))
                .expect("inline pem")
                .as_deref(),
            Some(SAMPLE)
        );

        // Unset counts as absent -- the shape of an unset GitHub secret, which
        // expands to the empty string.
        for blank in [None, Some(""), Some("   ")] {
            assert!(
                resolve_ci_public_key(env(None, blank))
                    .expect("no key")
                    .is_none(),
                "blank PEM {blank:?} should resolve to no key"
            );
        }
        // Same for the path variable, and whitespace counts as blank: the Python
        // gate's `env_path` strips the value and falls through to the PEM, so
        // erroring here instead would make the two sides disagree about which key
        // a release is verified with.
        for blank in [None, Some(""), Some("   "), Some("\t\n")] {
            assert_eq!(
                resolve_ci_public_key(env(blank, Some(SAMPLE)))
                    .unwrap_or_else(|e| panic!("blank path {blank:?} should not error: {e}"))
                    .as_deref(),
                Some(SAMPLE),
                "a blank path variable {blank:?} should fall through to the PEM"
            );
        }

        // A named key that cannot be read is an error, never a quiet fall-through
        // to a different key than the operator named. The *message* is pinned too:
        // a failed read that fell through as an empty string would still be an
        // error, but the operator would be told the key "is empty" and go looking
        // at a file that is fine.
        let missing = dir.path().join("absent.pem");
        let read_error = resolve_ci_public_key(env(missing.to_str(), Some(SAMPLE)))
            .expect_err("an unreadable named key must fail rather than fall back to the PEM");
        assert!(
            format!("{read_error:#}").contains("failed to read"),
            "an unreadable named key should report the read failure, got: {read_error:#}"
        );

        // A named key that exists but holds nothing must be an error too. Passing
        // it on as `Some("")` made `check_ci_public_key` trim it away and report
        // "not configured" while a key was in fact configured -- exactly the
        // confusion that check's own doc forbids.
        for empty in ["", "   \n\t "] {
            let blank_file = dir.path().join("blank.pem");
            std::fs::write(&blank_file, empty).expect("write blank key");
            let error = resolve_ci_public_key(env(blank_file.to_str(), Some(SAMPLE)))
                .expect_err("an empty named key must be an error");
            assert!(
                error.to_string().contains("is empty"),
                "unexpected error for an empty named key: {error}"
            );
        }
    }

    /// `run_with` is the whole path `run` takes, so the resolved key must reach
    /// the comparison rather than being dropped on the way.
    ///
    /// Nothing previously ran through this seam. A `run` whose lookup always
    /// queried the PEM name, always returned `None`, or simply handed
    /// `check_pinned_keys` a `None` turned the path resolution into a silent
    /// no-op and reported "not configured — cross-check skipped" on a run that
    /// did have a key. Each case below fails under exactly those mutations.
    #[test]
    fn run_with_carries_the_resolved_key_into_the_comparison() {
        let root = repo_root();
        let canonical =
            std::fs::read_to_string(root.join("docs/keys/rocm-cli-release-current-public.pem"))
                .expect("the canonical current release key is committed");
        let dir = tempfile::tempdir().expect("tempdir");

        let env = |path: Option<&str>, pem: Option<&str>| {
            let path = path.map(OsString::from);
            let pem = pem.map(OsString::from);
            move |name: &str| match name {
                CI_PUBLIC_KEY_PATH_ENV => path.clone(),
                CI_PUBLIC_KEY_PEM_ENV => pem.clone(),
                other => panic!("unexpected variable queried: {other}"),
            }
        };

        // No key anywhere: the run passes, and says the cross-check was skipped.
        // This is the only place that message is exercised through the code path
        // that actually prints it.
        let skipped = run_with(&root, env(None, None)).expect("no key configured is not an error");
        assert!(
            skipped
                .iter()
                .any(|m| m.contains("not configured — cross-check skipped")),
            "a run with no key must say the cross-check was skipped, got {skipped:?}"
        );

        // A key named by path must be compared, not skipped: a wrong one fails.
        let wrong = dir.path().join("wrong.pem");
        std::fs::write(&wrong, SAMPLE).expect("write wrong key");
        let error = run_with(&root, env(wrong.to_str(), None))
            .expect_err("a path key that disagrees with docs/keys must fail the run");
        assert!(
            format!("{error:#}").contains("does not match the canonical current release key"),
            "unexpected failure for a mismatched path key: {error:#}"
        );

        // ...and the right one passes, reported as a match rather than a skip.
        let right = dir.path().join("right.pem");
        std::fs::write(&right, &canonical).expect("write canonical key");
        let matched = run_with(&root, env(right.to_str(), None)).expect("canonical key matches");
        assert!(
            matched
                .iter()
                .any(|m| m.contains("matches the canonical current release key")),
            "a path key equal to docs/keys must be reported as a match, got {matched:?}"
        );

        // The PEM source still reaches the comparison too, and loses to the path.
        let via_pem = run_with(&root, env(None, Some(&canonical))).expect("canonical pem matches");
        assert!(
            via_pem
                .iter()
                .any(|m| m.contains("matches the canonical current release key")),
            "an inline canonical PEM must be reported as a match, got {via_pem:?}"
        );
        assert!(
            run_with(&root, env(wrong.to_str(), Some(&canonical))).is_err(),
            "the path key must win, so a wrong path key is not rescued by a good PEM"
        );
    }

    /// `run`'s only remaining untestable surface is this shim, so pin it.
    ///
    /// It must pass the name it is given straight to the environment. A shim that
    /// ignored its argument, or always answered `None`, would make `run` resolve
    /// the wrong variable or no variable at all while every fake-lookup test above
    /// stayed green. `PATH` is used because it is set in every environment this
    /// runs in, so a shim that ignores the name cannot accidentally agree.
    #[test]
    fn env_lookup_passes_the_name_through() {
        assert_eq!(env_lookup("PATH"), std::env::var_os("PATH"));
        assert!(
            env_lookup("PATH").is_some(),
            "PATH is unset, so this test cannot tell a pass-through from a stub"
        );
        for name in [CI_PUBLIC_KEY_PATH_ENV, CI_PUBLIC_KEY_PEM_ENV] {
            assert_eq!(
                env_lookup(name),
                std::env::var_os(name),
                "{name} must be read from the environment under its own name"
            );
        }
        assert_eq!(
            env_lookup("ROCM_CLI_VERIFY_PINNED_KEYS_NOT_A_REAL_VARIABLE"),
            None,
            "an unset variable must resolve to None, not to some other variable"
        );
    }
}
