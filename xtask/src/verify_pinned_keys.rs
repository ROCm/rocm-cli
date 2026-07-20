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
//! - **CI cross-check.** When `ROCM_CLI_SIGNING_PUBLIC_KEY_PEM` is set (as in
//!   release/nightly CI), it must equal the canonical *current* release key, so the
//!   key CI verifies against is exactly the one users pin.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

const PEM_BEGIN: &str = "-----BEGIN PUBLIC KEY-----";
const PEM_END: &str = "-----END PUBLIC KEY-----";

const CURRENT_RELEASE_KEY: &str = "release-current";

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
/// is the inline PEM CI would verify against (from `ROCM_CLI_SIGNING_PUBLIC_KEY_PEM`)
/// or `None`; it is passed in rather than read here so tests need not mutate the
/// process environment.
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
fn check_ci_public_key(
    ci_public_key: Option<&str>,
    current_release_body: Option<&str>,
) -> Result<Vec<String>> {
    let env_pem = ci_public_key.unwrap_or_default();
    if env_pem.trim().is_empty() {
        return Ok(Vec::new());
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

fn repo_root() -> PathBuf {
    // `CARGO_MANIFEST_DIR` is the `xtask/` crate dir; its parent is the repo root,
    // so this is correct regardless of the working directory CI invokes us from.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask crate has a parent directory")
        .to_path_buf()
}

pub fn run() -> Result<()> {
    let ci_public_key = std::env::var("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM").ok();
    for message in check_pinned_keys(&repo_root(), ci_public_key.as_deref())? {
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

        // No CI key configured -> nothing to assert.
        assert!(
            check_ci_public_key(None, Some(&current))
                .unwrap()
                .is_empty()
        );
        assert!(
            check_ci_public_key(Some("   "), Some(&current))
                .unwrap()
                .is_empty()
        );

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
}
