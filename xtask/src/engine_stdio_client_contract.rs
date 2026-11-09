// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Static contract test: only the shared client may send an engine child its
//! request.
//!
//! `rocm` and `rocmd` used to carry their own copies of the JSON-over-stdio
//! request to an engine process, and the two drifted (different environment,
//! different treatment of a failing exit, a SIGPIPE guard in only one). The
//! single client in `crates/rocm-engine-protocol/src/stdio_client.rs` replaced
//! both. The behavioural tests of that client and of each caller catch a copy
//! that *behaves* differently; none of them can see that a second copy exists,
//! because the regression is a new function nobody calls from a test.
//!
//! So this guard is on the ingredient. A client has to put a request on a
//! child's stdin, which takes a piped stdin, and the request is an engine
//! envelope, so a file that does both is a client. [`CLIENT`] is the only file
//! allowed to; a caller reaches an engine through `run_engine_stdio` and has
//! neither a piped stdin nor a need to name an envelope for a child. A second
//! client has to either use the shared one or add itself to [`ALLOWED`] in a
//! reviewed diff that says why, as for [`crate::crate_edges::ALLOWLIST`].
//!
//! What this does **not** claim: it is a source-text scan, not a call-graph
//! analysis. A copy that builds its stdin another way (a pipe made by hand), or
//! reaches the envelope types under another name, is outside what it can see,
//! and it says nothing about whether a caller then uses the shared client
//! correctly (the callers' own tests pin that). It closes the cheap, likely
//! regression: somebody pasting a spawn-write-read helper back into a binary.
//!
//! Bidirectional, like the other allowlists: an entry that no longer matches
//! fails too, so the exemption does not outlive its reason.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    /// The one file that is the client.
    const CLIENT: &str = "crates/rocm-engine-protocol/src/stdio_client.rs";

    /// Every file that may both pipe a child's stdin and name an engine
    /// envelope, with the reason it is allowed to.
    const ALLOWED: &[(&str, &str)] = &[(
        CLIENT,
        "the shared client: it spawns the engine child, writes the request line \
         to its stdin and reads the answer",
    )];

    /// This file, which the scan must skip: its fixtures quote both ingredients
    /// as data, and a text scan cannot tell a fixture string from code.
    const SELF_PATH: &str = "xtask/src/engine_stdio_client_contract.rs";

    /// What names an engine envelope: the request and response types, or the
    /// helper that builds a request.
    const ENVELOPE_TOKENS: &[&str] = &[
        "EngineRequestEnvelope",
        "EngineResponseEnvelope",
        "engine_request_envelope",
    ];

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    /// Every `.rs` file in the repository as repo-relative, `/`-separated
    /// paths. `target/` and dot-directories are skipped so the verdict never
    /// depends on state outside the commit under test (the same two exclusions
    /// as the other contract tests here).
    fn workspace_rust_sources(root: &Path) -> Vec<String> {
        let mut found = Vec::new();
        collect_rust_files(root, root, &mut found);
        assert!(
            !found.is_empty(),
            "found no Rust sources to scan -- the walk is broken, not the tree"
        );
        found.sort();
        found
    }

    fn collect_rust_files(root: &Path, dir: &Path, found: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skip = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_none_or(|name| name == "target" || name.starts_with('.'));
                if skip {
                    continue;
                }
                collect_rust_files(root, &path, found);
            } else if path.extension().is_some_and(|ext| ext == "rs")
                && let Ok(relative) = path.strip_prefix(root)
            {
                found.push(relative.to_string_lossy().replace('\\', "/"));
            }
        }
    }

    /// `line` with any `//` comment tail removed, so prose that names the
    /// ingredients is not read as code. Crude on purpose (it does not track
    /// string literals): that fails open only for a pathological line.
    fn strip_comment(line: &str) -> &str {
        match line.find("//") {
            Some(at) => &line[..at],
            None => line,
        }
    }

    /// Whether `text` pipes a child's stdin *and* names an engine envelope,
    /// ignoring comments: the two ingredients of an engine stdio client.
    fn is_a_client(text: &str) -> bool {
        let code: Vec<&str> = text.lines().map(strip_comment).collect();
        let pipes_stdin = code
            .iter()
            .any(|line| line.contains(".stdin(") && line.contains("piped()"));
        let names_envelope = code
            .iter()
            .any(|line| ENVELOPE_TOKENS.iter().any(|token| line.contains(token)));
        pipes_stdin && names_envelope
    }

    fn client_files(root: &Path) -> Vec<String> {
        workspace_rust_sources(root)
            .into_iter()
            .filter(|relative| relative != SELF_PATH)
            .filter(|relative| {
                std::fs::read_to_string(root.join(relative)).is_ok_and(|text| is_a_client(&text))
            })
            .collect()
    }

    /// Check a set of client files against [`ALLOWED`] in both directions.
    fn check(clients: &[String]) -> Vec<String> {
        let unexpected = clients
            .iter()
            .filter(|file| !ALLOWED.iter().any(|(allowed, _)| allowed == file))
            .map(|file| {
                format!(
                    "`{file}` pipes a child's stdin and names an engine envelope, which is an \
                     engine stdio client. Use `rocm_engine_protocol::run_engine_stdio` (in \
                     {CLIENT}), or add `{file}` to ALLOWED in \
                     xtask/src/engine_stdio_client_contract.rs with the reason a second client \
                     is needed"
                )
            });
        let stale = ALLOWED
            .iter()
            .filter(|(allowed, _)| !clients.iter().any(|file| file == allowed))
            .map(|(allowed, _)| {
                format!(
                    "ALLOWED entry `{allowed}` is stale: it no longer pipes a child's stdin and \
                     names an engine envelope. Remove it, or the exemption outlives its reason"
                )
            });
        unexpected.chain(stale).collect()
    }

    #[test]
    fn there_is_only_one_engine_stdio_client() {
        let violations = check(&client_files(&repo_root()));
        assert!(
            violations.is_empty(),
            "rocm and rocmd share one engine stdio client; a second copy can drift from it \
             (ROCMAI-875):\n{}",
            violations.join("\n")
        );
    }

    #[test]
    fn every_allowed_client_carries_a_reason() {
        for (file, reason) in ALLOWED {
            assert!(
                reason.len() > 40,
                "ALLOWED entry `{file}` has no real justification: {reason:?}"
            );
        }
    }

    /// The shape the two replaced copies had: spawn with a piped stdin, write an
    /// envelope to it, read the answer back.
    const OLD_STYLE_COPY: &str = "
fn engine_request(engine_binary: &Path, envelope: &EngineRequestEnvelope) -> Result<()> {
    let mut command = ProcessCommand::new(engine_binary);
    command
        .arg(\"__engine-stdio\")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
    let mut child = command.spawn()?;
    serde_json::to_writer(child.stdin.as_mut().unwrap(), envelope)?;
    let output = child.wait_with_output()?;
    let _answer: EngineResponseEnvelope = serde_json::from_slice(&output.stdout)?;
    Ok(())
}
";

    #[test]
    fn a_pasted_back_copy_of_the_old_client_is_recognised_and_reported() {
        assert!(is_a_client(OLD_STYLE_COPY));
        let mut clients: Vec<String> = ALLOWED.iter().map(|(file, _)| (*file).to_owned()).collect();
        clients.push("apps/rocmd/src/engine_ipc.rs".to_owned());
        let violations = check(&clients);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].contains("apps/rocmd/src/engine_ipc.rs"));
        assert!(violations[0].contains("run_engine_stdio"));
    }

    #[test]
    fn a_caller_of_the_shared_client_is_not_a_client() {
        let caller = "
fn engine_request(engine_binary: &Path, envelope: &EngineRequestEnvelope) -> Result<R> {
    decode_engine_response(run_engine_stdio(engine_binary, [\"stdio\"], &options, envelope)?)
}
";
        assert!(!is_a_client(caller));
        // Piping stdin for something that is not an engine request is fine.
        let unrelated = "let child = Command::new(\"tar\").stdin(Stdio::piped()).spawn()?;";
        assert!(!is_a_client(unrelated));
        // Naming the envelope without a child is fine too (the serving side).
        let serving = "fn handle(envelope: EngineRequestEnvelope) -> EngineResponseEnvelope {}";
        assert!(!is_a_client(serving));
    }

    #[test]
    fn a_mention_in_a_comment_is_not_a_client() {
        let prose = "
// builds an EngineRequestEnvelope and pipes it with .stdin(Stdio::piped())
/// see EngineResponseEnvelope; the child gets `.stdin(Stdio::piped())`
fn nothing() {}
";
        assert!(!is_a_client(prose));
    }

    #[test]
    fn the_client_that_is_allowed_stops_being_allowed_if_it_stops_matching() {
        let violations = check(&[]);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("is stale"));
        assert!(violations[0].contains(CLIENT));
    }

    #[test]
    fn the_self_exclusion_names_a_file_that_exists_and_would_otherwise_be_reported() {
        // A rename would leave SELF_PATH pointing at nothing, and the exclusion
        // would silently stop applying.
        let root = repo_root();
        let text = std::fs::read_to_string(root.join(SELF_PATH))
            .unwrap_or_else(|e| panic!("SELF_PATH `{SELF_PATH}` does not name a real file: {e}"));
        assert!(
            text.contains("OLD_STYLE_COPY"),
            "`{SELF_PATH}` no longer holds the fixture the exclusion exists for"
        );
    }

    #[test]
    fn the_walk_reaches_the_client_and_both_callers() {
        // Without this the contract could pass by scanning nothing.
        let files = workspace_rust_sources(&repo_root());
        for expected in [
            CLIENT,
            "apps/rocm/src/engine_ipc.rs",
            "apps/rocmd/src/engine_ipc.rs",
            "xtask/src/engine_stdio_client_contract.rs",
        ] {
            assert!(
                files.iter().any(|file| file == expected),
                "the source walk missed `{expected}`, so the contract is scanning less than \
                 the tree"
            );
        }
    }
}
