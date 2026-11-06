// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Static contract test: only a pre-flighted call site may resolve the
//! `amd-smi` binary.
//!
//! Structural guard for EAI-8457. The `/dev/kfd` pre-flight exists because
//! `amd-smi` can park in uninterruptible sleep (D-state) against a device whose
//! `open` never returns, where no signal and therefore no caller timeout
//! reaches it. A gate like that protects nothing unless *every* launch site
//! goes through it, and before this file nothing in the tree failed if one
//! stopped: `apps/rocm/src/gpu_policy.rs`'s two probes could revert to
//! `Command::new(rocm_core::resolve_amd_smi_binary())` and the whole suite
//! would stay green, because the regression is the *absence* of a call, which
//! no behavioural test can observe.
//!
//! So the guard is on the ingredient rather than the behaviour. A launch needs
//! a binary to launch, and [`RESOLVER`] is the only way to find one — the
//! managed ROCm SDK ships `amd-smi` inside the runtime wheel's bin directory
//! rather than on `PATH`, so a hand-rolled `Command::new("amd-smi")` finds
//! nothing on the hosts this product targets. Pinning who may *call* the
//! resolver therefore pins who may launch, and a new launch site has to either
//! route through `rocm_core::amd_smi_json` (which pre-flights) or add itself to
//! [`ALLOWED_CALLERS`] in a reviewed diff that says why it is safe. That review
//! is the entire point, exactly as it is for
//! [`crate::crate_edges::ALLOWLIST`].
//!
//! What this does **not** claim: it is a source-text scan, not a call-graph
//! analysis. It cannot see a resolver call reached through a `fn` pointer, a
//! re-export under a different name, or a path assembled at runtime, and it
//! says nothing about whether an allowlisted caller then does the right thing
//! with what it gets. It closes the cheap, likely regression — somebody adding
//! a third probe and spawning it directly, the way both existing probes were
//! written before EAI-8457 — and nothing more.
//!
//! Bidirectional, like the crate-edge allowlist: an entry with no matching call
//! fails too. A site that stops launching `amd-smi` must lose its entry, or the
//! exemption outlives the reason for it and the next direct spawn lands in an
//! already-blessed file.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    /// The call this guard tracks, matched as a call rather than a mention.
    ///
    /// The trailing `(` is load-bearing in both directions. It keeps prose out
    /// — several modules name `resolve_amd_smi_binary` in a doc comment to say
    /// where the path comes from, and none of them launches anything — and it
    /// keeps the two private helpers out too, since
    /// `resolve_amd_smi_binary_in_registry(` and `resolve_amd_smi_binary_in_home(`
    /// do not contain this string. Those two are `rocm-core`-private and
    /// unreachable from another crate anyway.
    const RESOLVER: &str = "resolve_amd_smi_binary(";

    /// Every file that may call [`RESOLVER`], with the reason it is allowed to.
    ///
    /// Paths are repo-relative and `/`-separated; the comparison normalises
    /// Windows separators so the guard gives the same verdict on every lane.
    const ALLOWED_CALLERS: &[(&str, &str)] = &[
        (
            "crates/rocm-core/src/amd_smi.rs",
            "defines the resolver, and calls it from `amd_smi_json` — the one \
             synchronous launch path, and the only place the `/dev/kfd` pre-flight runs",
        ),
        (
            "apps/rocm/src/dash.rs",
            "resolves the path for the dashboard collector rather than launching anything \
             itself; `rocm-dash-collectors` runs its own pre-flight (`preflight_passes`) on \
             the same device, through the same shared `rocm_core::device_readable_bounded` \
             probe, before it spawns",
        ),
    ];

    fn repo_root() -> PathBuf {
        // CARGO_MANIFEST_DIR is the xtask/ crate dir; its parent is the repo
        // root (same idiom as env_mutation_contract::repo_root).
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    /// Every `.rs` file in the repository, as repo-relative `/`-separated
    /// paths.
    ///
    /// A whole-tree walk rather than a list of source directories, so a crate
    /// added under a new top-level directory is covered without anyone
    /// remembering to extend this. `target/` (generated and vendored code) and
    /// dot-directories (which include the `.claude/worktrees` checkouts of
    /// other branches) are skipped, so the verdict never depends on state
    /// outside the commit under test — the same two exclusions, for the same
    /// two reasons, as `env_mutation_contract`.
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

    /// `line` with any `//` comment tail removed.
    ///
    /// Without this the guard reports prose. A doc comment is free to write
    /// `resolve_amd_smi_binary()` with the parens — the natural way to name a
    /// function in English — and that is a description of a call, not one.
    ///
    /// Deliberately crude: it does not track string literals, so a `//` inside
    /// one truncates the line early and could hide a real call after it. That
    /// direction fails *open* for a pathological line no file in this tree has,
    /// and the alternative (a Rust lexer in a contract test) costs more than
    /// the risk. Block comments are not stripped for the same reason; rustfmt
    /// renders every doc comment in this tree as `///`.
    fn strip_comment(line: &str) -> &str {
        match line.find("//") {
            Some(at) => &line[..at],
            None => line,
        }
    }

    /// This file, which the scan must skip.
    ///
    /// Not an exemption like [`ALLOWED_CALLERS`]: this file spawns nothing. Its
    /// fixtures quote `resolve_amd_smi_binary(` as *data* — a guard that cannot
    /// recognise the thing it guards against is a guard nobody can test — and a
    /// text scan has no way to tell a fixture string from a call. Excluding
    /// exactly one path, by name, is narrower than teaching [`strip_comment`]
    /// about string literals, and leaves no room for a real launch site to hide
    /// behind the same trick.
    const SELF_PATH: &str = "xtask/src/amd_smi_launch_contract.rs";

    /// Files that really call [`RESOLVER`], ignoring mentions in comments and
    /// this file's own fixtures.
    fn files_calling_resolver(root: &Path) -> Vec<String> {
        workspace_rust_sources(root)
            .into_iter()
            .filter(|relative| relative != SELF_PATH)
            .filter(|relative| {
                std::fs::read_to_string(root.join(relative)).is_ok_and(|text| {
                    text.lines()
                        .any(|line| strip_comment(line).contains(RESOLVER))
                })
            })
            .collect()
    }

    /// Check a set of calling files against [`ALLOWED_CALLERS`] in both
    /// directions, returning every violation found.
    fn check(calling: &[String]) -> Vec<String> {
        let unexpected = calling
            .iter()
            .filter(|file| !ALLOWED_CALLERS.iter().any(|(allowed, _)| allowed == file))
            .map(|file| {
                format!(
                    "`{file}` calls `{RESOLVER}..)`, which is how an `amd-smi` launch site is \
                     built. Route the launch through `rocm_core::amd_smi_json`, which runs the \
                     mandatory `/dev/kfd` pre-flight, or add `{file}` to ALLOWED_CALLERS in \
                     xtask/src/amd_smi_launch_contract.rs with the reason the site is safe \
                     without it"
                )
            });
        let stale = ALLOWED_CALLERS
            .iter()
            .filter(|(allowed, _)| !calling.iter().any(|file| file == allowed))
            .map(|(allowed, _)| {
                format!(
                    "ALLOWED_CALLERS entry `{allowed}` is stale: it no longer calls \
                     `{RESOLVER}..)`. Remove it, or the exemption outlives its reason and the \
                     next direct spawn lands in an already-blessed file"
                )
            });
        unexpected.chain(stale).collect()
    }

    #[test]
    fn only_preflighted_sites_resolve_the_amd_smi_binary() {
        let root = repo_root();
        let violations = check(&files_calling_resolver(&root));
        assert!(
            violations.is_empty(),
            "the `/dev/kfd` pre-flight only protects the launch sites that go through it \
             (EAI-8457):\n{}",
            violations.join("\n")
        );
    }

    #[test]
    fn every_allowed_caller_carries_a_reason() {
        // An entry with an empty reason is the failure mode this guard exists
        // to prevent, written into the guard: a red run "fixed" by pasting a
        // path into the list instead of routing the launch through the gate.
        for (file, reason) in ALLOWED_CALLERS {
            assert!(
                reason.len() > 40,
                "ALLOWED_CALLERS entry `{file}` has no real justification: {reason:?}"
            );
        }
    }

    #[test]
    fn a_new_direct_resolver_call_fails() {
        let mut calling: Vec<String> = ALLOWED_CALLERS
            .iter()
            .map(|(file, _)| (*file).to_owned())
            .collect();
        calling.push("apps/rocm/src/gpu_policy.rs".to_owned());
        let violations = check(&calling);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("apps/rocm/src/gpu_policy.rs"));
        assert!(violations[0].contains("amd_smi_json"));
    }

    #[test]
    fn an_allowed_caller_that_stops_calling_fails() {
        let calling: Vec<String> = ALLOWED_CALLERS
            .iter()
            .skip(1)
            .map(|(file, _)| (*file).to_owned())
            .collect();
        let violations = check(&calling);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("is stale"));
        assert!(violations[0].contains(ALLOWED_CALLERS[0].0));
    }

    #[test]
    fn a_mention_in_a_comment_is_not_a_call() {
        // Both shapes real files in this tree use. The second is why
        // `strip_comment` exists at all: parens around the name read as a call
        // to a plain `contains`.
        assert!(
            !strip_comment("    /// (via `rocm_core::resolve_amd_smi_binary`)").contains(RESOLVER)
        );
        assert!(
            !strip_comment("    // none of the paths resolve_amd_smi_binary() searches")
                .contains(RESOLVER)
        );
        assert!(strip_comment("    let binary = resolve_amd_smi_binary();").contains(RESOLVER));
        // Code before a trailing comment still counts.
        assert!(
            strip_comment("    let b = resolve_amd_smi_binary(); // resolved here")
                .contains(RESOLVER)
        );
    }

    #[test]
    fn the_private_resolver_helpers_are_not_matched() {
        // `rocm-core`-private, unreachable from another crate, and each is a
        // lookup in one location rather than the whole resolution. Matching
        // them would make every `rocm-core` test that exercises path
        // resolution a launch site.
        for helper in [
            "    let r = resolve_amd_smi_binary_in_registry(&dir);",
            "    let r = resolve_amd_smi_binary_in_home(Some(dir));",
            "    fn resolve_amd_smi_binary_prefers_home_rocm_venv_path() {",
        ] {
            assert!(!strip_comment(helper).contains(RESOLVER), "{helper}");
        }
    }

    #[test]
    fn the_self_exclusion_names_a_file_that_exists_and_would_otherwise_be_reported() {
        // Two ways `SELF_PATH` could go quietly wrong: a rename leaves it
        // pointing at nothing (so the exclusion silently stops applying, and
        // this contract starts reporting itself again on the next fixture
        // added), or the fixtures are removed and the exclusion is now
        // suppressing a file that would pass anyway -- at which point the
        // exclusion should go, not linger as a hole.
        let root = repo_root();
        let text = std::fs::read_to_string(root.join(SELF_PATH))
            .unwrap_or_else(|e| panic!("SELF_PATH `{SELF_PATH}` does not name a real file: {e}"));
        assert!(
            text.lines()
                .any(|line| strip_comment(line).contains(RESOLVER)),
            "`{SELF_PATH}` no longer quotes `{RESOLVER}..)` outside a comment, so the \
             self-exclusion is suppressing nothing and should be deleted"
        );
    }

    #[test]
    fn the_walk_reaches_every_first_party_crate_directory() {
        // Without this the contract could pass by scanning nothing: a walk
        // that silently stopped at the repo root would report no violations
        // forever. Names a file in each top-level source directory a launch
        // site could plausibly be added to.
        let files = workspace_rust_sources(&repo_root());
        for expected in [
            "apps/rocm/src/gpu_policy.rs",
            "apps/rocmd/src/host_snapshot.rs",
            "crates/rocm-core/src/amd_smi.rs",
            "crates/rocm-dash-collectors/src/amd_smi.rs",
            "crates/rocm-dash-daemon/src/runner.rs",
            "xtask/src/crate_edges.rs",
        ] {
            assert!(
                files.iter().any(|file| file == expected),
                "the source walk missed `{expected}`, so the contract is scanning less than \
                 the tree"
            );
        }
    }
}
