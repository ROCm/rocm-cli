// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Static contract test: a test that mutates the process environment must
//! serialize itself.
//!
//! Regression guard for EAI-8397. `std::env::set_var`/`remove_var` change state
//! shared by every thread in the process, so two tests touching the same key
//! race — one reads the other's value and fails an assertion that has nothing
//! to do with what it is testing.
//!
//! This is invisible on most of CI. `ci.yml` runs the Linux lanes under
//! `cargo nextest` (a process per test, so the mutation cannot escape), but
//! `windows-build-and-test` runs `cargo test --workspace --all-targets` — all
//! tests as threads in ONE process. So the Windows lane, a required check, was
//! the only place the race could fire, and it read as "your change broke
//! Windows" on branches that never touched the crate. Nothing about the bug is
//! Windows-specific; it is a property of the runner.
//!
//! The repo already had two ways of handling this — `ScopedTestEnv` in
//! `apps/rocm/src/main.rs` and `PROCESS_ENV_TEST_LOCK` in
//! `apps/rocm/src/therock.rs`, both of which take a process-wide lock — and the
//! `fix.rs` tests that flaked were simply the ones that skipped the discipline.
//! This guard makes the omission a build failure rather than an intermittent
//! red on a required check.
//!
//! Best is to not touch the environment at all: pass the value in through a
//! test seam, as `discover_rocm_installs_on_host_in` and
//! `newest_rocm_install_dir_in` do. Where a test must exercise the real
//! env-reading path, holding one of the shared locks above is the accepted
//! alternative and satisfies this guard.

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    fn repo_root() -> PathBuf {
        // CARGO_MANIFEST_DIR is the xtask/ crate dir; its parent is the repo
        // root (same idiom as workflow_contract::repo_root).
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    /// Every `.rs` file in the repository.
    ///
    /// A whole-tree walk rather than a list of source directories, so a crate
    /// added under a new top-level directory is covered without anyone
    /// remembering to extend this. Two kinds of directory are skipped:
    /// `target/`, which holds generated and vendored code we do not own, and
    /// dot-directories, which include the `.claude/worktrees` checkouts of
    /// other branches — scanning either would make the guard's verdict depend
    /// on state outside the commit under test.
    fn workspace_rust_sources() -> Vec<PathBuf> {
        let mut found = Vec::new();
        collect_rust_files(&repo_root(), &mut found);
        assert!(
            !found.is_empty(),
            "found no Rust sources to scan -- the walk is broken, not the tree"
        );
        found.sort();
        found
    }

    fn collect_rust_files(dir: &Path, found: &mut Vec<PathBuf>) {
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
                collect_rust_files(&path, found);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }

    /// The mutating calls that need serializing.
    const MUTATIONS: [&str; 2] = ["env::set_var", "env::remove_var"];

    /// Markers that show a file already serializes its env mutations behind a
    /// process-wide lock. Both are existing repo mechanisms; a file using
    /// either has opted into the discipline this guard enforces.
    const SERIALIZERS: [&str; 2] = ["ScopedTestEnv", "PROCESS_ENV_TEST_LOCK"];

    /// Line numbers (1-based) inside a `#[cfg(test)]` module or a `#[test]`
    /// function that call one of [`MUTATIONS`].
    ///
    /// Brace-depth tracking rather than a whole-file substring match: production
    /// code in the same file may legitimately set an environment variable (it
    /// owns the process), and a file-level match could not tell the two apart.
    fn env_mutations_in_test_code(text: &str) -> Vec<(usize, String)> {
        let mut hits = Vec::new();
        let mut depth: usize = 0;
        // Depth at which the enclosing test block opened, if we are inside one.
        let mut test_block: Option<usize> = None;
        let mut pending_test_attr = false;

        for (index, raw) in text.lines().enumerate() {
            let line = raw.trim();
            let code = line.split("//").next().unwrap_or(line);

            if test_block.is_none() && (line.contains("#[cfg(test)]") || line.contains("#[test]")) {
                pending_test_attr = true;
            }

            if test_block.is_some()
                && let Some(found) = MUTATIONS.iter().find(|needle| code.contains(*needle))
            {
                hits.push((index + 1, (*found).to_owned()));
            }

            let opens = code.matches('{').count();
            let closes = code.matches('}').count();
            if pending_test_attr {
                if opens > 0 {
                    test_block.get_or_insert(depth);
                    pending_test_attr = false;
                } else if code.trim_end().ends_with(';') {
                    // The attribute belonged to a braceless item -- a
                    // `#[cfg(test)] use ...;` import, say. Disarm, or the next
                    // brace anywhere in the file would be mistaken for the
                    // start of a test block and production code below it would
                    // be scanned as test code.
                    pending_test_attr = false;
                }
            }
            // Saturating: a brace inside a string or macro can make a line look
            // unbalanced, and the scan should degrade rather than panic.
            depth = (depth + opens).saturating_sub(closes);
            if let Some(open_depth) = test_block
                && depth <= open_depth
            {
                test_block = None;
            }
        }
        hits
    }

    /// Whether `text` serializes its env mutations behind a process-wide lock.
    ///
    /// File-level on purpose: both existing mechanisms declare the lock once and
    /// take it in each test that needs it, so the guard asks whether the file
    /// has opted in rather than trying to prove each call site is covered.
    fn serializes_env_mutations(text: &str) -> bool {
        SERIALIZERS.iter().any(|marker| text.contains(marker))
    }

    #[test]
    fn a_test_mutating_the_environment_serializes_itself() {
        let root = repo_root();
        let mut offenders: Vec<String> = Vec::new();

        for path in workspace_rust_sources() {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
            if serializes_env_mutations(&text) {
                continue;
            }
            let relative = path.strip_prefix(&root).unwrap_or(&path).display();
            for (line, call) in env_mutations_in_test_code(&text) {
                offenders.push(format!("{relative}:{line}: {call}"));
            }
        }

        assert!(
            offenders.is_empty(),
            "a test that mutates the process environment must serialize itself \
             -- env is shared by every thread, so an unguarded mutation races \
             any test reading the same key and fails only under a threaded \
             runner (the Windows lane, a required check). Prefer passing the \
             value in through a seam (see `newest_rocm_install_dir_in`); \
             otherwise hold `ScopedTestEnv` or `PROCESS_ENV_TEST_LOCK`. \
             Offenders:\n{}",
            offenders.join("\n")
        );
    }

    /// A mutating call, assembled at runtime.
    ///
    /// Spelling `env::set_var` literally in a fixture below would make this
    /// file's own test code match the scan it defines, so the fixtures build
    /// the needle instead of containing it.
    fn mutation_call(kind: &str) -> String {
        format!("unsafe {{ std::env::{kind}(\"KEY\", \"value\") }}")
    }

    #[test]
    fn the_scanner_sees_a_mutation_inside_a_test_module() {
        let call = mutation_call("set_var");
        let source = format!(
            "fn production() {{\n    {call}\n}}\n\
             #[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n        {call}\n    }}\n}}\n"
        );
        let hits = env_mutations_in_test_code(&source);
        assert_eq!(hits.len(), 1, "expected exactly the in-test hit: {hits:?}");
        assert_eq!(hits[0].0, 8, "should flag the line inside the test module");
    }

    #[test]
    fn the_scanner_ignores_production_mutations() {
        // Production code owns the process and may legitimately set a variable;
        // the hazard is specific to tests sharing one process under `cargo test`.
        let source = format!("fn production() {{\n    {}\n}}\n", mutation_call("set_var"));
        assert!(
            env_mutations_in_test_code(&source).is_empty(),
            "a mutation outside test code is not an offense"
        );
    }

    #[test]
    fn the_scanner_stops_flagging_after_the_test_module_closes() {
        let source = format!(
            "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n        let _ = 1;\n    }}\n}}\n\
             fn later_production() {{\n    {}\n}}\n",
            mutation_call("remove_var")
        );
        assert!(
            env_mutations_in_test_code(&source).is_empty(),
            "the test module ended; the later mutation is production code"
        );
    }

    #[test]
    fn the_scanner_ignores_a_cfg_test_attribute_on_a_braceless_item() {
        // `#[cfg(test)] use ...;` gates an import, not a block. Treating it as
        // the opening of a test block would make every brace after it -- the
        // whole rest of the file -- read as test code. Several files in this
        // repo (crates/rocm-dash-tui/src/ui/dock.rs, apps/rocmd/src/lib.rs) do
        // exactly this above production code that sets an env var.
        let source = format!(
            "#[cfg(test)]\nuse foo::Bar;\n\nfn production() {{\n    {}\n}}\n",
            mutation_call("set_var")
        );
        assert!(
            env_mutations_in_test_code(&source).is_empty(),
            "the attribute gated an import; nothing below it is test code"
        );
    }

    #[test]
    fn a_serialized_file_is_exempt_but_an_unguarded_one_is_not() {
        let unguarded = format!(
            "#[cfg(test)]\nmod tests {{\n    #[test]\n    fn t() {{\n        {}\n    }}\n}}\n",
            mutation_call("set_var")
        );
        assert!(
            !serializes_env_mutations(&unguarded),
            "nothing in this source takes a process-wide lock"
        );

        let guarded = unguarded.replace(
            "    #[test]",
            "    static PROCESS_ENV_TEST_LOCK: Mutex<()> = Mutex::new(());\n    #[test]",
        );
        assert!(
            serializes_env_mutations(&guarded),
            "declaring the shared lock opts the file into the discipline"
        );
    }
}
