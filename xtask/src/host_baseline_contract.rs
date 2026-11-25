// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Contract tests for `.github/scripts/host-baseline.sh`, the glibc/libstdc++
//! floor the shared GPU preflight asserts on every native Linux GPU lane that
//! runs it.
//!
//! Two things are held here:
//!
//! * the script's verdict, driven for real under `bash` with `getconf` and
//!   `ldconfig` replaced by stubs and the libstdc++ it inspects replaced by a
//!   fixture file carrying `GLIBCXX_*` version strings, so every outcome —
//!   met, short on either value, unreadable — is reached without needing a
//!   host that has it; and
//! * that the glibc/`GLIBCXX` pair it asserts is the product's: the constants
//!   in the script must equal the "glibc X with `GLIBCXX_Y`" sentence in
//!   `README.md`, and each place that restates the pair (`docs/wsl.md`, the
//!   WSL distro check and its fix recipe in `rocm-core`) must agree with it,
//!   so no side can move alone. Only the pair is tied: the Ubuntu release
//!   the same texts name (README.md, `docs/wsl.md`, `WSL_MIN_UBUNTU` in
//!   `rocm-core`) is not compared, so raising the floor also means checking
//!   those by hand.
//!
//! Where the script sits in the preflight, and that a failing baseline stops
//! it before any GPU query, is `workflow_contract.rs`'s to drive.

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

    const SCRIPT: &str = ".github/scripts/host-baseline.sh";

    fn read(rel: &str) -> String {
        let p = repo_root().join(rel);
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("reading {}: {e}", p.display()))
    }

    /// `(glibc, GLIBCXX)` as the script's `MIN_GLIBC=` / `MIN_GLIBCXX=` lines
    /// set them.
    fn script_floor() -> (String, String) {
        let text = read(SCRIPT);
        let constant = |name: &str| {
            let prefix = format!("{name}=");
            let values: Vec<&str> = text
                .lines()
                .filter_map(|l| l.strip_prefix(prefix.as_str()))
                .collect();
            assert_eq!(
                values.len(),
                1,
                "{SCRIPT} must assign {name} exactly once at the start of a line; found {values:?}"
            );
            values[0].trim().to_owned()
        };
        (constant("MIN_GLIBC"), constant("MIN_GLIBCXX"))
    }

    /// The `(glibc, GLIBCXX)` pair `pattern` captures in `rel`, read with
    /// whitespace collapsed so a sentence wrapped across lines still matches.
    fn doc_floor(rel: &str, pattern: &str) -> (String, String) {
        let text = read(rel).split_whitespace().collect::<Vec<_>>().join(" ");
        let re = regex::Regex::new(pattern).expect("floor pattern compiles");
        let found: Vec<(String, String)> = re
            .captures_iter(&text)
            .map(|c| (c[1].to_owned(), c[2].to_owned()))
            .collect();
        assert_eq!(
            found.len(),
            1,
            "expected {rel} to state the glibc/GLIBCXX floor exactly once in the form \
             /{pattern}/; found {found:?}. If the sentence was reworded, update this test \
             with it — it is what ties the CI glibc/GLIBCXX floor to the product's."
        );
        found.into_iter().next().expect("one match")
    }

    /// Files outside README.md that restate the floor as "2.38 /
    /// `GLIBCXX_3.4.32`", each exactly once: the WSL distro check's message
    /// and the comments beside it, and the fix recipe's rationale. Each must
    /// select xtask when it alone changes, or an edit to it skips the test
    /// that reads it: `every_floor_file_selects_the_test_that_reads_it` holds
    /// that.
    const SLASH_FLOOR_RESTATEMENTS: [&str; 4] = [
        "crates/rocm-core/src/diagnose/wsl.rs",
        "crates/rocm-core/src/examine.rs",
        "crates/rocm-core/src/examine/wsl.rs",
        "crates/rocm-core/src/fix/catalog.rs",
    ];

    /// The CI floor is the glibc/`GLIBCXX` half of the product floor, on
    /// purpose: a runner below it cannot run what users are told is supported,
    /// and one held above it would hide a break on the oldest supported host.
    /// The OpenSSL version README.md's floor also names is not checked.
    /// Raising either is a decision, and this makes sure it is taken for both
    /// at once — and that every place that tells a user the glibc/`GLIBCXX`
    /// pair moves with it. The Ubuntu release name those places also carry
    /// is not compared.
    #[test]
    fn the_ci_host_floor_is_the_documented_product_floor() {
        let script = script_floor();
        let readme = doc_floor(
            "README.md",
            r"glibc (\d+\.\d+) with `GLIBCXX_(\d+\.\d+\.\d+)`",
        );
        assert_eq!(
            script, readme,
            "{SCRIPT} asserts (glibc, GLIBCXX_) {script:?} but README.md states the product \
             floor as {readme:?}; the CI runner's glibc/GLIBCXX floor must equal the product's"
        );
        let wsl = doc_floor(
            "docs/wsl.md",
            // The parenthesis may go on after the pair ("; newer Lemonade
            // releases need more"), so it is anchored on the pair's own end.
            r"\(glibc (\d+\.\d+) and `GLIBCXX_(\d+\.\d+\.\d+)`[;)]",
        );
        assert_eq!(
            wsl, readme,
            "docs/wsl.md states the floor as {wsl:?} and README.md as {readme:?}"
        );
        for rel in SLASH_FLOOR_RESTATEMENTS {
            let restated = doc_floor(rel, r"(\d+\.\d+) / `?GLIBCXX_(\d+\.\d+\.\d+)`?");
            assert_eq!(
                restated, readme,
                "{rel} states the floor as {restated:?} and README.md as {readme:?}"
            );
        }
    }

    /// Every file `the_ci_host_floor_is_the_documented_product_floor` reads
    /// runs it when that file alone changes: `cargo xtask affected` (and the
    /// pre-push hook, which selects the same way) must pick xtask for it. Each
    /// gets there through its row of `CROSS_CRATE_READS` (cross_crate_reads.rs):
    /// the Markdown files through `README.md` and `docs/`, the Rust
    /// files through `*.rs`. `every_cross_crate_read_is_in_the_rust_filter`
    /// also holds those rows to CI's `rust` path category.
    #[test]
    fn every_floor_file_selects_the_test_that_reads_it() {
        let graph = crate::affected::load_graph().expect("read the workspace graph");
        for rel in ["README.md", "docs/wsl.md"]
            .into_iter()
            .chain(SLASH_FLOOR_RESTATEMENTS)
        {
            let selection = crate::affected::select_with_readers(&[rel.to_owned()], &graph);
            assert!(
                match &selection {
                    crate::affected::Selection::Workspace => true,
                    crate::affected::Selection::Packages(names) => names.contains("xtask"),
                    crate::affected::Selection::Empty => false,
                },
                "an edit to {rel} alone selects {selection:?}, which does not run xtask"
            );
        }
    }

    /// How the stubbed `getconf GNU_LIBC_VERSION` behaves.
    #[cfg(unix)]
    enum Getconf {
        /// Not on `PATH` at all.
        Missing,
        /// Exits non-zero, as musl's `getconf` does for this variable.
        Fails,
        /// Prints this line and exits 0.
        Prints(&'static str),
    }

    /// Where the stubbed `ldconfig` is, and whether it works.
    #[cfg(unix)]
    enum Ldconfig {
        /// On `PATH`, printing the cache.
        OnPath,
        /// Off `PATH`, in the directory the script falls back to — the
        /// `/usr/sbin` of a non-root `PATH` — printing the cache.
        InFallbackDir,
        /// In the fallback directory but without its execute bit, which is
        /// not an ldconfig the script can run.
        NotExecutableInFallbackDir,
        /// Nowhere the script looks.
        Missing,
        /// On `PATH`, but `ldconfig -p` fails.
        Fails,
    }

    /// What sits at the path the cache names for a `Lib`.
    #[cfg(unix)]
    enum LibFile {
        /// A fixture library carrying `GLIBCXX_*` versions.
        Fixture,
        /// Nothing: a stale cache entry.
        Missing,
        /// A directory, which passes `-r` but which `grep` cannot read.
        Directory,
    }

    /// One `libstdc++.so.6` the stubbed loader cache lists, in order.
    #[cfg(unix)]
    struct Lib {
        /// The ABI tag in the cache line's parentheses, e.g. `libc6,x86-64`.
        abi: &'static str,
        /// The newest `GLIBCXX_3.4.N` the fixture file carries; every older
        /// one is present too, as in a real library, and so is the bare
        /// `GLIBCXX_3.4`. `Some(0)` is a file with only that bare version;
        /// `None` is one with no `GLIBCXX_3.4*` version at all.
        newest: Option<u32>,
        file: LibFile,
    }

    /// One file named in `HOST_BASELINE_OS_RELEASE`, in order.
    #[cfg(unix)]
    enum OsRelease {
        /// Not there.
        Absent,
        /// There, with this `PRETTY_NAME`.
        Named(&'static str),
        /// There, with no `PRETTY_NAME` line.
        Unnamed,
        /// There, but a directory: readable, and not a file `sed` can read.
        Directory,
    }

    #[cfg(unix)]
    struct Case {
        getconf: Getconf,
        ldconfig: Ldconfig,
        libs: Vec<Lib>,
        /// Unrelated x86-64 entries listed after the libstdc++ ones, the bulk
        /// of a real cache.
        filler: usize,
        /// The os-release files handed to the script, first first, or `None`
        /// to leave `HOST_BASELINE_OS_RELEASE` unset so the script reads its
        /// default list, this host's own files.
        os_release: Option<Vec<OsRelease>>,
        /// `HOST_TOOLS` left off `PATH`.
        missing_tools: &'static [&'static str],
    }

    /// Tools the script calls that are not stubbed. `PATH` holds these and the
    /// stubs and nothing else, so "missing" really is missing.
    #[cfg(unix)]
    const HOST_TOOLS: [&str; 6] = ["sed", "head", "tr", "grep", "sort", "tail"];

    #[cfg(unix)]
    fn host_tool(tool: &str) -> PathBuf {
        ["/usr/bin", "/bin"]
            .iter()
            .map(|d| Path::new(d).join(tool))
            .find(|p| p.exists())
            .unwrap_or_else(|| panic!("{tool} on this host"))
    }

    /// The bytes of a fixture libstdc++: NUL-separated version strings, the
    /// way they sit in the real file's `.gnu.version_d` string table, plus
    /// look-alikes that must not count.
    #[cfg(unix)]
    fn libstdcxx_fixture(newest: Option<u32>) -> Vec<u8> {
        let mut bytes = b"\x7fELF\0\0CXXABI_1.3\0GLIBCXX_TUNABLES\0".to_vec();
        if let Some(newest) = newest {
            bytes.extend_from_slice(b"GLIBCXX_3.4\0");
            for n in 1..=newest {
                bytes.extend_from_slice(format!("GLIBCXX_3.4.{n}\0").as_bytes());
            }
        }
        bytes.extend_from_slice(b"CXXABI_1.3.15\0");
        bytes
    }

    /// Run the real script under `bash` against `case`; returns the exit code
    /// and stdout+stderr.
    #[cfg(unix)]
    fn run(case: &Case) -> (i32, String) {
        use rocm_core::fake_executable::write_fake_executable;
        use std::fmt::Write as _;

        let dir = tempfile::tempdir().expect("temp dir");
        let tools = dir.path().join("tools");
        std::fs::create_dir(&tools).expect("tools dir");
        for tool in HOST_TOOLS
            .into_iter()
            .filter(|t| !case.missing_tools.contains(t))
        {
            std::os::unix::fs::symlink(host_tool(tool), tools.join(tool)).expect("link a tool");
        }
        match case.getconf {
            Getconf::Missing => {}
            Getconf::Fails => write_fake_executable(
                &tools.join("getconf"),
                "#!/bin/sh\necho 'getconf: GNU_LIBC_VERSION: unknown variable' >&2\nexit 1\n",
            )
            .expect("plant getconf"),
            Getconf::Prints(line) => {
                write_fake_executable(
                    &tools.join("getconf"),
                    format!("#!/bin/sh\necho '{line}'\n"),
                )
                .expect("plant getconf");
            }
        }
        let mut cache = format!(
            "{} libs found in cache `/etc/ld.so.cache'\n",
            case.libs.len() + 1
        );
        cache.push_str(
            "\tlibc.so.6 (libc6,x86-64, OS ABI: Linux 3.2.0) => /lib/x86_64-linux-gnu/libc.so.6\n",
        );
        for (i, lib) in case.libs.iter().enumerate() {
            let path = dir.path().join(format!("lib{i}/libstdc++.so.6"));
            std::fs::create_dir_all(path.parent().expect("parent")).expect("lib dir");
            match lib.file {
                LibFile::Fixture => {
                    std::fs::write(&path, libstdcxx_fixture(lib.newest))
                        .expect("write fixture lib");
                }
                LibFile::Missing => {}
                LibFile::Directory => std::fs::create_dir(&path).expect("lib as a directory"),
            }
            let _ = writeln!(
                cache,
                "\tlibstdc++.so.6 ({}) => {}",
                lib.abi,
                path.display()
            );
        }
        for n in 0..case.filler {
            let _ = writeln!(
                cache,
                "\tlibfiller{n}.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libfiller{n}.so.1"
            );
        }
        let cache_path = dir.path().join("ld.so.cache.txt");
        std::fs::write(&cache_path, cache).expect("write cache listing");
        // Always handed to the script, so the host's own /usr/sbin/ldconfig
        // can never stand in for a stub this case left out.
        let sbin = dir.path().join("sbin");
        std::fs::create_dir(&sbin).expect("sbin dir");
        let printing = format!(
            "#!/bin/sh\n[ \"$1\" = -p ] || exit 2\nexec /bin/cat '{}'\n",
            cache_path.display()
        );
        match case.ldconfig {
            Ldconfig::OnPath => write_fake_executable(&tools.join("ldconfig"), printing),
            Ldconfig::InFallbackDir => write_fake_executable(&sbin.join("ldconfig"), printing),
            Ldconfig::NotExecutableInFallbackDir => std::fs::write(sbin.join("ldconfig"), printing),
            Ldconfig::Missing => Ok(()),
            Ldconfig::Fails => write_fake_executable(
                &tools.join("ldconfig"),
                "#!/bin/sh\necho 'ldconfig: cannot open cache' >&2\nexit 1\n",
            ),
        }
        .expect("plant ldconfig");
        let mut os_release_paths = Vec::new();
        for (i, entry) in case.os_release.iter().flatten().enumerate() {
            let path = dir.path().join(format!("os-release{i}"));
            let body = match entry {
                OsRelease::Absent => None,
                OsRelease::Named(name) => Some(format!("NAME=x\nPRETTY_NAME=\"{name}\"\nID=x\n")),
                OsRelease::Unnamed => Some("NAME=x\nID=x\n".to_owned()),
                OsRelease::Directory => {
                    std::fs::create_dir(&path).expect("create os-release directory");
                    None
                }
            };
            if let Some(body) = body {
                std::fs::write(&path, body).expect("write os-release");
            }
            os_release_paths.push(path.display().to_string());
        }

        let mut cmd = std::process::Command::new(host_tool("bash"));
        cmd.arg(repo_root().join(SCRIPT))
            .env_clear()
            .env("PATH", &tools)
            .env("HOST_BASELINE_LDCONFIG_DIRS", &sbin);
        if case.os_release.is_some() {
            cmd.env("HOST_BASELINE_OS_RELEASE", os_release_paths.join(" "));
        }
        let out = cmd.output().expect("running host-baseline.sh");
        let mut log = String::from_utf8_lossy(&out.stdout).into_owned();
        log.push_str(&String::from_utf8_lossy(&out.stderr));
        (out.status.code().expect("exited, not signalled"), log)
    }

    #[cfg(unix)]
    fn x86_64(newest: u32) -> Lib {
        Lib {
            abi: "libc6,x86-64",
            newest: Some(newest),
            file: LibFile::Fixture,
        }
    }

    #[cfg(unix)]
    fn host(glibc: &'static str, libs: Vec<Lib>) -> Case {
        Case {
            getconf: Getconf::Prints(glibc),
            ldconfig: Ldconfig::OnPath,
            libs,
            filler: 0,
            os_release: Some(vec![OsRelease::Named("Ubuntu 24.04.1 LTS")]),
            missing_tools: &[],
        }
    }

    #[cfg(unix)]
    const FLOOR: &str = "(floor glibc 2.38, GLIBCXX_3.4.32)";

    #[cfg(unix)]
    fn assert_passes(case: &Case, os: &str, glibc: &str, glibcxx: &str) {
        let (code, log) = run(case);
        assert_eq!(
            log.trim_end(),
            format!("host baseline: {os}, glibc {glibc}, GLIBCXX_{glibcxx} {FLOOR}"),
            "exit {code}"
        );
        assert_eq!(code, 0, "{log}");
    }

    /// Asserts a failure whose reason is exactly `reason`.
    #[cfg(unix)]
    fn assert_fails(case: &Case, os: &str, reason: &str) {
        let (code, log) = run(case);
        assert_eq!(
            log.trim_end(),
            format!(
                "::error::host baseline not met on {os}: {reason} (floor glibc 2.38, \
                 GLIBCXX_3.4.32; see README.md)"
            ),
            "exit {code}"
        );
        assert_eq!(code, 1, "{log}");
    }

    /// The x86-64 fixture's path in the scratch tree is not known up front, so
    /// a shortfall naming it is matched around it.
    #[cfg(unix)]
    fn assert_fails_on_glibcxx(case: &Case, prefix: &str, glibcxx: &str) {
        let (code, log) = run(case);
        let line = log.trim_end();
        let re = regex::Regex::new(&format!(
            r"^::error::host baseline not met on Ubuntu 24\.04\.1 LTS: {}GLIBCXX_{} \(/\S+/lib\d/libstdc\+\+\.so\.6\) is below GLIBCXX_3\.4\.32 \(floor glibc 2\.38, GLIBCXX_3\.4\.32; see README\.md\)$",
            regex::escape(prefix),
            regex::escape(glibcxx),
        ))
        .expect("pattern compiles");
        assert!(re.is_match(line), "exit {code}, logged:\n{log}");
        assert_eq!(code, 1, "{log}");
    }

    #[cfg(unix)]
    #[test]
    fn a_host_exactly_at_the_floor_passes() {
        assert_passes(
            &host("glibc 2.38", vec![x86_64(32)]),
            "Ubuntu 24.04.1 LTS",
            "2.38",
            "3.4.32",
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_host_above_the_floor_passes() {
        assert_passes(
            &host("glibc 2.39", vec![x86_64(33)]),
            "Ubuntu 24.04.1 LTS",
            "2.39",
            "3.4.33",
        );
    }

    /// Ubuntu 22.04's pair: both values short, and both named.
    #[cfg(unix)]
    #[test]
    fn a_host_short_on_both_names_both() {
        assert_fails_on_glibcxx(
            &host("glibc 2.35", vec![x86_64(30)]),
            "glibc 2.35 is below 2.38; ",
            "3.4.30",
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_old_glibc_fails_even_with_a_new_libstdcxx() {
        assert_fails(
            &host("glibc 2.35", vec![x86_64(33)]),
            "Ubuntu 24.04.1 LTS",
            "glibc 2.35 is below 2.38",
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_old_libstdcxx_fails_even_with_a_new_glibc() {
        assert_fails_on_glibcxx(&host("glibc 2.39", vec![x86_64(30)]), "", "3.4.30");
    }

    /// Versions compared as strings get all three of these wrong.
    #[cfg(unix)]
    #[test]
    fn versions_compare_by_component_not_as_strings() {
        // "2.4" > "2.38" as a string.
        assert_fails(
            &host("glibc 2.4", vec![x86_64(33)]),
            "Ubuntu 24.04.1 LTS",
            "glibc 2.4 is below 2.38",
        );
        // "2.100" < "2.38" as a string.
        assert_passes(
            &host("glibc 2.100", vec![x86_64(33)]),
            "Ubuntu 24.04.1 LTS",
            "2.100",
            "3.4.33",
        );
        // "3.4.4" > "3.4.32" as a string, so a string compare passes this.
        assert_fails_on_glibcxx(&host("glibc 2.39", vec![x86_64(4)]), "", "3.4.4");
        // Every 3.4.N up to 33 present: a string sort ranks 3.4.9 newest, and
        // the exact line asserted names 3.4.33.
        assert_passes(
            &host("glibc 2.39", vec![x86_64(33)]),
            "Ubuntu 24.04.1 LTS",
            "2.39",
            "3.4.33",
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_host_without_getconf_fails() {
        let mut case = host("unused", vec![x86_64(33)]);
        case.getconf = Getconf::Missing;
        assert_fails(
            &case,
            "Ubuntu 24.04.1 LTS",
            "getconf is not installed, so the glibc version cannot be read",
        );
    }

    /// musl's `getconf` has no `GNU_LIBC_VERSION`.
    #[cfg(unix)]
    #[test]
    fn a_host_whose_getconf_cannot_report_glibc_fails() {
        let mut case = host("unused", vec![x86_64(33)]);
        case.getconf = Getconf::Fails;
        assert_fails(
            &case,
            "Ubuntu 24.04.1 LTS",
            "getconf GNU_LIBC_VERSION failed (exit 1) — not a glibc host",
        );
    }

    /// The whole line must be "glibc" and a version: a version with
    /// something before or after it is not one this check can trust.
    #[cfg(unix)]
    #[test]
    fn an_unparseable_glibc_version_fails() {
        for line in [
            "glibc two-point-thirty-nine",
            "glibc 2.39 (Debian)",
            "eglibc 2.39",
        ] {
            assert_fails(
                &host(line, vec![x86_64(33)]),
                "Ubuntu 24.04.1 LTS",
                &format!("getconf GNU_LIBC_VERSION printed '{line}', not a glibc version"),
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_host_with_no_libstdcxx_in_the_loader_cache_fails() {
        assert_fails(
            &host("glibc 2.39", vec![]),
            "Ubuntu 24.04.1 LTS",
            "no x86-64 libstdc++.so.6 in the loader cache (ldconfig -p)",
        );
    }

    /// A file the cache names but that carries no version is not a library
    /// this check can judge, so it fails rather than passes.
    #[cfg(unix)]
    #[test]
    fn a_libstdcxx_with_no_glibcxx_version_fails() {
        let mut lib = x86_64(0);
        lib.newest = None;
        let (code, log) = run(&host("glibc 2.39", vec![lib]));
        let re = regex::Regex::new(
            r"^::error::host baseline not met on Ubuntu 24\.04\.1 LTS: /\S+/lib0/libstdc\+\+\.so\.6 carries no GLIBCXX_3\.4 symbol version \(floor glibc 2\.38, GLIBCXX_3\.4\.32; see README\.md\)$",
        )
        .expect("pattern compiles");
        assert!(re.is_match(log.trim_end()), "exit {code}, logged:\n{log}");
        assert_eq!(code, 1, "{log}");
    }

    /// A library whose only version is the bare `GLIBCXX_3.4` is judged on
    /// it, and is below the floor, rather than read as carrying none.
    #[cfg(unix)]
    #[test]
    fn a_libstdcxx_with_only_the_bare_glibcxx_3_4_is_judged_on_it() {
        assert_fails_on_glibcxx(&host("glibc 2.39", vec![x86_64(0)]), "", "3.4");
    }

    /// A ranking that failed is a failed read, not a library with no versions
    /// in it: the script runs under `pipefail` so a broken stage is seen.
    #[cfg(unix)]
    #[test]
    fn a_host_whose_versions_cannot_be_ranked_fails_as_a_read_error() {
        let mut case = host("glibc 2.39", vec![x86_64(33)]);
        case.missing_tools = &["sort"];
        let (code, log) = run(&case);
        let re = regex::Regex::new(
            r"(?m)^::error::host baseline not met on Ubuntu 24\.04\.1 LTS: could not rank the GLIBCXX versions read from /\S+/lib0/libstdc\+\+\.so\.6 \(floor glibc 2\.38, GLIBCXX_3\.4\.32; see README\.md\)$",
        )
        .expect("pattern compiles");
        // bash's own "command not found" follows on stderr; the verdict line
        // is what is pinned.
        assert!(re.is_match(&log), "exit {code}, logged:\n{log}");
        assert_eq!(code, 1, "{log}");
    }

    /// Multilib hosts list the x32 and 32-bit copies too, and often first. The
    /// engine loads the x86-64 one, so that is the one judged — in both
    /// directions.
    #[cfg(unix)]
    #[test]
    fn only_the_x86_64_libstdcxx_is_judged() {
        let x32 = |newest| Lib {
            abi: "libc6,x32",
            newest: Some(newest),
            file: LibFile::Fixture,
        };
        let i386 = |newest| Lib {
            abi: "libc6",
            newest: Some(newest),
            file: LibFile::Fixture,
        };
        assert_passes(
            &host("glibc 2.39", vec![x32(30), x86_64(33), i386(30)]),
            "Ubuntu 24.04.1 LTS",
            "2.39",
            "3.4.33",
        );
        assert_fails_on_glibcxx(
            &host("glibc 2.39", vec![x32(33), x86_64(30), i386(33)]),
            "",
            "3.4.30",
        );
        // Only non-x86-64 copies: nothing the engine could load.
        assert_fails(
            &host("glibc 2.39", vec![x32(33), i386(33)]),
            "Ubuntu 24.04.1 LTS",
            "no x86-64 libstdc++.so.6 in the loader cache (ldconfig -p)",
        );
    }

    /// Some hosts list two x86-64 copies — a toolchain's newer one ahead of
    /// the distribution's, say. The loader takes the first, so that is the
    /// one judged, in both directions.
    #[cfg(unix)]
    #[test]
    fn the_first_x86_64_libstdcxx_in_the_cache_is_judged() {
        assert_passes(
            &host("glibc 2.39", vec![x86_64(33), x86_64(30)]),
            "Ubuntu 24.04.1 LTS",
            "2.39",
            "3.4.33",
        );
        let (code, log) = run(&host("glibc 2.39", vec![x86_64(30), x86_64(33)]));
        assert!(
            log.contains("GLIBCXX_3.4.30 (") && log.contains("/lib0/libstdc++.so.6)"),
            "the first x86-64 entry (lib0, 3.4.30) was not the one judged; exit {code}, \
             logged:\n{log}"
        );
        assert_eq!(code, 1, "{log}");
    }

    /// A real cache lists thousands of libraries. Reading it must not trip
    /// `pipefail` on a SIGPIPE when the libstdc++ entry is found early, which
    /// would end the script with status 141 and no `::error::` line.
    #[cfg(unix)]
    #[test]
    fn a_large_loader_cache_is_read_to_a_verdict() {
        let mut case = host("glibc 2.39", vec![x86_64(33)]);
        case.filler = 5000;
        assert_passes(&case, "Ubuntu 24.04.1 LTS", "2.39", "3.4.33");
    }

    /// A non-root `PATH` often leaves out the sbin directory ldconfig lives
    /// in; the script looks there before concluding it is missing.
    #[cfg(unix)]
    #[test]
    fn ldconfig_off_path_is_found_in_sbin() {
        let mut case = host("glibc 2.39", vec![x86_64(33)]);
        case.ldconfig = Ldconfig::InFallbackDir;
        assert_passes(&case, "Ubuntu 24.04.1 LTS", "2.39", "3.4.33");
    }

    #[cfg(unix)]
    #[test]
    fn a_host_without_ldconfig_fails() {
        let mut case = host("glibc 2.39", vec![x86_64(33)]);
        case.ldconfig = Ldconfig::Missing;
        assert_fails(
            &case,
            "Ubuntu 24.04.1 LTS",
            "ldconfig is not installed, so libstdc++ cannot be located",
        );
    }

    /// A file called ldconfig that cannot be executed is not an ldconfig.
    #[cfg(unix)]
    #[test]
    fn a_non_executable_ldconfig_in_sbin_is_not_used() {
        let mut case = host("glibc 2.39", vec![x86_64(33)]);
        case.ldconfig = Ldconfig::NotExecutableInFallbackDir;
        assert_fails(
            &case,
            "Ubuntu 24.04.1 LTS",
            "ldconfig is not installed, so libstdc++ cannot be located",
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_failing_ldconfig_fails() {
        let mut case = host("glibc 2.39", vec![x86_64(33)]);
        case.ldconfig = Ldconfig::Fails;
        assert_fails(
            &case,
            "Ubuntu 24.04.1 LTS",
            "ldconfig -p failed, so libstdc++ cannot be located",
        );
    }

    /// A stale cache entry names a file that is gone.
    #[cfg(unix)]
    #[test]
    fn a_cache_entry_naming_a_missing_file_fails() {
        let mut lib = x86_64(33);
        lib.file = LibFile::Missing;
        let (code, log) = run(&host("glibc 2.39", vec![lib]));
        let re = regex::Regex::new(
            r"^::error::host baseline not met on Ubuntu 24\.04\.1 LTS: the loader cache names /\S+/lib0/libstdc\+\+\.so\.6, which cannot be read \(floor glibc 2\.38, GLIBCXX_3\.4\.32; see README\.md\)$",
        )
        .expect("pattern compiles");
        assert!(re.is_match(log.trim_end()), "exit {code}, logged:\n{log}");
        assert_eq!(code, 1, "{log}");
    }

    /// A path that passes the readability test but that `grep` still cannot
    /// read is a failed read, not a library with no versions in it.
    #[cfg(unix)]
    #[test]
    fn a_libstdcxx_grep_cannot_read_fails_as_a_read_error() {
        let mut lib = x86_64(33);
        lib.file = LibFile::Directory;
        let (code, log) = run(&host("glibc 2.39", vec![lib]));
        let re = regex::Regex::new(
            r"(?m)^::error::host baseline not met on Ubuntu 24\.04\.1 LTS: grep could not read /\S+/lib0/libstdc\+\+\.so\.6 \(exit 2\) \(floor glibc 2\.38, GLIBCXX_3\.4\.32; see README\.md\)$",
        )
        .expect("pattern compiles");
        // grep's own complaint follows on stderr; the verdict line is what is
        // pinned.
        assert!(re.is_match(&log), "exit {code}, logged:\n{log}");
        assert_eq!(code, 1, "{log}");
    }

    /// The floor is the libraries; the distribution is logged, never gated on.
    #[cfg(unix)]
    #[test]
    fn a_non_ubuntu_host_with_the_libraries_passes() {
        let mut case = host("glibc 2.40", vec![x86_64(33)]);
        case.os_release = Some(vec![OsRelease::Named("Fedora Linux 41 (Server Edition)")]);
        assert_passes(&case, "Fedora Linux 41 (Server Edition)", "2.40", "3.4.33");
    }

    #[cfg(unix)]
    #[test]
    fn a_host_without_os_release_passes_and_says_unknown() {
        let mut case = host("glibc 2.39", vec![x86_64(33)]);
        case.os_release = Some(vec![OsRelease::Absent]);
        assert_passes(&case, "unknown", "2.39", "3.4.33");
    }

    /// Of the files listed, the second is read when the first is absent, and
    /// the first readable one wins outright, named or not, as os-release(5)
    /// has it for /etc/os-release and /usr/lib/os-release.
    #[cfg(unix)]
    #[test]
    fn the_first_readable_os_release_names_the_host() {
        for (files, os) in [
            (
                vec![
                    OsRelease::Absent,
                    OsRelease::Named("Debian GNU/Linux 13 (trixie)"),
                ],
                "Debian GNU/Linux 13 (trixie)",
            ),
            (
                vec![
                    OsRelease::Named("Ubuntu 24.04.1 LTS"),
                    OsRelease::Named("Debian"),
                ],
                "Ubuntu 24.04.1 LTS",
            ),
            (
                vec![OsRelease::Unnamed, OsRelease::Named("Debian")],
                "unknown",
            ),
        ] {
            let mut case = host("glibc 2.39", vec![x86_64(33)]);
            case.os_release = Some(files);
            assert_passes(&case, os, "2.39", "3.4.33");
        }
    }

    /// The OS name is logging only, so an os-release path that is not a
    /// regular file is passed over for the next one, or for "unknown" when
    /// none is left, rather than ending the script: `[[ -r ]]` alone is true
    /// of a directory, and the `sed` that then fails would take the script
    /// down under `set -e` with no `::error::` line and an exit status the
    /// preflight reads as "could not be run".
    #[cfg(unix)]
    #[test]
    fn an_os_release_that_is_a_directory_is_passed_over() {
        for (files, os) in [
            (
                vec![
                    OsRelease::Directory,
                    OsRelease::Named("Debian GNU/Linux 13 (trixie)"),
                ],
                "Debian GNU/Linux 13 (trixie)",
            ),
            (vec![OsRelease::Directory], "unknown"),
        ] {
            let mut case = host("glibc 2.39", vec![x86_64(33)]);
            case.os_release = Some(files);
            assert_passes(&case, os, "2.39", "3.4.33");
        }
    }

    /// The script's default os-release list, written out: /etc/os-release,
    /// then /usr/lib/os-release, the order os-release(5) gives them.
    const DEFAULT_OS_RELEASE: [&str; 2] = ["/etc/os-release", "/usr/lib/os-release"];

    /// The directories the script searches for `ldconfig` when it is off PATH,
    /// written out: the sbin directories a non-root shell leaves out.
    const DEFAULT_LDCONFIG_DIRS: [&str; 2] = ["/usr/sbin", "/sbin"];

    /// Every default the script gives `var` (the text after `${var:-` up to the
    /// closing brace), one entry per occurrence outside comments.
    fn declared_defaults(script: &str, var: &str) -> Vec<String> {
        let opener = format!("${{{var}:-");
        script
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .flat_map(|l| l.split(opener.as_str()).skip(1))
            .map(|rest| {
                rest.split('}')
                    .next()
                    .expect("split yields one item")
                    .to_owned()
            })
            .collect()
    }

    /// The script declares exactly [`DEFAULT_OS_RELEASE`] and
    /// [`DEFAULT_LDCONFIG_DIRS`], each in one place and in that order, so
    /// dropping or reordering an entry of either list fails here. This pins
    /// only what the script declares: that the defaults are used at runtime is
    /// `the_default_os_release_list_names_this_host`'s to show for os-release,
    /// and `preflight_runs_the_real_host_baseline`'s (workflow_contract.rs)
    /// for both, since each leaves its variable unset. That the os-release
    /// list is walked in order, the first readable file winning, is
    /// `the_first_readable_os_release_names_the_host`'s to show with fixtures.
    #[test]
    fn the_script_declares_its_default_search_lists() {
        for (var, want) in [
            ("HOST_BASELINE_OS_RELEASE", DEFAULT_OS_RELEASE),
            ("HOST_BASELINE_LDCONFIG_DIRS", DEFAULT_LDCONFIG_DIRS),
        ] {
            assert_eq!(
                declared_defaults(&read(SCRIPT), var),
                [want.join(" ")],
                "{SCRIPT} must default {var} to {want:?}, in that order, in exactly one place"
            );
        }
    }

    /// With `HOST_BASELINE_OS_RELEASE` unset the script really does fall back
    /// to the default list rather than to nothing: the name it logs is this
    /// host's, read the same way. On a host with /etc/os-release, CI's
    /// included, that run reads only the first entry; which entries the list
    /// holds is `the_script_declares_its_default_search_lists`'s to pin.
    #[cfg(unix)]
    #[test]
    fn the_default_os_release_list_names_this_host() {
        let expected = DEFAULT_OS_RELEASE
            .into_iter()
            .find_map(|f| std::fs::read_to_string(f).ok())
            .and_then(|text| {
                text.lines()
                    .find_map(|l| l.strip_prefix("PRETTY_NAME="))
                    .map(|v| v.replace(['"', '\''], ""))
            })
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "unknown".to_owned());
        let mut case = host("glibc 2.39", vec![x86_64(33)]);
        case.os_release = None;
        assert_passes(&case, &expected, "2.39", "3.4.33");
    }
}
