// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Contract tests for `.github/scripts/host-baseline.sh`, the glibc/libstdc++
//! floor the shared GPU preflight asserts on every self-hosted Linux lane.
//!
//! Two things are held here:
//!
//! * the script's verdict, driven for real under `bash` with `getconf` and
//!   `ldconfig` replaced by stubs and the libstdc++ it inspects replaced by a
//!   fixture file carrying `GLIBCXX_*` version strings, so every outcome —
//!   met, short on either value, unreadable — is reached without needing a
//!   host that has it; and
//! * that the floor it asserts is the product's: the constants in the script
//!   must equal the "glibc X with `GLIBCXX_Y`" sentence in `README.md` and
//!   the same pair in `docs/wsl.md`, so neither side can move alone.
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
             with it — it is what ties the CI floor to the product floor."
        );
        found.into_iter().next().expect("one match")
    }

    /// The CI floor is the product floor, on purpose: a runner below it cannot
    /// run what users are told is supported, and one held above it would hide
    /// a break on the oldest supported host. Raising either is a decision, and
    /// this makes sure it is taken for both at once.
    #[test]
    fn the_ci_host_floor_is_the_documented_product_floor() {
        let script = script_floor();
        let readme = doc_floor(
            "README.md",
            r"glibc (\d+\.\d+) with `GLIBCXX_(\d+\.\d+\.\d+)`",
        );
        let wsl = doc_floor(
            "docs/wsl.md",
            r"\(glibc (\d+\.\d+) and `GLIBCXX_(\d+\.\d+\.\d+)`\)",
        );
        assert_eq!(
            script, readme,
            "{SCRIPT} asserts (glibc, GLIBCXX_) {script:?} but README.md states the product \
             floor as {readme:?}; the CI runner floor must equal the product floor"
        );
        assert_eq!(
            wsl, readme,
            "docs/wsl.md states the floor as {wsl:?} and README.md as {readme:?}"
        );
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
        /// one is present too, as in a real library. 0 means a file with no
        /// `GLIBCXX_3.4*` version at all.
        newest: u32,
        file: LibFile,
    }

    #[cfg(unix)]
    struct Case {
        getconf: Getconf,
        ldconfig: Ldconfig,
        libs: Vec<Lib>,
        /// Unrelated x86-64 entries listed after the libstdc++ ones, the bulk
        /// of a real cache.
        filler: usize,
        /// `PRETTY_NAME` of the os-release file, or `None` for no file.
        os: Option<&'static str>,
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
    fn libstdcxx_fixture(newest: u32) -> Vec<u8> {
        let mut bytes = b"\x7fELF\0\0CXXABI_1.3\0GLIBCXX_TUNABLES\0".to_vec();
        if newest > 0 {
            bytes.extend_from_slice(b"GLIBCXX_3.4\0");
        }
        for n in 1..=newest {
            bytes.extend_from_slice(format!("GLIBCXX_3.4.{n}\0").as_bytes());
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
        for tool in HOST_TOOLS {
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
            Ldconfig::Missing => Ok(()),
            Ldconfig::Fails => write_fake_executable(
                &tools.join("ldconfig"),
                "#!/bin/sh\necho 'ldconfig: cannot open cache' >&2\nexit 1\n",
            ),
        }
        .expect("plant ldconfig");
        let os_release = dir.path().join("os-release");
        if let Some(name) = case.os {
            std::fs::write(
                &os_release,
                format!("NAME=x\nPRETTY_NAME=\"{name}\"\nID=x\n"),
            )
            .expect("write os-release");
        }

        let out = std::process::Command::new(host_tool("bash"))
            .arg(repo_root().join(SCRIPT))
            .env_clear()
            .env("PATH", &tools)
            .env("HOST_BASELINE_OS_RELEASE", &os_release)
            .env("HOST_BASELINE_LDCONFIG_DIRS", &sbin)
            .output()
            .expect("running host-baseline.sh");
        let mut log = String::from_utf8_lossy(&out.stdout).into_owned();
        log.push_str(&String::from_utf8_lossy(&out.stderr));
        (out.status.code().expect("exited, not signalled"), log)
    }

    #[cfg(unix)]
    fn x86_64(newest: u32) -> Lib {
        Lib {
            abi: "libc6,x86-64",
            newest,
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
            os: Some("Ubuntu 24.04.1 LTS"),
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

    #[cfg(unix)]
    #[test]
    fn an_unparseable_glibc_version_fails() {
        assert_fails(
            &host("glibc two-point-thirty-nine", vec![x86_64(33)]),
            "Ubuntu 24.04.1 LTS",
            "getconf GNU_LIBC_VERSION printed 'glibc two-point-thirty-nine', not a glibc version",
        );
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
        let (code, log) = run(&host("glibc 2.39", vec![x86_64(0)]));
        let re = regex::Regex::new(
            r"^::error::host baseline not met on Ubuntu 24\.04\.1 LTS: /\S+/lib0/libstdc\+\+\.so\.6 carries no GLIBCXX_3\.4 symbol version \(floor glibc 2\.38, GLIBCXX_3\.4\.32; see README\.md\)$",
        )
        .expect("pattern compiles");
        assert!(re.is_match(log.trim_end()), "exit {code}, logged:\n{log}");
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
            newest,
            file: LibFile::Fixture,
        };
        let i386 = |newest| Lib {
            abi: "libc6",
            newest,
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
        case.os = Some("Fedora Linux 41 (Server Edition)");
        assert_passes(&case, "Fedora Linux 41 (Server Edition)", "2.40", "3.4.33");
    }

    #[cfg(unix)]
    #[test]
    fn a_host_without_os_release_passes_and_says_unknown() {
        let mut case = host("glibc 2.39", vec![x86_64(33)]);
        case.os = None;
        assert_passes(&case, "unknown", "2.39", "3.4.33");
    }
}
