// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Whether this host's glibc and OpenSSL are new enough for the pinned Linux
//! `lemond`, checked before anything is downloaded and before `lemond` starts.
//!
//! Without this check, a `lemond` built against a newer glibc or OpenSSL than
//! the host has exits at once with a dynamic-loader error ("version
//! `GLIBC_2.43' not found"), and the user sees only that the Lemonade server
//! did not become ready. The versions the pinned build needs are recorded next
//! to its archive digest ([`crate::install::LINUX_LEMOND_NEEDS`]); the host's
//! are read from the libraries the dynamic loader would hand `lemond`, as the
//! newest `GLIBC_*` / `OPENSSL_*` version node each one defines, which is the
//! same thing the loader compares.
//!
//! Each library is found through the `ldconfig` cache, with one of three
//! results ([`HostLibrary`]):
//!
//! - a version, read from the library the cache lists;
//! - missing: the cache was read and lists `libc.so.6`, but not this library's
//!   soname. A host with only OpenSSL 1.1 has no `libssl.so.3`, so this is how
//!   such a host reads, and it refuses, since `lemond` cannot load at all;
//! - unknown: no cache could be read, the cache lists not even `libc.so.6`, or
//!   the library listed did not parse. Unknown never refuses: `lemond`'s own
//!   loader error still reaches the startup log tail then, and a false refusal
//!   would block a host that runs it fine.

use std::cmp::Ordering;
use std::fmt;
#[cfg(target_os = "linux")]
use std::sync::OnceLock;

use anyhow::{Result, bail};

/// A dotted library version (`2.38`, `3.0.0`), compared numerically with
/// missing components read as zero, so `3.0` and `3.0.0` are equal.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LibVersion {
    parts: [u32; 4],
    len: usize,
}

impl LibVersion {
    /// Parse a dotted numeric version of one to four components.
    pub(crate) fn parse(text: &str) -> Option<Self> {
        let mut parts = [0_u32; 4];
        let mut len = 0;
        for piece in text.split('.') {
            if len == parts.len() || piece.is_empty() || !piece.bytes().all(|b| b.is_ascii_digit())
            {
                return None;
            }
            parts[len] = piece.parse().ok()?;
            len += 1;
        }
        Some(Self { parts, len })
    }
}

impl PartialEq for LibVersion {
    fn eq(&self, other: &Self) -> bool {
        self.parts == other.parts
    }
}

impl Eq for LibVersion {}

impl PartialOrd for LibVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LibVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        self.parts.cmp(&other.parts)
    }
}

impl fmt::Display for LibVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, part) in self.parts[..self.len].iter().enumerate() {
            if index > 0 {
                f.write_str(".")?;
            }
            write!(f, "{part}")?;
        }
        Ok(())
    }
}

/// What a pinned Linux `lemond` needs from the host: the newest `GLIBC_*` and
/// `OPENSSL_*` symbol versions it links against, as `objdump -T lemond` lists
/// them, for the archive with digest `archive_sha256`.
pub(crate) struct LemondHostNeeds {
    /// The Linux embeddable archive these were read from. A unit test holds it
    /// equal to the pinned digest, so a bump cannot keep the old needs.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read only by the test that ties it to the pin")
    )]
    pub(crate) archive_sha256: &'static str,
    pub(crate) glibc: &'static str,
    pub(crate) openssl: &'static str,
}

/// What the probe found for one library.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum HostLibrary {
    /// Not read: no loader cache, a cache that lists not even `libc.so.6`, or a
    /// library that did not parse. Never refuses.
    #[default]
    Unknown,
    /// The cache was read and lists `libc.so.6`, but not this library's soname.
    Missing,
    /// The newest version node the library defines.
    Version(LibVersion),
}

/// The host's glibc and OpenSSL, as the probe found them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostLibraries {
    pub(crate) glibc: HostLibrary,
    pub(crate) openssl: HostLibrary,
}

/// The soname `lemond` loads for glibc and for OpenSSL, and the prefix of the
/// version nodes it defines.
const GLIBC_SONAME: &str = "libc.so.6";
const OPENSSL_SONAME: &str = "libssl.so.3";

/// Why this host cannot run the pinned `lemond`, or `None` when it can or when
/// nothing short of the needs was found.
///
/// Names each library that falls short with the version needed and what the
/// host has: its version, or that the loader cache has no library of that
/// soname. So the message alone says what to upgrade.
pub(crate) fn host_shortfall(
    lemonade_version: &str,
    needs: &LemondHostNeeds,
    host: &HostLibraries,
) -> Option<String> {
    let mut short = Vec::new();
    for (name, soname, needed, have) in [
        ("glibc", GLIBC_SONAME, needs.glibc, host.glibc),
        ("OpenSSL", OPENSSL_SONAME, needs.openssl, host.openssl),
    ] {
        let needed = LibVersion::parse(needed)
            .unwrap_or_else(|| panic!("pinned {name} need {needed:?} is not a version"));
        match have {
            HostLibrary::Version(have) if have < needed => short.push(format!(
                "{name} {needed} or newer (this host has {name} {have})"
            )),
            HostLibrary::Missing => short.push(format!(
                "{name} {needed} or newer (this host's loader cache has no {soname})"
            )),
            HostLibrary::Version(_) | HostLibrary::Unknown => {}
        }
    }
    if short.is_empty() {
        return None;
    }
    Some(format!(
        "this host's libraries are too old for the Lemonade {lemonade_version} server: it needs {}",
        short.join(" and ")
    ))
}

/// Fail with [`host_shortfall`]'s message plus `outcome`, which says what the
/// refusal left undone ("Nothing was downloaded.").
pub(crate) fn refuse_if_host_too_old(
    lemonade_version: &str,
    needs: &LemondHostNeeds,
    host: &HostLibraries,
    outcome: &str,
) -> Result<()> {
    if let Some(shortfall) = host_shortfall(lemonade_version, needs, host) {
        bail!("{shortfall}. {outcome}");
    }
    Ok(())
}

/// The newest numeric version among `names` that start with `prefix`
/// (`GLIBC_2.38` with prefix `GLIBC_`). Names such as `GLIBC_PRIVATE` or the
/// library's own soname entry are skipped.
pub(crate) fn newest_version_node<'a>(
    names: impl IntoIterator<Item = &'a [u8]>,
    prefix: &str,
) -> Option<LibVersion> {
    names
        .into_iter()
        .filter_map(|name| std::str::from_utf8(name).ok())
        .filter_map(|name| name.strip_prefix(prefix))
        .filter_map(LibVersion::parse)
        .max()
}

/// The path `ldconfig -p` gives for `soname` on x86-64, the only Linux
/// architecture the embeddable is installed on.
pub(crate) fn x86_64_library_path<'a>(ldconfig_cache: &'a str, soname: &str) -> Option<&'a str> {
    ldconfig_cache.lines().find_map(|line| {
        let (left, path) = line.split_once("=>")?;
        let left = left.trim();
        let (name, flags) = left.split_once(' ')?;
        (name == soname && flags.contains("x86-64")).then(|| path.trim())
    })
}

/// Read the host's libraries from `ldconfig -p`'s text (`None` when it could not
/// be run), with `read_version(path, prefix)` reading one library's newest
/// version node. Pure apart from `read_version`, so a test plants the cache.
pub(crate) fn host_libraries_from_cache(
    cache: Option<&str>,
    read_version: impl Fn(&str, &str) -> Option<LibVersion>,
) -> HostLibraries {
    let Some(cache) = cache else {
        return HostLibraries::default();
    };
    // A cache that lists not even the C library says nothing about the others.
    let cache_lists_libc = x86_64_library_path(cache, GLIBC_SONAME).is_some();
    let probe = |soname: &str, prefix: &str| match x86_64_library_path(cache, soname) {
        Some(path) => read_version(path, prefix).map_or(HostLibrary::Unknown, HostLibrary::Version),
        None if cache_lists_libc => HostLibrary::Missing,
        None => HostLibrary::Unknown,
    };
    HostLibraries {
        glibc: probe(GLIBC_SONAME, "GLIBC_"),
        openssl: probe(OPENSSL_SONAME, "OPENSSL_"),
    }
}

/// The version node names an ELF shared library defines (`SHT_GNU_VERDEF`).
#[cfg(target_os = "linux")]
fn defined_version_names(bytes: &[u8]) -> Option<Vec<&[u8]>> {
    use object::read::elf::ElfFile64;

    let file = ElfFile64::<object::Endianness>::parse(bytes).ok()?;
    let endian = file.endian();
    let sections = file.elf_section_table();
    let (mut defs, link) = sections.gnu_verdef(endian, bytes).ok()??;
    let strings = sections.strings(endian, bytes, link).ok()?;
    let mut names = Vec::new();
    while let Some((_, mut aux)) = defs.next().ok()? {
        if let Some(first) = aux.next().ok()? {
            names.push(first.name(endian, strings).ok()?);
        }
    }
    Some(names)
}

/// The newest `prefix` version node the library at `path` defines.
#[cfg(target_os = "linux")]
fn library_version(path: &str, prefix: &str) -> Option<LibVersion> {
    let bytes = std::fs::read(path).ok()?;
    newest_version_node(defined_version_names(&bytes)?, prefix)
}

/// Debug/test override of what [`probe_host_libraries`] reports, so a
/// black-box scenario reaches the refusal on a host whose libraries are new.
/// Each takes a version, or `missing` for a library the cache does not list.
#[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
const HOST_GLIBC_TEST_ENV: &str = "ROCM_E2E_LEMONADE_HOST_GLIBC";
#[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
const HOST_OPENSSL_TEST_ENV: &str = "ROCM_E2E_LEMONADE_HOST_OPENSSL";

/// This host's glibc and OpenSSL, as the libraries the loader cache lists for
/// `libc.so.6` and `libssl.so.3` define them. Probed once per process: the
/// install and every `lemond` start ask, and the answer does not change.
#[cfg(target_os = "linux")]
pub(crate) fn probe_host_libraries() -> HostLibraries {
    static PROBED: OnceLock<HostLibraries> = OnceLock::new();
    *PROBED.get_or_init(|| {
        #[cfg(feature = "e2e-test-hooks")]
        {
            let read = |key: &str| match std::env::var(key).ok().as_deref() {
                Some("missing") => HostLibrary::Missing,
                Some(value) => {
                    LibVersion::parse(value).map_or(HostLibrary::Unknown, HostLibrary::Version)
                }
                None => HostLibrary::Unknown,
            };
            let hooked = HostLibraries {
                glibc: read(HOST_GLIBC_TEST_ENV),
                openssl: read(HOST_OPENSSL_TEST_ENV),
            };
            if hooked != HostLibraries::default() {
                return hooked;
            }
        }
        host_libraries_from_cache(rocm_core::ldconfig_cache().as_deref(), library_version)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEEDS: LemondHostNeeds = LemondHostNeeds {
        archive_sha256: "",
        glibc: "2.38",
        openssl: "3.0.0",
    };

    fn version(text: &str) -> Option<LibVersion> {
        Some(LibVersion::parse(text).unwrap())
    }

    /// One library as a test writes it: a version, `missing`, or `None` for
    /// unknown.
    fn library(text: Option<&str>) -> HostLibrary {
        match text {
            None => HostLibrary::Unknown,
            Some("missing") => HostLibrary::Missing,
            Some(text) => HostLibrary::Version(LibVersion::parse(text).unwrap()),
        }
    }

    fn host(glibc: Option<&str>, openssl: Option<&str>) -> HostLibraries {
        HostLibraries {
            glibc: library(glibc),
            openssl: library(openssl),
        }
    }

    #[test]
    fn versions_compare_numerically_and_ignore_trailing_zeros() {
        assert!(version("2.9") < version("2.38"));
        assert!(version("2.39") < version("2.43"));
        assert!(version("3.0.13") < version("3.2.0"));
        assert_eq!(version("3.0"), version("3.0.0"));
        assert_eq!(LibVersion::parse("2.2.5").unwrap().to_string(), "2.2.5");
        for bad in ["", "2.", ".2", "2.x", "PRIVATE", "1.2.3.4.5", "-1"] {
            assert!(LibVersion::parse(bad).is_none(), "{bad:?} parsed");
        }
    }

    /// The states the check can be in, one row each, as the probe can produce
    /// them: every library new enough (Ubuntu 24.04 against the 2026.44.1
    /// needs, and exactly at the need); glibc too old (Ubuntu 22.04); OpenSSL 3
    /// missing with glibc too old (Ubuntu 20.04, OpenSSL 1.1 only); OpenSSL 3
    /// missing alone; and unknown versions, which never refuse.
    #[test]
    fn shortfall_names_each_library_that_falls_short_and_what_the_host_has() {
        let rows: [(HostLibraries, Option<&str>); 7] = [
            (host(Some("2.39"), Some("3.0.13")), None),
            (host(Some("2.38"), Some("3.0.0")), None),
            (
                host(Some("2.35"), Some("3.0.2")),
                Some("it needs glibc 2.38 or newer (this host has glibc 2.35)"),
            ),
            (
                host(Some("2.31"), Some("missing")),
                Some(
                    "it needs glibc 2.38 or newer (this host has glibc 2.31) and \
                     OpenSSL 3.0.0 or newer (this host's loader cache has no libssl.so.3)",
                ),
            ),
            (
                host(Some("2.39"), Some("missing")),
                Some(
                    "it needs OpenSSL 3.0.0 or newer (this host's loader cache has no libssl.so.3)",
                ),
            ),
            (host(None, Some("3.0.13")), None),
            (host(None, None), None),
        ];
        for (host, expected) in rows {
            let message = host_shortfall("9.9.9", &NEEDS, &host);
            match expected {
                None => assert_eq!(message, None, "{host:?}"),
                Some(expected) => {
                    let message = message.unwrap_or_else(|| panic!("no refusal for {host:?}"));
                    assert!(
                        message.starts_with(
                            "this host's libraries are too old for the Lemonade 9.9.9 server: "
                        ),
                        "{message}"
                    );
                    assert!(message.ends_with(expected), "{message}");
                }
            }
        }
    }

    /// An OpenSSL 3 older than a pin needs is reported by version: no pin needs
    /// that today, but 2026.45.3 needed 3.2.0, which Ubuntu 24.04's 3.0.13 is not.
    #[test]
    fn an_openssl_3_older_than_the_need_is_named_by_version() {
        let needs = LemondHostNeeds {
            archive_sha256: "",
            glibc: "2.43",
            openssl: "3.2.0",
        };
        let message = host_shortfall("9.9.9", &needs, &host(Some("2.39"), Some("3.0.13"))).unwrap();
        assert!(
            message.ends_with(
                "it needs glibc 2.43 or newer (this host has glibc 2.39) and \
                 OpenSSL 3.2.0 or newer (this host has OpenSSL 3.0.13)"
            ),
            "{message}"
        );
    }

    #[test]
    fn refusal_appends_what_it_left_undone_and_passes_a_new_enough_host() {
        let error = refuse_if_host_too_old(
            "9.9.9",
            &NEEDS,
            &host(Some("2.35"), None),
            "Nothing was downloaded.",
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.ends_with("(this host has glibc 2.35). Nothing was downloaded."),
            "{error}"
        );
        refuse_if_host_too_old("9.9.9", &NEEDS, &host(Some("2.39"), Some("3.0.13")), "x").unwrap();
    }

    const LIBC_LINE: &str =
        "\tlibc.so.6 (libc6,x86-64, OS ABI: Linux 3.2.0) => /lib/x86_64-linux-gnu/libc.so.6\n";
    const LIBSSL_LINE: &str = "\tlibssl.so.3 (libc6,x86-64) => /lib/x86_64-linux-gnu/libssl.so.3\n";
    const LIBSSL11_LINE: &str =
        "\tlibssl.so.1.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libssl.so.1.1\n";

    /// A reader standing in for the ELF read: glibc 2.31 and OpenSSL 3.0.13,
    /// or `None` for a library that does not parse.
    fn planted_reader(unparseable: &'static str) -> impl Fn(&str, &str) -> Option<LibVersion> {
        move |path, _prefix| {
            if path.ends_with(unparseable) {
                None
            } else if path.ends_with("libc.so.6") {
                LibVersion::parse("2.31")
            } else {
                LibVersion::parse("3.0.13")
            }
        }
    }

    /// The probe over a planted `ldconfig -p`, one row per state: an OpenSSL
    /// 1.1-only host reads OpenSSL as missing and is refused; a host with both
    /// reads versions; and an unreadable cache, a cache that lists no C library,
    /// and a listed library that does not parse all read unknown.
    #[test]
    fn a_read_cache_without_libssl_3_reports_openssl_missing_and_refuses() {
        let header = "3 libs found in cache `/etc/ld.so.cache'\n";
        let openssl_1_1_only = format!("{header}{LIBC_LINE}{LIBSSL11_LINE}");
        let probed = host_libraries_from_cache(Some(&openssl_1_1_only), planted_reader("-"));
        assert_eq!(probed, host(Some("2.31"), Some("missing")));
        let message = host_shortfall("9.9.9", &NEEDS, &probed).expect("refused");
        assert!(
            message
                .contains("OpenSSL 3.0.0 or newer (this host's loader cache has no libssl.so.3)"),
            "{message}"
        );

        let both = format!("{header}{LIBC_LINE}{LIBSSL_LINE}");
        assert_eq!(
            host_libraries_from_cache(Some(&both), planted_reader("-")),
            host(Some("2.31"), Some("3.0.13"))
        );
        assert_eq!(
            host_libraries_from_cache(Some(&both), planted_reader("libssl.so.3")),
            host(Some("2.31"), None)
        );
        assert_eq!(
            host_libraries_from_cache(None, planted_reader("-")),
            HostLibraries::default()
        );
        let no_libc = format!("{header}{LIBSSL11_LINE}");
        assert_eq!(
            host_libraries_from_cache(Some(&no_libc), planted_reader("-")),
            HostLibraries::default()
        );
    }

    #[test]
    fn newest_version_node_skips_non_numeric_and_other_prefixes() {
        let names: [&[u8]; 6] = [
            b"libc.so.6",
            b"GLIBC_2.2.5",
            b"GLIBC_2.39",
            b"GLIBC_2.4",
            b"GLIBC_PRIVATE",
            b"OPENSSL_3.2.0",
        ];
        assert_eq!(newest_version_node(names, "GLIBC_"), version("2.39"));
        assert_eq!(newest_version_node(names, "OPENSSL_"), version("3.2.0"));
        assert_eq!(newest_version_node(names, "GLIBCXX_"), None);
    }

    #[test]
    fn library_path_is_the_x86_64_entry_for_exactly_that_soname() {
        let cache = "4 libs found in cache `/etc/ld.so.cache'\n\
             \tlibssl.so.3 (libc6) => /usr/lib/i386-linux-gnu/libssl.so.3\n\
             \tlibssl.so.3 (libc6,x86-64) => /lib/x86_64-linux-gnu/libssl.so.3\n\
             \tlibssl.so.30 (libc6,x86-64) => /opt/other/libssl.so.30\n\
             \tlibc.so.6 (libc6,x86-64, OS ABI: Linux 3.2.0) => /lib/x86_64-linux-gnu/libc.so.6\n";
        assert_eq!(
            x86_64_library_path(cache, "libssl.so.3"),
            Some("/lib/x86_64-linux-gnu/libssl.so.3")
        );
        assert_eq!(
            x86_64_library_path(cache, "libc.so.6"),
            Some("/lib/x86_64-linux-gnu/libc.so.6")
        );
        assert_eq!(x86_64_library_path(cache, "libcrypto.so.3"), None);
    }

    /// The ELF reader against real libraries: this host's glibc defines version
    /// nodes from `GLIBC_2.2.5` up, so the newest is at least 2.17, and an
    /// OpenSSL 3 `libssl.so.3` defines `OPENSSL_3.0.0` at least.
    #[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
    #[test]
    fn the_running_hosts_glibc_defines_a_readable_version() {
        let cache = rocm_core::ldconfig_cache().expect("ldconfig -p");
        let path = x86_64_library_path(&cache, "libc.so.6").expect("libc.so.6 listed");
        let found = library_version(path, "GLIBC_").expect("GLIBC_ version nodes");
        assert!(found >= LibVersion::parse("2.17").unwrap(), "{found}");
        // OpenSSL 3 is optional on a test host; where the cache lists it, its
        // own version nodes start at `OPENSSL_3.0.0`.
        if let Some(path) = x86_64_library_path(&cache, "libssl.so.3") {
            let found = library_version(path, "OPENSSL_").expect("OPENSSL_ version nodes");
            assert!(found >= LibVersion::parse("3.0.0").unwrap(), "{found}");
        }
    }
}
