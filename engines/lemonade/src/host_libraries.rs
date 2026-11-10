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
//! A host version that cannot be read (no `ldconfig`, a library the cache does
//! not list, a file that does not parse) is reported as unknown and never
//! refuses: `lemond`'s own loader error still reaches the startup log tail then,
//! and a false refusal would block a host that runs it fine.

use std::cmp::Ordering;
use std::fmt;

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

/// The host's glibc and OpenSSL versions, each `None` when it could not be read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HostLibraries {
    pub(crate) glibc: Option<LibVersion>,
    pub(crate) openssl: Option<LibVersion>,
}

/// Why this host cannot run the pinned `lemond`, or `None` when it can or when
/// nothing older than needed was found.
///
/// Names each library that is too old with the version needed and the version
/// the host has, so the message alone says what to upgrade.
pub(crate) fn host_shortfall(
    lemonade_version: &str,
    needs: &LemondHostNeeds,
    host: &HostLibraries,
) -> Option<String> {
    let mut short = Vec::new();
    for (name, needed, have) in [
        ("glibc", needs.glibc, host.glibc),
        ("OpenSSL", needs.openssl, host.openssl),
    ] {
        let needed = LibVersion::parse(needed)
            .unwrap_or_else(|| panic!("pinned {name} need {needed:?} is not a version"));
        if let Some(have) = have
            && have < needed
        {
            short.push(format!(
                "{name} {needed} or newer (this host has {name} {have})"
            ));
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
#[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
const HOST_GLIBC_TEST_ENV: &str = "ROCM_E2E_LEMONADE_HOST_GLIBC";
#[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
const HOST_OPENSSL_TEST_ENV: &str = "ROCM_E2E_LEMONADE_HOST_OPENSSL";

/// This host's glibc and OpenSSL, as the libraries the loader cache lists for
/// `libc.so.6` and `libssl.so.3` define them.
#[cfg(target_os = "linux")]
pub(crate) fn probe_host_libraries() -> HostLibraries {
    #[cfg(feature = "e2e-test-hooks")]
    {
        let read = |key: &str| {
            std::env::var(key)
                .ok()
                .and_then(|value| LibVersion::parse(&value))
        };
        let (glibc, openssl) = (read(HOST_GLIBC_TEST_ENV), read(HOST_OPENSSL_TEST_ENV));
        if glibc.is_some() || openssl.is_some() {
            return HostLibraries { glibc, openssl };
        }
    }
    let Some(cache) = rocm_core::ldconfig_cache() else {
        return HostLibraries::default();
    };
    let probe = |soname: &str, prefix: &str| {
        x86_64_library_path(&cache, soname).and_then(|path| library_version(path, prefix))
    };
    HostLibraries {
        glibc: probe("libc.so.6", "GLIBC_"),
        openssl: probe("libssl.so.3", "OPENSSL_"),
    }
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

    fn host(glibc: Option<&str>, openssl: Option<&str>) -> HostLibraries {
        HostLibraries {
            glibc: glibc.and_then(LibVersion::parse),
            openssl: openssl.and_then(LibVersion::parse),
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

    /// The states the check can be in, one row each: every library new enough
    /// (Ubuntu 24.04 against the 2026.44.1 needs, and exactly at the need), one
    /// too old, both too old, and unknown host versions, which never refuse.
    #[test]
    fn shortfall_names_each_library_that_is_too_old_with_both_versions() {
        let rows: [(HostLibraries, Option<&str>); 7] = [
            (host(Some("2.39"), Some("3.0.13")), None),
            (host(Some("2.38"), Some("3.0.0")), None),
            (
                host(Some("2.35"), Some("3.0.2")),
                Some("it needs glibc 2.38 or newer (this host has glibc 2.35)"),
            ),
            (
                host(Some("2.39"), Some("1.1.1")),
                Some("it needs OpenSSL 3.0.0 or newer (this host has OpenSSL 1.1.1)"),
            ),
            (
                host(Some("2.17"), Some("1.1.1")),
                Some(
                    "it needs glibc 2.38 or newer (this host has glibc 2.17) and \
                     OpenSSL 3.0.0 or newer (this host has OpenSSL 1.1.1)",
                ),
            ),
            (
                host(None, Some("1.1.1")),
                Some("it needs OpenSSL 3.0.0 or newer (this host has OpenSSL 1.1.1)"),
            ),
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
