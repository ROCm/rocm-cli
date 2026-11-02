// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The WSL probe: the guest-side WSL2 facts, and the remote probe that reads the
//! same facts from a named WSL distribution when `rocm examine` runs on the Windows
//! host.

use super::{Examination, WSL_MIN_UBUNTU, WSL_PLATFORM_NOTE, WslFacts};
use crate::probe_exec::{MEDIUM, run, run_utf16le, which};
use std::collections::BTreeMap;
use std::time::Duration;

/// Collect the WSL-specific facts the WSL half of the catalog reasons over.
///
/// Reuses [`crate::detect_wsl_summary`] for the plumbing it already probes rather
/// than restating those paths, and adds the facts no existing caller needed: the
/// WSL major version, whether the distro release clears the supported floor, and
/// the Windows host driver version.
pub(super) fn probe_wsl(e: &mut Examination) {
    let summary = crate::detect_wsl_summary();
    let (host_reachable, host_driver_version) = host_driver_fields(crate::detect_wsl_host_driver());
    let rocminfo = which("rocminfo");
    e.wsl = Some(WslFacts {
        version: if crate::is_wsl1_kernel(&e.kernel_release) {
            1
        } else {
            2
        },
        dxg_device: summary.as_ref().is_some_and(|s| s.dxg_device),
        dxcore: summary.as_ref().is_some_and(|s| s.dxcore),
        wsl_lib_dir: crate::host_path("/usr/lib/wsl/lib").is_dir(),
        librocdxg: summary.as_ref().is_some_and(|s| s.librocdxg),
        rocdxg_dids: summary.as_ref().is_some_and(|s| s.rocdxg_dids),
        // `None` when ldconfig itself could not be run, which the summary's
        // bool cannot express -- recover it from the same source the summary used.
        ldconfig_librocdxg: crate::ldconfig_lists_librocdxg(),
        rocminfo,
        rocm_sees_gpu: rocminfo.then(probe_rocminfo_sees_gpu),
        distro_supported: distro_clears_wsl_floor(&e.distro_id, &e.distro_version),
        host_driver_version,
        host_reachable,
        locally_probed: true,
    });
    sync_shared_fields_from_wsl(e);
    e.notes.push(WSL_PLATFORM_NOTE.to_owned());
}

/// Flatten a host-driver probe into the two [`WslFacts`] fields that carry it.
///
/// One place, so the in-guest and host-side probes cannot drift on the point
/// that matters: `None` means the question went unanswered, and `Some("")` means
/// it was answered with "no AMD adapter". Only the second is evidence.
fn host_driver_fields(probe: crate::WslHostDriverProbe) -> (bool, Option<String>) {
    match probe {
        crate::WslHostDriverProbe::Unreachable => (false, None),
        crate::WslHostDriverProbe::NoAmdDisplay => (true, Some(String::new())),
        crate::WslHostDriverProbe::Version(version) => (true, Some(version)),
    }
}

/// Collect the WSL facts from *outside* the distro, over `wsl.exe`.
///
/// Emits `key=value` lines rather than JSON so the guest side needs nothing but
/// a POSIX shell. The Python preflight this replaces injected a Python program
/// and so required `python3` in the distro — on a machine being checked precisely
/// because it is not set up yet.
///
/// `librocdxg` is globbed across `/opt/rocm*` for the same reason the in-guest
/// probe resolves it across installs: a versioned root must not read as missing.
///
/// The distro's `id`/`version` fields fall back from `/etc/os-release` to
/// `/usr/lib/os-release` per `os-release(5)`, matching every other reader in
/// this crate — this is the one that runs inside the distro over `wsl.exe`
/// rather than through [`crate::host_path`], so it needs its own copy of the
/// fallback rather than sharing `hardware_root::read_os_release`.
const WSL_REMOTE_PROBE: &str = r#"
echo "kernel=$(uname -r 2>/dev/null)"
if [ -e /dev/dxg ]; then echo dxg=1; else echo dxg=0; fi
if [ -e /usr/lib/wsl/lib/libdxcore.so ]; then echo dxcore=1; else echo dxcore=0; fi
if [ -d /usr/lib/wsl/lib ]; then echo wsllib=1; else echo wsllib=0; fi
if ls /opt/rocm*/lib/librocdxg.so >/dev/null 2>&1 \
  || ls /usr/local/rocm*/lib/librocdxg.so >/dev/null 2>&1; then echo librocdxg=1; else echo librocdxg=0; fi
if ls /opt/rocm*/share/rocdxg/dids.conf >/dev/null 2>&1 \
  || ls /usr/local/rocm*/share/rocdxg/dids.conf >/dev/null 2>&1; then echo dids=1; else echo dids=0; fi
for ldc in ldconfig /sbin/ldconfig /usr/sbin/ldconfig; do
  if cache=$(command -v "$ldc" >/dev/null 2>&1 && "$ldc" -p 2>/dev/null); then
    case "$cache" in *librocdxg.so*) echo ldconfig=1 ;; *) echo ldconfig=0 ;; esac
    break
  fi
done
if command -v rocminfo >/dev/null 2>&1; then
  echo rocminfo=1
  if rocminfo 2>/dev/null | grep -qi gfx; then echo rocmgfx=1; else echo rocmgfx=0; fi
else
  echo rocminfo=0
fi
try_os_release() {
  if [ -e "$1" ]; then . "$1" 2>/dev/null
  elif [ -e "$2" ]; then . "$2" 2>/dev/null
  fi
}
try_os_release /etc/os-release /usr/lib/os-release
echo "id=${ID}"
echo "version=${VERSION_ID}"
"#;

/// The `try_os_release` shell function embedded in [`WSL_REMOTE_PROBE`]: sources
/// `$1`, falling back to `$2` per `os-release(5)` when `$1` does not exist — an
/// existence check (`[ -e ... ]`), not a readability one, so a permission-denied
/// primary is not silently replaced by a possibly stale fallback.
///
/// Duplicated as its own constant, with the paths left as parameters, so the
/// test below can run this exact function body against temp-directory paths
/// instead of the real `/etc`; [`try_os_release_fn_is_embedded_in_the_wsl_remote_probe`]
/// guards against the two drifting apart.
#[cfg(test)]
const TRY_OS_RELEASE_FN: &str = "try_os_release() {\n  if [ -e \"$1\" ]; then . \"$1\" 2>/dev/null\n  elif [ -e \"$2\" ]; then . \"$2\" 2>/dev/null\n  fi\n}";

/// Parse `wsl.exe -l -q` into distribution names.
///
/// `-q` prints one bare name per line, so the whole line is the name. It is NOT
/// split on whitespace: `wsl --import "My Distro"` is legal, and truncating that
/// to `My` would both fail to match what the user asked for and hand a wrong
/// name to `wsl.exe -d`.
///
/// The header and `*` handling below is for tolerance only — `-q` emits neither,
/// but a caller passing `-l -v` should not silently get its header row back as a
/// distribution.
#[must_use]
pub fn parse_wsl_distro_list(text: &str) -> Vec<String> {
    text.replace('\u{0}', "")
        .lines()
        .filter_map(|raw| {
            let line = raw.trim();
            if line.is_empty() || line.to_uppercase().starts_with("NAME") {
                return None;
            }
            let line = line.strip_prefix('*').map_or(line, str::trim);
            (!line.is_empty()).then(|| line.to_owned())
        })
        .collect()
}

fn parse_remote_flag(fields: &BTreeMap<String, String>, key: &str) -> bool {
    fields.get(key).is_some_and(|value| value == "1")
}

/// A remote flag that can also report that the question went unanswered.
fn parse_remote_tristate(fields: &BTreeMap<String, String>, key: &str) -> Option<bool> {
    match fields.get(key).map(String::as_str) {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }
}

/// Inspect a WSL distribution from the Windows host.
///
/// Returns an [`Examination`] the ordinary catalog can be run against, so the
/// host-side check and the in-distro one share a single set of rules. Nothing
/// needs to be installed in the target distro.
///
/// # Errors
///
/// When `wsl.exe` is unavailable, no distribution matches, or the probe cannot
/// be run inside the selected distribution.
pub fn probe_wsl_distro_from_host(distro: Option<&str>) -> Result<Examination, String> {
    if !which("wsl.exe") {
        return Err(
            "wsl.exe was not found; inspecting a distribution this way only works from the Windows host"
                .to_owned(),
        );
    }
    // `-q` prints names only. `-l -v` adds a header row that is localised, and a
    // header the parser fails to recognise is not skipped -- it is taken for a
    // distribution name.
    let (rc, listed) = run_utf16le("wsl.exe", &["-l", "-q"], MEDIUM);
    if rc != 0 {
        return Err("could not list WSL distributions".to_owned());
    }
    let distros = parse_wsl_distro_list(&listed);
    let selected = select_wsl_distro(distro, &distros)?;

    let (rc, out, _) = run(
        "wsl.exe",
        &["-d", &selected, "--exec", "/bin/sh", "-c", WSL_REMOTE_PROBE],
        Duration::from_secs(30),
    );
    if rc != 0 {
        return Err(format!(
            "could not inspect '{selected}'; the distribution may be stopped or unreachable"
        ));
    }
    let fields: BTreeMap<String, String> = out
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect();
    let (host_reachable, host_driver_version) =
        host_driver_fields(crate::detect_local_windows_host_driver());

    let mut e = Examination {
        os_family: "linux".to_owned(),
        is_wsl: true,
        kernel_release: fields.get("kernel").cloned().unwrap_or_default(),
        distro_id: fields.get("id").cloned().unwrap_or_default(),
        distro_version: fields.get("version").cloned().unwrap_or_default(),
        ..Examination::default()
    };
    let rocminfo = parse_remote_flag(&fields, "rocminfo");
    e.wsl = Some(WslFacts {
        version: if crate::is_wsl1_kernel(&e.kernel_release) {
            1
        } else {
            2
        },
        dxg_device: parse_remote_flag(&fields, "dxg"),
        dxcore: parse_remote_flag(&fields, "dxcore"),
        wsl_lib_dir: parse_remote_flag(&fields, "wsllib"),
        librocdxg: parse_remote_flag(&fields, "librocdxg"),
        rocdxg_dids: parse_remote_flag(&fields, "dids"),
        ldconfig_librocdxg: parse_remote_tristate(&fields, "ldconfig"),
        rocminfo,
        rocm_sees_gpu: rocminfo.then(|| parse_remote_flag(&fields, "rocmgfx")),
        distro_supported: distro_clears_wsl_floor(&e.distro_id, &e.distro_version),
        // Running on the host, the driver is a local question rather than one
        // that has to cross the interop boundary. It can still go unanswered —
        // the inventory query can fail — and that must stay distinguishable from
        // "the host has no AMD adapter", which is a finding.
        host_driver_version,
        host_reachable,
        locally_probed: false,
    });
    sync_shared_fields_from_wsl(&mut e);
    e.status = "wsl".to_owned();
    Ok(e)
}

/// Choose which WSL distribution to inspect, given the requested name (if
/// any) and the list of installed ones.
///
/// Pulled out of [`probe_wsl_distro_from_host`] so this refusal logic -- one
/// of the few genuinely host-independent, blocking checks in the WSL host
/// path -- can be exercised without `wsl.exe`, which the rest of that
/// function requires and which this crate's e2e coverage otherwise never
/// touches on a non-Windows test runner.
fn select_wsl_distro(distro: Option<&str>, distros: &[String]) -> Result<String, String> {
    match distro {
        Some(name) => {
            if !distros.iter().any(|d| d.eq_ignore_ascii_case(name)) {
                return Err(format!(
                    "no WSL distribution named '{name}'; found: {}",
                    distros.join(", ")
                ));
            }
            Ok(name.to_owned())
        }
        None => match distros {
            [] => Err("no WSL distributions were found".to_owned()),
            [only] => Ok(only.clone()),
            many => Err(format!(
                "several WSL distributions are installed; name one with --distro: {}",
                many.join(", ")
            )),
        },
    }
}

/// Whether `rocminfo` enumerates a GPU agent.
///
/// Only the yes/no answer is taken. Parsing the agents into `gpus` is the job of
/// the bare-metal probe, which stays skipped here — this exists so the WSL
/// catalog can tell "the plumbing is complete but no device is reachable" from
/// "the plumbing is incomplete", which is the difference between blaming the
/// Windows host driver and blaming the distro.
fn probe_rocminfo_sees_gpu() -> bool {
    let (rc, out, _) = run("rocminfo", &[], MEDIUM);
    rc == 0 && out.to_lowercase().contains("gfx")
}

/// Mirror the WSL facts onto the shared fields the cross-platform checks read.
///
/// Those checks (PATH, the wheel/ROCm pairing) are valid on WSL and enabled
/// there, but they read fields the bare-metal GPU probe populates — and that
/// probe is skipped here. Left at their defaults they do not read as "unknown",
/// they read as "absent": `rocminfo_present: false` made the PATH check score 50
/// on every WSL host that had ROCm installed.
fn sync_shared_fields_from_wsl(e: &mut Examination) {
    let Some(wsl) = e.wsl.as_ref() else {
        return;
    };
    e.rocminfo_present = wsl.rocminfo;
    e.rocminfo_status = match (wsl.rocminfo, wsl.rocm_sees_gpu) {
        (false, _) => "missing".to_owned(),
        (true, Some(true)) => "ok".to_owned(),
        (true, Some(false)) => "no-agents".to_owned(),
        (true, None) => "unknown".to_owned(),
    };
}

/// Whether the distro release clears the WSL floor in [`WSL_MIN_UBUNTU`].
///
/// `None` means the release could not be read as a `major.minor` pair. That is
/// deliberately not "supported": an unparseable release is not evidence of a good
/// one, and reporting a perfect host on a release nobody could identify is how a
/// user ends up chasing a GPU fault that is really a glibc floor.
///
/// Only Ubuntu carries a floor today, because that is the only distro the WSL
/// path documents. Anything else returns `None` rather than a false verdict.
pub(super) fn distro_clears_wsl_floor(distro_id: &str, distro_version: &str) -> Option<bool> {
    if !distro_id.eq_ignore_ascii_case("ubuntu") {
        return None;
    }
    let (major, minor) = distro_version.split_once('.')?;
    let major: u32 = major.parse().ok()?;
    let minor: u32 = minor.parse().ok()?;
    Some((major, minor) >= WSL_MIN_UBUNTU)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe_exec::decode_utf16le;

    /// Ported from the Python preflight this replaced, whose `wsl.exe` parser
    /// was covered by a self-test that CI ran on both lanes. That coverage has to
    /// land here, or deleting the script quietly drops it.
    #[test]
    fn the_distro_list_survives_the_markers_wsl_puts_around_it() {
        // `-l -q` prints one bare name per line, which is what the probe asks
        // for.
        assert_eq!(
            parse_wsl_distro_list("Ubuntu\nDebian\n"),
            vec!["Ubuntu", "Debian"]
        );
        assert_eq!(
            parse_wsl_distro_list("Ubuntu-24.04\n"),
            vec!["Ubuntu-24.04"]
        );

        // A name may contain spaces: `wsl --import "My Distro"` is legal.
        // Splitting on whitespace truncated it to "My", which then neither
        // matched what the user asked for nor named a real distribution when
        // handed back to `wsl.exe -d`.
        assert_eq!(
            parse_wsl_distro_list("My Distro\nUbuntu\n"),
            vec!["My Distro", "Ubuntu"]
        );

        // The NUL padding of the raw UTF-16 output must not become part of a
        // name, and blank lines are not distributions.
        assert_eq!(
            parse_wsl_distro_list("\0U\0b\0u\0n\0t\0u\0\n\0"),
            vec!["Ubuntu"]
        );
        assert!(parse_wsl_distro_list("").is_empty());
        assert!(parse_wsl_distro_list("\n\n  \n").is_empty());

        // Tolerance only: `-q` emits no header, but a `-l -v` header must never
        // come back as a distribution named "NAME".
        assert!(parse_wsl_distro_list("  NAME   STATE   VERSION\n").is_empty());
    }

    #[test]
    fn a_host_that_could_not_be_queried_is_unknown_not_driverless() {
        use crate::WslHostDriverProbe;

        // The distinction the catalog acts on. An earlier version flattened this
        // to `Option<String>` and defaulted the `None`, so "could not ask the
        // host" arrived as `Some("")` -- which reads as "the host has no AMD
        // adapter" and reported a missing driver on a machine never looked at.
        assert_eq!(
            host_driver_fields(WslHostDriverProbe::Unreachable),
            (false, None),
            "unreachable must stay unknown"
        );
        assert_eq!(
            host_driver_fields(WslHostDriverProbe::NoAmdDisplay),
            (true, Some(String::new())),
            "answered with no adapter is evidence, and is not the same thing"
        );
        assert_eq!(
            host_driver_fields(WslHostDriverProbe::Version("32.0.1".to_owned())),
            (true, Some("32.0.1".to_owned()))
        );
    }

    #[test]
    fn a_named_distro_that_does_not_exist_is_refused() {
        // This is the blocking refusal `probe_wsl_distro_from_host` returns
        // to the caller of `rocm diagnose --distro <name>` -- a plain error,
        // not a `continue-on-error` note the e2e suite can shrug off. It sat
        // behind `wsl.exe` and was untested outside a real Windows+WSL host,
        // which is not a lane this crate's unit tests run on.
        let distros = vec!["Ubuntu".to_owned(), "Debian".to_owned()];
        let err = select_wsl_distro(Some("Fedora"), &distros).expect_err("must be refused");
        assert!(err.contains("Fedora"), "names the distro asked for: {err}");
        assert!(err.contains("Ubuntu"), "lists what does exist: {err}");
        assert!(err.contains("Debian"), "lists what does exist: {err}");
    }

    #[test]
    fn a_named_distro_that_exists_is_selected_case_insensitively() {
        // Matched case-insensitively against the installed list, but the name
        // handed to `wsl.exe -d` afterwards is what the caller typed, not the
        // list's original casing -- pre-existing behavior, preserved as-is by
        // this extraction.
        let distros = vec!["Ubuntu-24.04".to_owned()];
        assert_eq!(
            select_wsl_distro(Some("ubuntu-24.04"), &distros).expect("must be selected"),
            "ubuntu-24.04"
        );
    }

    #[test]
    fn no_distro_named_falls_back_to_the_lone_one_or_refuses_ambiguity() {
        assert_eq!(
            select_wsl_distro(None, &["Ubuntu".to_owned()]).expect("the only one"),
            "Ubuntu"
        );
        assert!(select_wsl_distro(None, &[]).is_err(), "nothing installed");
        let many = vec!["Ubuntu".to_owned(), "Debian".to_owned()];
        let err = select_wsl_distro(None, &many).expect_err("ambiguous without --distro");
        assert!(
            err.contains("--distro"),
            "tells the user how to resolve it: {err}"
        );
    }

    #[test]
    fn sync_shared_fields_from_wsl_copies_the_actual_rocminfo_values() {
        // The fields the cross-platform PATH and wheel/ROCm checks read.
        // Asserting only that they are *set* would still pass if the sync
        // copied the wrong value or a hardcoded default -- assert the actual
        // values reached from each distinct `WslFacts` fixture instead.
        let mut missing = Examination {
            wsl: Some(WslFacts {
                rocminfo: false,
                rocm_sees_gpu: None,
                ..WslFacts::default()
            }),
            ..Examination::default()
        };
        sync_shared_fields_from_wsl(&mut missing);
        assert!(!missing.rocminfo_present);
        assert_eq!(missing.rocminfo_status, "missing");

        let mut healthy = Examination {
            wsl: Some(WslFacts {
                rocminfo: true,
                rocm_sees_gpu: Some(true),
                ..WslFacts::default()
            }),
            ..Examination::default()
        };
        sync_shared_fields_from_wsl(&mut healthy);
        assert!(healthy.rocminfo_present);
        assert_eq!(healthy.rocminfo_status, "ok");

        let mut no_agents = Examination {
            wsl: Some(WslFacts {
                rocminfo: true,
                rocm_sees_gpu: Some(false),
                ..WslFacts::default()
            }),
            ..Examination::default()
        };
        sync_shared_fields_from_wsl(&mut no_agents);
        assert!(no_agents.rocminfo_present);
        assert_eq!(no_agents.rocminfo_status, "no-agents");

        let mut unasked = Examination {
            wsl: Some(WslFacts {
                rocminfo: true,
                rocm_sees_gpu: None,
                ..WslFacts::default()
            }),
            ..Examination::default()
        };
        sync_shared_fields_from_wsl(&mut unasked);
        assert!(unasked.rocminfo_present);
        assert_eq!(unasked.rocminfo_status, "unknown");
    }

    #[test]
    fn probe_wsl_leaves_rocminfo_status_synced_not_at_its_uninitialized_default() {
        // `probe_wsl` never sets `rocminfo_status` itself -- only
        // `sync_shared_fields_from_wsl`, called at its very end, does. If that
        // call were ever removed, this field would stay at
        // `Examination::default()`'s empty string on every host, WSL or not,
        // regardless of whether rocminfo happens to be installed here -- so
        // this holds without depending on real host state, unlike a test that
        // compared against the actual probed rocminfo presence.
        let mut e = Examination::default();
        probe_wsl(&mut e);
        assert_ne!(
            e.rocminfo_status, "",
            "sync_shared_fields_from_wsl must have run and set a real status"
        );
    }

    #[test]
    fn the_distro_list_decodes_as_utf16_whatever_script_it_is_in() {
        fn utf16le(text: &str, bom: bool) -> Vec<u8> {
            let mut bytes = if bom { vec![0xFF, 0xFE] } else { Vec::new() };
            for unit in text.encode_utf16() {
                bytes.extend_from_slice(&unit.to_le_bytes());
            }
            bytes
        }

        assert_eq!(decode_utf16le(&utf16le("Ubuntu\n", true)), "Ubuntu\n");
        assert_eq!(decode_utf16le(&utf16le("Ubuntu\n", false)), "Ubuntu\n");
        assert_eq!(decode_utf16le(b""), "");

        // The reason this is decoded by declaration rather than sniffed: in a
        // Latin or Cyrillic script every UTF-16LE byte is below 0x80, so the
        // bytes are *valid UTF-8* and decode without error into mojibake. Read
        // as UTF-8 these names come back as "#\u{4}1\u{4}..." rather than
        // failing, so no validity or NUL-density test could catch them.
        for name in [
            "Ubuntu-24.04\nÉtat\n",
            "Ubuntu\nУбунту\n",
            "Ubuntu\n日本語\n",
        ] {
            assert_eq!(decode_utf16le(&utf16le(name, true)), name);
            assert_eq!(decode_utf16le(&utf16le(name, false)), name);
        }

        // A non-ASCII name must survive all the way through the parser.
        assert_eq!(
            parse_wsl_distro_list(&decode_utf16le(&utf16le("Ubuntu\nУбунту\n", true))),
            vec!["Ubuntu", "Убунту"]
        );

        // An odd trailing byte is dropped rather than panicking.
        let mut truncated = utf16le("Ubuntu", false);
        truncated.push(0x00);
        assert_eq!(decode_utf16le(&truncated), "Ubuntu");
    }

    /// Ported from the same Python preflight, which owned this rule until the
    /// catalog took it over. Its version was well covered and its tests went with
    /// it, so the coverage has to live here or the floor becomes an untested
    /// constant.
    #[test]
    fn the_distro_floor_fails_closed_on_anything_it_cannot_read() {
        for supported in ["24.04", "24.10", "25.04", "26.04", "28.04"] {
            assert_eq!(
                distro_clears_wsl_floor("ubuntu", supported),
                Some(true),
                "ubuntu {supported} clears the floor"
            );
        }
        // 22.04 ships glibc 2.35, below the 2.38 / GLIBCXX_3.4.32 floor the
        // engines are linked against, so it cannot run them at all.
        assert_eq!(distro_clears_wsl_floor("ubuntu", "22.04"), Some(false));
        assert_eq!(distro_clears_wsl_floor("ubuntu", "20.04"), Some(false));

        // Unreadable is `None`, never `Some(true)`. Reporting a release nobody
        // could parse as supported is how a user ends up chasing a GPU fault
        // that is really a glibc floor -- but claiming it is too old would send
        // them to reinstall a perfectly good distro, so neither answer is safe.
        for unreadable in ["24.04.1", "unknown", "", "24", "24.x", "..", "24.04.1.2"] {
            assert_eq!(
                distro_clears_wsl_floor("ubuntu", unreadable),
                None,
                "{unreadable:?} cannot be read as a release"
            );
        }

        // Only Ubuntu carries a documented floor. Anything else abstains rather
        // than applying Ubuntu's numbering to a distro that does not share it --
        // Debian 12 is not "below 24.04".
        for other in ["debian", "fedora", "arch", ""] {
            assert_eq!(distro_clears_wsl_floor(other, "12.0"), None, "{other}");
        }
        assert_eq!(distro_clears_wsl_floor("UBUNTU", "24.04"), Some(true));
    }

    /// Guards [`TRY_OS_RELEASE_FN`] against drifting from the copy embedded in
    /// [`WSL_REMOTE_PROBE`]: the test below exercises the standalone constant,
    /// so if a future edit touches one and not the other, this catches it
    /// rather than the test silently proving the wrong logic.
    #[test]
    fn try_os_release_fn_is_embedded_in_the_wsl_remote_probe() {
        assert!(
            WSL_REMOTE_PROBE.contains(TRY_OS_RELEASE_FN),
            "WSL_REMOTE_PROBE's os-release handling has drifted from the tested function"
        );
    }

    /// The WSL remote probe's `try_os_release` shell function, run for real
    /// under `/bin/sh` against temp-directory paths rather than the real
    /// `/etc`: it sources the first path that exists, falling back to the
    /// second only when the first is absent, matching `os-release(5)` and the
    /// Rust-side `hardware_root::read_os_release`.
    #[cfg(target_os = "linux")]
    #[test]
    fn try_os_release_sources_the_fallback_only_when_the_primary_is_absent() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let primary = dir.path().join("primary");
        let fallback = dir.path().join("fallback");
        std::fs::write(&fallback, "ID=fallback\n").expect("write fallback");

        let script = format!("{TRY_OS_RELEASE_FN}\ntry_os_release \"$1\" \"$2\"\necho \"id=$ID\"");
        let run = || {
            std::process::Command::new("sh")
                .arg("-c")
                .arg(&script)
                .arg("sh")
                .arg(&primary)
                .arg(&fallback)
                .output()
                .expect("run sh")
        };

        let before = run();
        assert_eq!(
            String::from_utf8_lossy(&before.stdout).trim(),
            "id=fallback",
            "stderr: {}",
            String::from_utf8_lossy(&before.stderr)
        );

        std::fs::write(&primary, "ID=primary\n").expect("write primary");
        let after = run();
        assert_eq!(
            String::from_utf8_lossy(&after.stdout).trim(),
            "id=primary",
            "stderr: {}",
            String::from_utf8_lossy(&after.stderr)
        );
    }

    /// `[ -e "$1" ]` is an existence check, not a readability one: a primary
    /// that exists but cannot be sourced (here, a directory in its place, the
    /// same portable non-readable fixture the Rust-side test above uses)
    /// must not fall through to the `elif` and source the fallback — that
    /// would silently read a file that may not describe this host, matching
    /// `hardware_root::read_os_release`'s own non-absence rule.
    ///
    /// This is deliberately a directory, not a permission-denied file: CI
    /// commonly runs as root, where a `chmod 000` file is still readable, so
    /// a mode-000 fixture would pass or fail depending on who runs it. A
    /// directory fails to source for a different, root-safe reason ("is a
    /// directory"), which is enough to prove this one claim — existence is
    /// not readability — but is **not** the same failure mode as a genuinely
    /// unreadable primary. Under `/bin/sh` (`dash` on Debian/Ubuntu),
    /// sourcing a *permission-denied* file is a fatal shell-level error that
    /// aborts the whole embedded script immediately, not just this function —
    /// a materially different outcome than "`$ID` stays unset" and one this
    /// test does not exercise. That abort-on-unreadable-primary behavior
    /// predates the `/usr/lib/os-release` fallback this module adds; fixing
    /// it is out of scope here (tracked as a possible follow-up, not a
    /// blocker for the fallback itself).
    #[cfg(target_os = "linux")]
    #[test]
    fn try_os_release_does_not_source_the_fallback_when_the_primary_is_not_sourceable() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let primary = dir.path().join("primary");
        let fallback = dir.path().join("fallback");
        std::fs::create_dir(&primary).expect("create primary as a directory");
        std::fs::write(&fallback, "ID=fallback\n").expect("write fallback");

        let script = format!("{TRY_OS_RELEASE_FN}\ntry_os_release \"$1\" \"$2\"\necho \"id=$ID\"");
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .arg("sh")
            .arg(&primary)
            .arg(&fallback)
            .output()
            .expect("run sh");

        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "id=",
            "stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
