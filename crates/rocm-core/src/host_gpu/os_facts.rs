// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Operating-system facts the host summary reports: kernel version, distro name,
//! CPU model and installed RAM.

use super::capture::{
    OPTIONAL_COMMAND_TIMEOUT, capture_optional_command, capture_optional_command_with_timeout,
};
use crate::{hardware_root, host_path, runtime_is_linux, runtime_is_windows};
use std::fs;

pub(super) fn detect_kernel_version() -> Option<String> {
    if runtime_is_windows() {
        capture_optional_command("cmd", &["/C", "ver"])
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    } else {
        capture_optional_command("uname", &["-r"])
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    }
}

pub(crate) fn detect_distro_name() -> Option<String> {
    if runtime_is_windows() {
        return Some("Windows".to_owned());
    }

    if runtime_is_linux() {
        return match hardware_root::read_os_release(hardware_root::host_path_for_str) {
            // Absent matches the pre-fallback behaviour this function has
            // always had: no file at all leaves the distro name unknown
            // entirely, rather than reporting "Linux" for a machine that
            // might not even be running a standard distro.
            hardware_root::OsRelease::Absent => None,
            read => Some(distro_name_from_os_release(&read)),
        };
    }

    None
}

/// The distribution name `rocm examine` prints, from `read`: `PRETTY_NAME` as
/// `sh` reads it, else `Linux`.
///
/// An unreadable line says which file and line made it so, on the line a
/// person reads — `rocm examine`'s `probe_failures`, which carries the same
/// reason, is only in the `--json` form. A blank `PRETTY_NAME` also falls
/// back to `Linux`. A file that exists but could not be read at all is
/// reported as itself, distinctly from a bad line it never reached.
fn distro_name_from_os_release(read: &hardware_root::OsRelease) -> String {
    if let hardware_root::OsRelease::Unreadable { path, error } = read {
        return format!("Linux ({})", hardware_root::unreadable_prefix(path, error));
    }
    match crate::os_release::parse(read.text()) {
        Ok(mut fields) => fields
            .remove("PRETTY_NAME")
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "Linux".to_owned()),
        Err(unreadable) => format!("Linux ({} {unreadable})", read.path()),
    }
}

pub(crate) fn detect_cpu_model() -> Option<String> {
    if runtime_is_windows() {
        return detect_cpu_model_windows(|timeout| {
            let script =
                "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name";
            capture_optional_command_with_timeout(
                "powershell",
                &["-NoProfile", "-Command", script],
                timeout,
            )
        });
    }

    if runtime_is_linux()
        && let Some(model) = fs::read_to_string(host_path("/proc/cpuinfo"))
            .ok()
            .and_then(|text| cpu_model_from_cpuinfo(&text))
    {
        return Some(model);
    }

    None
}

/// The summary's own Windows CPU query, with the query injected so a test on any
/// host can drive it. `query` is handed the timeout the summary allows it.
pub(super) fn detect_cpu_model_windows(
    query: impl FnOnce(std::time::Duration) -> Option<String>,
) -> Option<String> {
    query(OPTIONAL_COMMAND_TIMEOUT)
        .map(|value| normalize_cpu_model(&value))
        .filter(|value| !value.is_empty())
}

/// The first `model name` or `Hardware` line of `/proc/cpuinfo` text that has a
/// value, whitespace collapsed.
pub(super) fn cpu_model_from_cpuinfo(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let value = line
            .strip_prefix("model name")
            .and_then(|rest| rest.split_once(':').map(|(_, value)| value))
            .or_else(|| {
                line.strip_prefix("Hardware")
                    .and_then(|rest| rest.split_once(':').map(|(_, value)| value))
            })?;
        let value = normalize_cpu_model(value);
        (!value.is_empty()).then_some(value)
    })
}

pub fn detect_system_ram_gib() -> Option<f64> {
    if runtime_is_windows() {
        let script = "(Get-CimInstance -ClassName Win32_ComputerSystem -Property TotalPhysicalMemory).TotalPhysicalMemory";
        return capture_optional_command_with_timeout(
            "powershell",
            &["-NoProfile", "-Command", script],
            OPTIONAL_COMMAND_TIMEOUT,
        )
        .and_then(|value| bytes_text_to_gib(&value));
    }

    if runtime_is_linux()
        && let Some(kib) = fs::read_to_string(host_path("/proc/meminfo"))
            .ok()
            .and_then(|text| {
                text.lines().find_map(|line| {
                    let value = line.strip_prefix("MemTotal:")?.trim();
                    let number = value.split_whitespace().next()?.parse::<f64>().ok()?;
                    Some(number)
                })
            })
    {
        return Some(kib / 1024.0 / 1024.0);
    }

    None
}

pub(super) fn bytes_text_to_gib(value: &str) -> Option<f64> {
    let bytes = value.trim().parse::<f64>().ok()?;
    (bytes > 0.0).then_some(bytes / 1024.0 / 1024.0 / 1024.0)
}

pub(super) fn format_gib_value(value: f64) -> String {
    if value >= 10.0 {
        format!("{value:.0} GiB")
    } else {
        format!("{value:.1} GiB")
    }
}

pub(super) fn normalize_cpu_model(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_os_release_pretty_name() {
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::found(
                "NAME=Ubuntu\nPRETTY_NAME=\"Ubuntu 24.04.2 LTS\"\n"
            )),
            "Ubuntu 24.04.2 LTS"
        );
    }

    /// The distro name is read through the shared `os_release` parser, so it
    /// names the distro `sh` would: the last assignment wins and a trailing
    /// comment is not part of the value. A file with no usable `PRETTY_NAME`
    /// is plain `Linux`; an unreadable one says which line, rather than
    /// letting `Linux` read as a description of the machine.
    #[test]
    fn pretty_name_reads_as_sh_reads_it() {
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::found(
                "PRETTY_NAME=\"Debian GNU/Linux 12\"\nPRETTY_NAME='Ubuntu 24.04 LTS' # LTS\n"
            )),
            "Ubuntu 24.04 LTS"
        );
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::found(
                "PRETTY_NAME=\"Ubuntu 24.04 LTS\"\nunset ID\n"
            )),
            "Linux (/etc/os-release line 2 is not a plain assignment: \"unset ID\")"
        );
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::found("PRETTY_NAME=\"  \"\n")),
            "Linux"
        );
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::found("ID=ubuntu\n")),
            "Linux"
        );
    }

    /// A file that exists but could not be read names itself and the error —
    /// new behavior; it used to be folded into "unknown" like an absent file.
    /// It is distinct from a bad line it never reached, and from "no file at
    /// all", which `detect_distro_name` still reports as unknown (see its own
    /// doc comment) — that part is unchanged.
    #[test]
    fn distro_name_names_an_unreadable_file_instead_of_guessing_linux() {
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::Unreadable {
                path: "/usr/lib/os-release",
                error: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
            }),
            "Linux (/usr/lib/os-release could not be read: denied)"
        );
    }

    /// The same malformed line as the second case in
    /// [`pretty_name_reads_as_sh_reads_it`], but read through the fallback
    /// path: the message must name `/usr/lib/os-release`, not
    /// `/etc/os-release`. That test only exercises
    /// [`hardware_root::OsRelease::found`], which hardcodes
    /// `/etc/os-release`, so a regression that hardcoded the path back into
    /// `distro_name_from_os_release`'s message would still pass it.
    #[test]
    fn a_fallback_reads_malformed_line_names_the_fallback_path() {
        assert_eq!(
            distro_name_from_os_release(&hardware_root::OsRelease::Found {
                path: "/usr/lib/os-release",
                text: "PRETTY_NAME=\"Ubuntu 24.04 LTS\"\nunset ID\n".to_owned(),
            }),
            "Linux (/usr/lib/os-release line 2 is not a plain assignment: \"unset ID\")"
        );
    }

    // ── CPU model: the summary's read and the examination's read ──────────
    //
    // Characterisation tests for the CPU-model pair of docs/host-probing.md.
    // `rocm examine --json` reads the CPU twice: the summary (`summary.cpu`, the
    // `cpu:` line of the text) through `detect_cpu_model` and the Windows
    // inventory's CPU row, and the examination (`cpu_model` and `cpu_vendor`)
    // through `probe_cpu_linux` and `probe_cpu_windows`. Each case below gives
    // both reads the same input and pins what each returns, so the change that
    // merges them moves exactly the assertions it means to. Where the two
    // agree, one value is asserted for both. Where they differ, each read's own
    // answer is asserted, under a comment that says so.

    use crate::examine::{Examination, probe_cpu_linux_from, probe_cpu_windows_with};
    use crate::host_gpu::windows_inventory::{
        InventoryScope, WindowsExamineInventory, WindowsInventorySources,
        assemble_windows_inventory, windows_system_inventory_with,
    };
    use std::time::Duration;

    /// One `/proc/cpuinfo` and what each read makes of it.
    struct CpuinfoCase {
        name: &'static str,
        text: &'static str,
        /// `detect_cpu_model`'s answer.
        summary: Option<&'static str>,
        /// `Examination::cpu_model` and `Examination::cpu_vendor`.
        examination: (&'static str, &'static str),
    }

    const INTEL_CPUINFO: &str = "processor\t: 0\nvendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel name\t: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz\nprocessor\t: 1\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz\n";

    #[test]
    fn linux_cpuinfo_the_two_reads_agree_on_an_ordinary_cpu() {
        let cases = [
            CpuinfoCase {
                name: "Intel, model name and vendor_id",
                text: INTEL_CPUINFO,
                summary: Some("Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz"),
                examination: ("Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz", "intel"),
            },
            CpuinfoCase {
                name: "AMD, model name and vendor_id",
                text: "vendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 9 9950X 16-Core Processor\n",
                summary: Some("AMD Ryzen 9 9950X 16-Core Processor"),
                examination: ("AMD Ryzen 9 9950X 16-Core Processor", "amd"),
            },
            CpuinfoCase {
                // No `vendor_id` (Arm kernels print none): the model is read, the
                // vendor stays "unknown" in the examination.
                name: "model name and no vendor_id",
                text: "processor\t: 0\nmodel name\t: ARMv8 Processor rev 1 (v8l)\n",
                summary: Some("ARMv8 Processor rev 1 (v8l)"),
                examination: ("ARMv8 Processor rev 1 (v8l)", "unknown"),
            },
            CpuinfoCase {
                name: "a blank model name line, then a real one",
                text: "vendor_id\t: GenuineIntel\nmodel name\t:   \nmodel name\t: Intel(R) Xeon(R) Gold 6248\n",
                summary: Some("Intel(R) Xeon(R) Gold 6248"),
                examination: ("Intel(R) Xeon(R) Gold 6248", "intel"),
            },
            CpuinfoCase {
                // `Hardware` after the per-core blocks is what an Arm board
                // prints, and it follows `model name`, so both read the latter.
                name: "model name first, Hardware last",
                text: "processor\t: 0\nmodel name\t: ARMv7 Processor rev 4 (v7l)\nHardware\t: BCM2835\n",
                summary: Some("ARMv7 Processor rev 4 (v7l)"),
                examination: ("ARMv7 Processor rev 4 (v7l)", "unknown"),
            },
            CpuinfoCase {
                name: "nothing to read",
                text: "processor\t: 0\nCPU implementer\t: 0x41\n",
                summary: None,
                examination: ("", "unknown"),
            },
            CpuinfoCase {
                name: "an empty file",
                text: "",
                summary: None,
                examination: ("", "unknown"),
            },
            CpuinfoCase {
                // The examination lower-cases a vendor it does not know, the
                // summary records no vendor at all.
                name: "a vendor_id that is neither AMD nor Intel",
                text: "vendor_id\t: HygonGenuine\nmodel name\t: Hygon C86 7185 32-core Processor\n",
                summary: Some("Hygon C86 7185 32-core Processor"),
                examination: ("Hygon C86 7185 32-core Processor", "hygongenuine"),
            },
        ];
        for case in cases {
            assert_cpuinfo_case(&case);
        }
    }

    /// The inputs on which the two reads give different answers today; each
    /// read's own answer is pinned.
    #[test]
    fn linux_cpuinfo_the_two_reads_differ_on_four_inputs() {
        let cases = [
            CpuinfoCase {
                // The summary collapses whitespace runs, the examination trims
                // only: one model string, two spellings in one `--json`.
                name: "repeated spaces inside the model name",
                text: "vendor_id\t: GenuineIntel\nmodel name\t:  Intel(R) Pentium(R) CPU  N3700  @ 1.60GHz  \n",
                summary: Some("Intel(R) Pentium(R) CPU N3700 @ 1.60GHz"),
                examination: ("Intel(R) Pentium(R) CPU  N3700  @ 1.60GHz", "intel"),
            },
            CpuinfoCase {
                // The summary falls back to `Hardware`; the examination reads
                // `model name` only, so it reports no CPU at all.
                name: "Hardware and no model name",
                text: "processor\t: 0\nBogoMIPS\t: 38.40\nHardware\t: BCM2835\nRevision\t: a02082\n",
                summary: Some("BCM2835"),
                examination: ("", "unknown"),
            },
            CpuinfoCase {
                // The summary takes whichever of the two comes first, the
                // examination only ever the `model name`.
                name: "Hardware before model name",
                text: "Hardware\t: BCM2835\nmodel name\t: ARMv7 Processor rev 4 (v7l)\n",
                summary: Some("BCM2835"),
                examination: ("ARMv7 Processor rev 4 (v7l)", "unknown"),
            },
            CpuinfoCase {
                // An empty `vendor_id` value is taken as the vendor by the
                // examination, which then stops looking for another.
                name: "an empty vendor_id value",
                text: "vendor_id\t:\nmodel name\t: Some CPU\nvendor_id\t: GenuineIntel\n",
                summary: Some("Some CPU"),
                examination: ("Some CPU", ""),
            },
        ];
        for case in cases {
            assert_cpuinfo_case(&case);
        }
    }

    fn assert_cpuinfo_case(case: &CpuinfoCase) {
        assert_eq!(
            cpu_model_from_cpuinfo(case.text).as_deref(),
            case.summary,
            "summary, {}",
            case.name
        );
        let mut examined = Examination::default();
        probe_cpu_linux_from(&mut examined, case.text);
        assert_eq!(
            (examined.cpu_model.as_str(), examined.cpu_vendor.as_str()),
            case.examination,
            "examination, {}",
            case.name
        );
        assert!(
            examined.probe_failures.is_empty(),
            "a cpuinfo with no CPU in it is not a probe failure, {}: {:?}",
            case.name,
            examined.probe_failures
        );
    }

    /// What the examination makes of one answer from its Windows query.
    fn examination_on_windows(rc: i32, stdout: &str) -> (Examination, Duration) {
        let mut examined = Examination::default();
        let mut asked = None;
        probe_cpu_windows_with(&mut examined, |timeout| {
            asked = Some(timeout);
            (rc, stdout.to_owned(), String::new())
        });
        (examined, asked.expect("the examination ran its query"))
    }

    /// The summary's system inventory with only `system` wired, so a test sees
    /// the query the CPU row comes from.
    struct SystemQuery<'a> {
        output: Option<&'a str>,
        asked: Option<(String, Duration)>,
    }

    impl WindowsInventorySources for SystemQuery<'_> {
        fn pnputil(&mut self) -> Option<WindowsExamineInventory> {
            None
        }
        fn video_controller(&mut self) -> Option<WindowsExamineInventory> {
            None
        }
        fn pnp_entity(&mut self) -> Option<WindowsExamineInventory> {
            None
        }
        fn system(&mut self) -> Option<WindowsExamineInventory> {
            windows_system_inventory_with(|script, timeout| {
                self.asked = Some((script.to_owned(), timeout));
                self.output.map(str::to_owned)
            })
        }
    }

    /// What the summary reports for the CPU on Windows when the system query
    /// prints `system_output` (`None`: it could not run) and its own fallback
    /// query prints `fallback_output`.
    struct SummaryOnWindows {
        cpu: Option<String>,
        system_query: Option<(String, Duration)>,
        fallback_timeout: Option<Duration>,
    }

    fn summary_on_windows(
        system_output: Option<&str>,
        fallback_output: Option<&str>,
    ) -> SummaryOnWindows {
        let mut sources = SystemQuery {
            output: system_output,
            asked: None,
        };
        let inventory = assemble_windows_inventory(&mut sources, InventoryScope::SUMMARY);
        let mut fallback_timeout = None;
        let cpu = crate::host_gpu::windows_cpu_model(inventory.as_ref(), || {
            detect_cpu_model_windows(|timeout| {
                fallback_timeout = Some(timeout);
                fallback_output.map(str::to_owned)
            })
        });
        SummaryOnWindows {
            cpu,
            system_query: sources.asked,
            fallback_timeout,
        }
    }

    const WINDOWS_CPU_NAME: &str = "AMD Ryzen 9 9950X 16-Core Processor";

    #[test]
    fn windows_the_two_reads_agree_on_a_name_the_query_returns() {
        let summary = summary_on_windows(
            Some(&format!("CPU\t{WINDOWS_CPU_NAME}\nRAM\t68719476736\n")),
            None,
        );
        assert_eq!(summary.cpu.as_deref(), Some(WINDOWS_CPU_NAME));
        assert_eq!(
            summary.fallback_timeout, None,
            "an inventory was read, so the summary's own query does not run"
        );
        let (examined, _) = examination_on_windows(0, &format!("{WINDOWS_CPU_NAME}\r\n"));
        assert_eq!(examined.cpu_model, WINDOWS_CPU_NAME);
        assert_eq!(examined.cpu_vendor, "amd");
        assert!(examined.probe_failures.is_empty());
    }

    /// The summary collapses whitespace runs, the examination trims and keeps
    /// the first line only. Windows pads the name of some CPUs.
    #[test]
    fn windows_the_two_reads_differ_on_whitespace_inside_the_name() {
        let summary = summary_on_windows(
            Some("CPU\t  Intel(R) Pentium(R) CPU  N3700  @ 1.60GHz   \n"),
            None,
        );
        assert_eq!(
            summary.cpu.as_deref(),
            Some("Intel(R) Pentium(R) CPU N3700 @ 1.60GHz")
        );
        let (examined, _) =
            examination_on_windows(0, "  Intel(R) Pentium(R) CPU  N3700  @ 1.60GHz   \r\n");
        assert_eq!(
            examined.cpu_model,
            "Intel(R) Pentium(R) CPU  N3700  @ 1.60GHz"
        );
        assert_eq!(examined.cpu_vendor, "intel");
        // Only the examination reads a first line out of several.
        let (several, _) = examination_on_windows(0, "AMD EPYC 7763\r\nAMD EPYC 7763\r\n");
        assert_eq!(several.cpu_model, "AMD EPYC 7763");
    }

    #[test]
    fn windows_the_examination_derives_the_vendor_from_the_model_string() {
        for (name, vendor) in [
            ("AMD Ryzen 7 5800X 8-Core Processor", "amd"),
            ("amd epyc 7763", "amd"),
            ("Intel(R) Core(TM) i9-13900K", "intel"),
            ("Genuine Intel(R) CPU 0000", "intel"),
            ("Qualcomm(R) Snapdragon(R) X Elite", "unknown"),
        ] {
            let (examined, _) = examination_on_windows(0, name);
            assert_eq!(examined.cpu_vendor, vendor, "{name}");
            assert_eq!(examined.cpu_model, name);
        }
    }

    /// The system query printed no CPU row (here only the RAM row), or nothing.
    /// The summary has an inventory, so it reports no CPU and does not ask
    /// again; the examination reads blank output as a failed query.
    #[test]
    fn windows_an_answer_with_no_cpu_in_it() {
        for (system_output, output) in [
            (Some("RAM\t17179869184\n"), "\r\n"),
            (Some("CPU\t   \nRAM\t17179869184\n"), "   "),
        ] {
            let summary = summary_on_windows(system_output, Some("must not be read"));
            assert_eq!(summary.cpu, None);
            assert_eq!(summary.fallback_timeout, None);
            let (examined, _) = examination_on_windows(0, output);
            assert_eq!(examined.cpu_model, "");
            assert_eq!(examined.cpu_vendor, "unknown");
            assert_eq!(
                examined.probe_failures,
                ["Get-CimInstance Win32_Processor failed; cannot identify CPU."]
            );
        }
    }

    /// The system query printed nothing at all, so there is no inventory. The
    /// summary asks the CPU question a second time, itself, and keeps that
    /// answer; the examination reports a failure.
    #[test]
    fn windows_an_empty_answer_sends_the_summary_to_its_own_query() {
        let summary = summary_on_windows(Some(""), Some("Intel(R) Core(TM) i5-12400\r\n"));
        assert_eq!(summary.cpu.as_deref(), Some("Intel(R) Core(TM) i5-12400"));
        assert!(summary.fallback_timeout.is_some());
        let (examined, _) = examination_on_windows(0, "");
        assert_eq!(examined.cpu_model, "");
        assert_eq!(examined.probe_failures.len(), 1);
    }

    /// The query could not run, failed or timed out. Both stacks read that as
    /// no CPU; the summary tries its own query once more first, the examination
    /// records a failure and leaves the vendor unknown.
    #[test]
    fn windows_a_query_that_fails_or_times_out() {
        // `run` answers 127 for a program it could not start and 124 for one
        // it killed at the deadline.
        for rc in [1, 124, 127] {
            let (examined, _) = examination_on_windows(rc, "");
            assert_eq!(examined.cpu_model, "", "rc {rc}");
            assert_eq!(examined.cpu_vendor, "unknown", "rc {rc}");
            assert_eq!(examined.probe_failures.len(), 1, "rc {rc}");
        }
        // A non-zero exit with a name on stdout is not trusted either.
        let (failed_with_output, _) = examination_on_windows(1, WINDOWS_CPU_NAME);
        assert_eq!(failed_with_output.cpu_model, "");
        assert_eq!(failed_with_output.probe_failures.len(), 1);

        let both_fail = summary_on_windows(None, None);
        assert_eq!(both_fail.cpu, None);
        assert!(both_fail.fallback_timeout.is_some());
        let second_try = summary_on_windows(None, Some(WINDOWS_CPU_NAME));
        assert_eq!(second_try.cpu.as_deref(), Some(WINDOWS_CPU_NAME));
    }

    /// The limits: the examination allows its query 8 s, the summary allows the
    /// system query behind the inventory's CPU row 1.5 s, and its own fallback
    /// query 1.5 s too. A host that answers in between is read by one and not
    /// the other.
    #[test]
    fn windows_the_two_reads_are_given_different_time() {
        let (_, examination_limit) = examination_on_windows(0, WINDOWS_CPU_NAME);
        assert_eq!(examination_limit, Duration::from_secs(8));
        let summary = summary_on_windows(None, None);
        let (script, system_limit) = summary.system_query.expect("the system query ran");
        assert_eq!(system_limit, Duration::from_millis(1_500));
        assert!(
            script.contains("Win32_Processor") && script.contains("Win32_ComputerSystem"),
            "the CPU row comes from the system query, which also asks for RAM: {script}"
        );
        assert_eq!(summary.fallback_timeout, Some(Duration::from_millis(1_500)));
    }

    /// With an inventory the summary never starts its own CPU query, even when
    /// the inventory has no CPU row: that is `windows_cpu_model`'s rule, and the
    /// fallback is for the case with no inventory at all.
    #[test]
    fn windows_the_summary_asks_its_own_query_only_without_an_inventory() {
        let with_cpu = crate::host_gpu::windows_inventory::parse_windows_examine_inventory(
            "CPU\tAMD EPYC 7763\n",
        );
        let without_cpu = crate::host_gpu::windows_inventory::parse_windows_examine_inventory(
            "RAM\t17179869184\n",
        );
        let never = || -> Option<String> { panic!("an inventory was given") };
        assert_eq!(
            crate::host_gpu::windows_cpu_model(Some(&with_cpu), never).as_deref(),
            Some("AMD EPYC 7763")
        );
        assert_eq!(
            crate::host_gpu::windows_cpu_model(Some(&without_cpu), never),
            None
        );
        assert_eq!(
            crate::host_gpu::windows_cpu_model(None, || Some("from the fallback".to_owned()))
                .as_deref(),
            Some("from the fallback")
        );
    }
}
