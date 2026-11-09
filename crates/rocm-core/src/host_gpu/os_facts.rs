// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Operating-system facts the host summary reports: kernel version, distro name,
//! CPU model and installed RAM.

use super::capture::{
    OPTIONAL_COMMAND_TIMEOUT, capture_optional_command, capture_optional_command_with_timeout,
};
use super::windows_inventory::{
    WindowsExamineInventory, run_windows_system_query, windows_system_inventory_with,
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

/// The CPU's model and vendor, read together by the one reader both stacks use:
/// the summary's `cpu` through [`detect_cpu_model`] and the Windows inventory,
/// the examination's `cpu_model` and `cpu_vendor` through `probe_cpu_linux` and
/// `probe_cpu_windows`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CpuFacts {
    /// The model with every run of whitespace collapsed to one space, or `None`
    /// when no model could be read. `None` covers every way of not knowing (no
    /// file, no line, a blank value, a query that failed or timed out), so a
    /// caller cannot tell a failed query from an absent line; the Windows
    /// examination records a failure for any `None`, and Linux for none.
    pub(crate) model: Option<String>,
    /// `amd` or `intel`, any other vendor lower-cased as the file spells it, or
    /// [`UNKNOWN_CPU_VENDOR`] when none could be read.
    pub(crate) vendor: String,
}

/// The vendor of a CPU no source named.
pub(crate) const UNKNOWN_CPU_VENDOR: &str = "unknown";

impl CpuFacts {
    fn unknown() -> Self {
        Self {
            model: None,
            vendor: UNKNOWN_CPU_VENDOR.to_owned(),
        }
    }
}

/// The CPU summary line: the model of [`read_cpu_facts`].
pub(crate) fn detect_cpu_model() -> Option<String> {
    read_cpu_facts().model
}

/// The CPU facts of the host this process runs on.
pub(crate) fn read_cpu_facts() -> CpuFacts {
    if runtime_is_windows() {
        read_windows_cpu_facts()
    } else if runtime_is_linux() {
        read_linux_cpu_facts()
    } else {
        CpuFacts::unknown()
    }
}

/// The CPU facts from `/proc/cpuinfo`, read through [`host_path`].
pub(crate) fn read_linux_cpu_facts() -> CpuFacts {
    cpu_facts_from_cpuinfo(&fs::read_to_string(host_path("/proc/cpuinfo")).unwrap_or_default())
}

/// The CPU facts from `/proc/cpuinfo` text.
///
/// The model is the first `model name` or `Hardware` line that has a value,
/// whichever comes first: an Arm board that prints no `model name` is still
/// named. The vendor is the first `vendor_id` line that has a value, `amd` or
/// `intel` by substring (`AuthenticAMD`, `GenuineIntel`) and any other
/// lower-cased. A file with no `vendor_id` (Arm kernels print none) has an
/// unknown vendor beside a known model.
pub(super) fn cpu_facts_from_cpuinfo(text: &str) -> CpuFacts {
    let mut facts = CpuFacts::unknown();
    for line in text.lines() {
        if facts.model.is_none()
            && let Some(value) = line
                .strip_prefix("model name")
                .or_else(|| line.strip_prefix("Hardware"))
                .and_then(|rest| rest.split_once(':').map(|(_, value)| value))
        {
            let value = normalize_cpu_model(value);
            if !value.is_empty() {
                facts.model = Some(value);
            }
        }
        if facts.vendor == UNKNOWN_CPU_VENDOR
            && let Some(value) = line
                .strip_prefix("vendor_id")
                .and_then(|rest| rest.split_once(':').map(|(_, value)| value.trim()))
                .filter(|value| !value.is_empty())
        {
            facts.vendor = if value.contains("AMD") {
                "amd".to_owned()
            } else if value.contains("Intel") {
                "intel".to_owned()
            } else {
                value.to_lowercase()
            };
        }
        if facts.model.is_some() && facts.vendor != UNKNOWN_CPU_VENDOR {
            break;
        }
    }
    facts
}

/// The CPU facts from the Windows system inventory's `CPU` row: the one query
/// the summary's inventory runs for its CPU and RAM.
pub(crate) fn read_windows_cpu_facts() -> CpuFacts {
    windows_cpu_facts_with(run_windows_system_query)
}

/// [`read_windows_cpu_facts`] with the query injected, so a test on any host can
/// drive it. `query` is handed the script and the time the inventory allows it,
/// and answers with what the script printed, or `None` when it could not run,
/// failed or timed out.
pub(super) fn windows_cpu_facts_with(
    query: impl FnOnce(&str, std::time::Duration) -> Option<String>,
) -> CpuFacts {
    cpu_facts_from_windows_inventory(windows_system_inventory_with(query).as_ref())
}

/// The CPU facts of a Windows inventory: its CPU row, or unknown when there is
/// no inventory or the row is absent or blank. The summary reads the inventory
/// it already holds through this, so it starts no second query for the CPU and
/// reads the same row the examination's reader does.
///
/// Windows reports no vendor field here, so it is derived from the model string
/// (`amd` or `intel`, else unknown).
pub(super) fn cpu_facts_from_windows_inventory(
    inventory: Option<&WindowsExamineInventory>,
) -> CpuFacts {
    let Some(model) = inventory.and_then(|inventory| inventory.cpu_model.clone()) else {
        return CpuFacts::unknown();
    };
    let lowered = model.to_lowercase();
    let vendor = if lowered.contains("amd") {
        "amd"
    } else if lowered.contains("intel") {
        "intel"
    } else {
        UNKNOWN_CPU_VENDOR
    };
    CpuFacts {
        model: Some(model),
        vendor: vendor.to_owned(),
    }
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
    // Tests of the CPU-model pair of docs/host-probing.md. `rocm examine --json`
    // reads the CPU in two places: the summary (`summary.cpu`, the `cpu:` line of
    // the text) and the examination (`cpu_model` and `cpu_vendor`). Both go
    // through one reader now, so each case below hands both the same input and
    // asserts one answer. The cases that used to differ say so in their
    // comment, with what each read answered before.

    use crate::examine::{Examination, record_cpu, record_cpu_windows};
    use crate::host_gpu::windows_inventory::{
        InventoryScope, WindowsExamineInventory, WindowsInventorySources,
        assemble_windows_inventory, windows_system_inventory_with,
    };
    use std::time::Duration;

    /// One `/proc/cpuinfo` and the CPU both reads make of it.
    struct CpuinfoCase {
        name: &'static str,
        text: &'static str,
        /// The model: `summary.cpu`, and `cpu_model` where `Some`, empty where `None`.
        model: Option<&'static str>,
        /// `cpu_vendor`.
        vendor: &'static str,
    }

    #[test]
    fn linux_cpuinfo_both_reads_report_one_cpu() {
        let cases = [
            CpuinfoCase {
                name: "Intel, model name and vendor_id",
                text: "processor\t: 0\nvendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel name\t: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz\nprocessor\t: 1\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz\n",
                model: Some("Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz"),
                vendor: "intel",
            },
            CpuinfoCase {
                name: "AMD, model name and vendor_id",
                text: "vendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 9 9950X 16-Core Processor\n",
                model: Some("AMD Ryzen 9 9950X 16-Core Processor"),
                vendor: "amd",
            },
            CpuinfoCase {
                // No `vendor_id` (Arm kernels print none): the model is read, the
                // vendor stays "unknown".
                name: "model name and no vendor_id",
                text: "processor\t: 0\nmodel name\t: ARMv8 Processor rev 1 (v8l)\n",
                model: Some("ARMv8 Processor rev 1 (v8l)"),
                vendor: "unknown",
            },
            CpuinfoCase {
                name: "a blank model name line, then a real one",
                text: "vendor_id\t: GenuineIntel\nmodel name\t:   \nmodel name\t: Intel(R) Xeon(R) Gold 6248\n",
                model: Some("Intel(R) Xeon(R) Gold 6248"),
                vendor: "intel",
            },
            CpuinfoCase {
                // `Hardware` after the per-core blocks is what an Arm board
                // prints, and it follows `model name`, so both read the latter.
                name: "model name first, Hardware last",
                text: "processor\t: 0\nmodel name\t: ARMv7 Processor rev 4 (v7l)\nHardware\t: BCM2835\n",
                model: Some("ARMv7 Processor rev 4 (v7l)"),
                vendor: "unknown",
            },
            CpuinfoCase {
                name: "nothing to read",
                text: "processor\t: 0\nCPU implementer\t: 0x41\n",
                model: None,
                vendor: "unknown",
            },
            CpuinfoCase {
                name: "an empty file",
                text: "",
                model: None,
                vendor: "unknown",
            },
            CpuinfoCase {
                // A vendor other than AMD and Intel is kept, lower-cased.
                name: "a vendor_id that is neither AMD nor Intel",
                text: "vendor_id\t: HygonGenuine\nmodel name\t: Hygon C86 7185 32-core Processor\n",
                model: Some("Hygon C86 7185 32-core Processor"),
                vendor: "hygongenuine",
            },
        ];
        for case in cases {
            assert_cpuinfo_case(&case);
        }
    }

    /// The inputs on which the two reads used to give different answers. Each
    /// comment says what the summary and the examination answered before the
    /// reads were merged; the answer asserted is the one both give now.
    #[test]
    fn linux_cpuinfo_the_inputs_the_two_reads_used_to_disagree_on() {
        let cases = [
            CpuinfoCase {
                // Before: summary collapsed whitespace runs, examination trimmed
                // only (`Intel(R) Pentium(R) CPU  N3700  @ 1.60GHz`).
                name: "repeated spaces inside the model name",
                text: "vendor_id\t: GenuineIntel\nmodel name\t:  Intel(R) Pentium(R) CPU  N3700  @ 1.60GHz  \n",
                model: Some("Intel(R) Pentium(R) CPU N3700 @ 1.60GHz"),
                vendor: "intel",
            },
            CpuinfoCase {
                // Before: summary read `Hardware`, examination read `model name`
                // only and reported no CPU (empty model, vendor unknown).
                name: "Hardware and no model name",
                text: "processor\t: 0\nBogoMIPS\t: 38.40\nHardware\t: BCM2835\nRevision\t: a02082\n",
                model: Some("BCM2835"),
                vendor: "unknown",
            },
            CpuinfoCase {
                // Before: summary took whichever of the two came first
                // (`BCM2835`), examination only the `model name`.
                name: "Hardware before model name",
                text: "Hardware\t: BCM2835\nmodel name\t: ARMv7 Processor rev 4 (v7l)\n",
                model: Some("BCM2835"),
                vendor: "unknown",
            },
            CpuinfoCase {
                // Before: examination took an empty `vendor_id` as the vendor
                // (`""`) and stopped looking. An empty value names no vendor, so
                // the next `vendor_id` line is read.
                name: "an empty vendor_id value",
                text: "vendor_id\t:\nmodel name\t: Some CPU\nvendor_id\t: GenuineIntel\n",
                model: Some("Some CPU"),
                vendor: "intel",
            },
            CpuinfoCase {
                // The same input where no later line names a vendor. Before:
                // examination `""`; now `unknown`, which is what the JSON says
                // for a vendor that was not read.
                name: "an empty vendor_id value and no other",
                text: "vendor_id\t:\nmodel name\t: Some CPU\n",
                model: Some("Some CPU"),
                vendor: "unknown",
            },
        ];
        for case in cases {
            assert_cpuinfo_case(&case);
        }
    }

    fn assert_cpuinfo_case(case: &CpuinfoCase) {
        // The summary's `cpu`: what `detect_cpu_model` makes of the file.
        assert_eq!(
            cpu_facts_from_cpuinfo(case.text).model.as_deref(),
            case.model,
            "summary, {}",
            case.name
        );
        // The examination's fields.
        let mut examined = Examination::default();
        record_cpu(&mut examined, cpu_facts_from_cpuinfo(case.text));
        assert_eq!(
            (examined.cpu_model.as_str(), examined.cpu_vendor.as_str()),
            (case.model.unwrap_or(""), case.vendor),
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

    /// What the examination makes of one answer from the system query.
    fn examination_on_windows(output: Option<&str>) -> (Examination, Duration) {
        let mut examined = Examination::default();
        let mut asked = None;
        record_cpu_windows(
            &mut examined,
            windows_cpu_facts_with(|_, timeout| {
                asked = Some(timeout);
                output.map(str::to_owned)
            }),
        );
        (examined, asked.expect("the examination ran its query"))
    }

    /// The summary's system inventory with only `system` wired, so a test sees
    /// the query the CPU row comes from and counts how many run.
    struct SystemQuery<'a> {
        output: Option<&'a str>,
        asked: Vec<(String, Duration)>,
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
                self.asked.push((script.to_owned(), timeout));
                self.output.map(str::to_owned)
            })
        }
    }

    /// What the summary reports for the CPU on Windows when the system query
    /// prints `output` (`None`: it could not run), and every query that ran.
    fn summary_on_windows(output: Option<&str>) -> (Option<String>, Vec<(String, Duration)>) {
        let mut sources = SystemQuery {
            output,
            asked: Vec::new(),
        };
        let inventory = assemble_windows_inventory(&mut sources, InventoryScope::SUMMARY);
        (
            cpu_facts_from_windows_inventory(inventory.as_ref()).model,
            sources.asked,
        )
    }

    const WINDOWS_CPU_NAME: &str = "AMD Ryzen 9 9950X 16-Core Processor";

    #[test]
    fn windows_both_reads_report_the_name_the_query_returns() {
        let output = format!("CPU\t{WINDOWS_CPU_NAME}\r\nRAM\t68719476736\r\n");
        let (summary, _) = summary_on_windows(Some(&output));
        assert_eq!(summary.as_deref(), Some(WINDOWS_CPU_NAME));
        let (examined, _) = examination_on_windows(Some(&output));
        assert_eq!(examined.cpu_model, WINDOWS_CPU_NAME);
        assert_eq!(examined.cpu_vendor, "amd");
        assert!(examined.probe_failures.is_empty());
    }

    /// Before: the summary collapsed whitespace runs and the examination
    /// trimmed and kept the first line only. Windows pads the name of some CPUs.
    /// Both collapse now.
    #[test]
    fn windows_both_reads_collapse_whitespace_inside_the_name() {
        let output = "CPU\t  Intel(R) Pentium(R) CPU  N3700  @ 1.60GHz   \r\n";
        let (summary, _) = summary_on_windows(Some(output));
        assert_eq!(
            summary.as_deref(),
            Some("Intel(R) Pentium(R) CPU N3700 @ 1.60GHz")
        );
        let (examined, _) = examination_on_windows(Some(output));
        assert_eq!(
            examined.cpu_model,
            "Intel(R) Pentium(R) CPU N3700 @ 1.60GHz"
        );
        assert_eq!(examined.cpu_vendor, "intel");
    }

    #[test]
    fn windows_the_vendor_is_derived_from_the_model_string() {
        for (name, vendor) in [
            ("AMD Ryzen 7 5800X 8-Core Processor", "amd"),
            ("amd epyc 7763", "amd"),
            ("Intel(R) Core(TM) i9-13900K", "intel"),
            ("Genuine Intel(R) CPU 0000", "intel"),
            ("Qualcomm(R) Snapdragon(R) X Elite", "unknown"),
        ] {
            let (examined, _) = examination_on_windows(Some(&format!("CPU\t{name}\n")));
            assert_eq!(examined.cpu_vendor, vendor, "{name}");
            assert_eq!(examined.cpu_model, name);
        }
    }

    /// The query printed no CPU row (only the RAM row, or a blank one), or
    /// nothing, or could not run: both reads report no CPU. The examination
    /// records a failure and leaves the vendor `unknown`; the summary starts no
    /// second query. Before: an empty answer or a query that could not run
    /// sent the summary to a second, separate PowerShell query for the CPU.
    #[test]
    fn windows_an_answer_with_no_cpu_is_no_cpu_in_both_reads() {
        for output in [
            Some("RAM\t17179869184\n"),
            Some("CPU\t   \nRAM\t17179869184\n"),
            Some(""),
            None,
        ] {
            let (summary, queries) = summary_on_windows(output);
            assert_eq!(summary, None, "{output:?}");
            assert_eq!(
                queries.len(),
                1,
                "one system query, no second one: {output:?}"
            );
            let (examined, _) = examination_on_windows(output);
            assert_eq!(examined.cpu_model, "", "{output:?}");
            assert_eq!(examined.cpu_vendor, "unknown", "{output:?}");
            assert_eq!(
                examined.probe_failures,
                ["Get-CimInstance Win32_Processor failed; cannot identify CPU."],
                "{output:?}"
            );
        }
    }

    /// One limit for both reads: 5 s, the limit of every inventory query. Before:
    /// 8 s for the examination, 1.5 s for the summary's system query and for its
    /// own CPU query.
    #[test]
    fn windows_both_reads_are_given_the_same_time() {
        let (_, examination_limit) = examination_on_windows(Some("CPU\tAMD EPYC 7763\n"));
        assert_eq!(examination_limit, Duration::from_secs(5));
        let (_, queries) = summary_on_windows(Some("CPU\tAMD EPYC 7763\n"));
        let (script, summary_limit) = &queries[0];
        assert_eq!(*summary_limit, Duration::from_secs(5));
        assert!(
            script.contains("Win32_Processor") && script.contains("Win32_ComputerSystem"),
            "the CPU row comes from the system query, which also asks for RAM: {script}"
        );
    }

    /// With no inventory at all the summary has no CPU, and reads the CPU row of
    /// an inventory it holds without asking again.
    #[test]
    fn windows_the_inventory_is_the_summarys_only_source() {
        assert_eq!(cpu_facts_from_windows_inventory(None), CpuFacts::unknown());
        let inventory = crate::host_gpu::windows_inventory::parse_windows_examine_inventory(
            "CPU\tAMD EPYC 7763\n",
        );
        assert_eq!(
            cpu_facts_from_windows_inventory(Some(&inventory)),
            CpuFacts {
                model: Some("AMD EPYC 7763".to_owned()),
                vendor: "amd".to_owned()
            }
        );
    }
}
