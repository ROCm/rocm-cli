<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Host probing: which stack owns each fact

Status: proposal. This note changes no code and no behaviour. It decides nothing by itself: it lists the host facts that `rocm-core` reads twice, says how the two reads differ, and proposes which one should survive and in what order a later series of changes could remove the other. Each pair is meant to be a separate change with its own review.

How to read the claims below. Everything marked "verified" was read in the source on `main` at the time of writing, and the symbols named are the identity if a file moves. Nothing was executed on a Windows or WSL host, and every count of runs or queries below comes from reading the call graph, not from measuring a process. Anything that depends on how a real machine behaves, rather than on what the code says, is marked UNCONFIRMED.

## The two stacks

| | Summary stack | Examination stack |
|---|---|---|
| Entry point | `ExamineSummary::gather` in [host_gpu.rs](../crates/rocm-core/src/host_gpu.rs) | `Examination::probe` in [examine.rs](../crates/rocm-core/src/examine.rs) |
| Printed by | the text of `rocm examine`, the `summary` object of `rocm examine --json`, the MCP and bridge snapshots, `rocm install driver` | the top level of `rocm examine --json`, `rocm diagnose`, Doctor reports, remote probes |
| Subprocess helper | `capture_optional_command*` in [capture.rs](../crates/rocm-core/src/host_gpu/capture.rs) | `run` and `which` in [probe_exec.rs](../crates/rocm-core/src/probe_exec.rs) |
| A probe that fails | reads as `None`, no trace (an opt-in stderr line with `ROCM_CLI_DEBUG_COMMAND_CAPTURE`) | usually a `probe_failures` entry, which also degrades `status` |

The two meet in one command. [examine_cmd.rs](../apps/rocm/src/examine_cmd.rs) builds the `--json` document by running `Examination::probe_with_interpreter` and then `ExamineSummary::gather`, so each fact below that the host has appears twice in one document, read by two separate runs (the WSL and Windows ones only on those hosts). The text form ([examine_report_text.rs](../apps/rocm/src/examine_report_text.rs)) prints the summary only, and `rocm diagnose` ([diagnose_cmd.rs](../apps/rocm/src/diagnose_cmd.rs)) reasons over the examination only. The existing scenarios `examine-machine-readable-report` and `examine-both-forms-agree-on-gpu` in [examine.feature](../tests/e2e-cucumber/features/examine.feature) hold the two forms level on GPU presence and on the facts the text states; none of them compares kernel, CPU or the gfx target of an individual adapter.

The direction of the import is already one-way, but sharing an implementation is not the same as sharing an observation, and the note keeps the two apart. Verified, in production code: the examination stack calls into the summary stack for `is_wsl_host` (`probe_os`, `probe_wsl`), `detect_wsl_summary` and `ldconfig_lists_librocdxg` (both in `probe_wsl`), `ldconfig_cache` (`probe_comgr` in [library_copies.rs](../crates/rocm-core/src/examine/library_copies.rs)), `detect_windows_amd_driver_info` (`probe_adrenalin_windows`) and `gfx_target_from_pnp_device_id` (`parse_gpus_windows_output`; the function lives in [gfx_target.rs](../crates/rocm-core/src/gfx_target.rs) and is re-exported through `host_gpu`). `detect_distro_name` is not on that list: the only references to it under `examine/` are two tests in [host_env.rs](../crates/rocm-core/src/examine/host_env.rs). In production `record_distro` calls `hardware_root::read_os_release` and `os_release::parse` directly, which is the reader and parser `detect_distro_name` is built on, so the two share an implementation and still read the file separately. Nothing under `host_gpu/` imports `examine` outside the summary module's own tests and doc comments.

Which of these is also one read is a separate question, and for most of them the answer is no. `rocm examine --json` runs `detect_wsl_summary` twice on WSL, once in `ExamineSummary::gather` and once in `probe_wsl`, and each run repeats the device-node and ROCm file checks, one `ldconfig -p` and the two `PATH` lookups; `probe_wsl` then asks `ldconfig` a third time through `ldconfig_lists_librocdxg`, and `probe_comgr` a fourth through `ldconfig_cache`. `is_wsl_host` re-reads `/dev/dxg` and `/proc/version` at every call. The os-release file is read once by each stack, and the Windows driver is read by its own inventory run (see "Windows GPU query"). So the WSL check, the os-release parser, the Windows driver and the `ldconfig` lookup are shared code, not yet shared results: none of them is a duplicate read that has already been removed, and the plan below does not count them as done. The proposals continue in the same import direction, so the rule of thumb is: the summary stack owns how a fact is asked of the operating system, the examination stack owns the document (field names, `probe_failures`, the JSON contract) and calls down. Whether the two stacks should also share the answer within one run, rather than ask twice through the same code, is left open at the end.

## Kernel version

Verdict: confirmed duplicate for `uname -r`. `uname -v` in `probe_os` has no twin.

| | Summary | Examination |
|---|---|---|
| Symbol | `detect_kernel_version` in [os_facts.rs](../crates/rocm-core/src/host_gpu/os_facts.rs) | `probe_os` in [host_env.rs](../crates/rocm-core/src/examine/host_env.rs) |
| Linux command | `uname -r`, 1.5 s | `uname -r`, 5 s (`SHORT`) |
| Non-zero exit | `None`, stdout dropped | exit code ignored, stdout kept |
| Failure | `None`, reads `<unknown>` | empty `kernel_release`, no `probe_failures` entry |
| Output decoding | strict UTF-8, else `None` | lossy |
| Windows | `cmd /C ver` as `kernel`, resolved through the System32 candidate | the same command as `os_version`; `kernel_release` stays empty |
| Field | `kernel:` in the text, `summary.kernel` in JSON | `kernel_release` (and `os_version` from `uname -v`) |

What a user can see: on a host where `uname` is slow or fails, `kernel:` can read `<unknown>` while `kernel_release` is filled in, or the reverse. UNCONFIRMED on a real host; the code allows it. `kernel_release` also feeds `is_wsl1_kernel` for the WSL facts, so its empty-versus-populated behaviour is not cosmetic.

Proposal: `detect_kernel_version` survives and `probe_os` calls it for `kernel_release` on Linux, keeping `uname -v` where it is. Calling it on Windows too would populate `kernel_release` there, which is a visible change to the JSON; decide that separately. The timeout is open (1.5 s or 5 s): the characterisation test should pin both before one is chosen.

## CPU model

Verdict: confirmed duplicate, and on Windows three implementations: `detect_cpu_model` (its own PowerShell call), the `CPU` row of `WINDOWS_SYSTEM_INVENTORY_SCRIPT` in [windows_inventory.rs](../crates/rocm-core/src/host_gpu/windows_inventory.rs), and `probe_cpu_windows`. They are not three reads of one run in the usual case: the summary reads the inventory row first and calls `detect_cpu_model` only when no inventory exists, so a host whose inventory answers is read twice in `rocm examine --json` and one whose inventory does not answer is read three times (derived from `detect_cpu_model_with_windows_inventory` in [host_gpu.rs](../crates/rocm-core/src/host_gpu.rs); UNCONFIRMED on a live host).

| | Summary | Examination |
|---|---|---|
| Symbols | `detect_cpu_model` and `normalize_cpu_model` in [os_facts.rs](../crates/rocm-core/src/host_gpu/os_facts.rs) | `probe_cpu_linux`, `probe_cpu_windows` in [host_env.rs](../crates/rocm-core/src/examine/host_env.rs) |
| Linux source | `/proc/cpuinfo`: the first line that is `model name` or `Hardware` | `/proc/cpuinfo`: `model name` only, plus `vendor_id` |
| Whitespace | collapsed | trimmed only |
| Windows | `Win32_Processor` name, 1.5 s | same class, 8 s, first line only |
| Vendor | not recorded | `cpu_vendor`: from `vendor_id` on Linux, from the model string (`amd` or `intel`) on Windows |
| Failure | `<unknown>` | Windows: a `probe_failures` entry; Linux: empty model, vendor `unknown` |

What a user can see: a board whose cpuinfo has `Hardware` but no `model name` shows a CPU in the text and an empty `cpu_model` with vendor `unknown` in the JSON. A model string with repeated spaces inside it differs between the two forms (UNCONFIRMED which real CPUs report one; the code path differs). A Windows host where the 1.5 s query times out shows `cpu: <unknown>` beside a populated `cpu_model` (UNCONFIRMED, timing). No diagnosis checker reads `cpu_model` or `cpu_vendor` today (verified by grep); they reach Doctor reports and the JSON only.

Proposal: one reader in `os_facts` that returns the model and the vendor together (Linux from cpuinfo with the `Hardware` fallback and `vendor_id`; Windows from the inventory row, deriving the vendor from the model as `probe_cpu_windows` does). `probe_cpu_linux` and `probe_cpu_windows` call it. Open: whether the whitespace collapse should reach `cpu_model` in Doctor reports.

## Windows GPU query

Verdict: confirmed duplicate, and wider than the ticket names: the summary side has several Windows adapter queries, all in [windows_inventory.rs](../crates/rocm-core/src/host_gpu/windows_inventory.rs), and the examination has one.

| | Summary | Examination |
|---|---|---|
| Symbols | `assemble_windows_inventory` over `pnputil`, `WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT`, `WINDOWS_PNP_ENTITY_INVENTORY_SCRIPT`; plus the one-line query in `detect_windows_amd_display_driver` | `WIN_GPU_SCRIPT` in [windows.rs](../crates/rocm-core/src/examine/windows.rs) |
| Vendors | AMD only | AMD and NVIDIA (`has_nvidia_gpu`, non-AMD entries in `gpus[]`) |
| Driver date | yes, `yyyy-MM-dd` in UTC | none; `Gpu` carries no driver fields |
| Source order | `pnputil` first, the video controller only when `pnputil` lists no adapter or the driver date is wanted | always the video controller |
| Timeout | 5 s | 8 s |
| Failure | the next source, else `None` | a `probe_failures` entry |
| Adapter choice | `preferred_amd_display`: a supported PCI target first, an unsupported iGPU last | every adapter, in the order WMI returns them |

The driver facts are not duplicated in the same way: `probe_adrenalin_windows` already fills `adrenalin_version` and `adrenalin_date` from the summary stack's inventory. What is duplicated is the adapter list, and the driver is read by two more inventory runs, one for the examination (`DRIVER_ONLY` scope) and one for the summary (`SUMMARY`). Counting the examination's own query, `rocm examine --json` asks Windows for `Win32_VideoController` three times in the usual case, where the summary's inventory exists: once in the summary's inventory run (as the display source, or as the driver-fact query after `pnputil`), once in `probe_gpus_windows`, and once in the `DRIVER_ONLY` run behind `probe_adrenalin_windows`. Three is not the upper bound. When the summary's inventory is absent, because no source in `assemble_windows_inventory` returned a display, a CPU or a RAM row, two more queries run on that absence: `detect_windows_display_gfx_target_with_inventory(None)` calls `detect_windows_display_gfx_target`, which runs `WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT` again, and `windows_driver_summary_from_inventory(None, ...)` runs the one-line `detect_windows_amd_display_driver` query. That is five logical queries, and `capture_optional_command_with_timeout` may retry each under further candidate spellings of `powershell` (all derived from the code; UNCONFIRMED on a live host, and the count depends on which sources answer). A failed inventory is the case where a host is already struggling to answer, so it is also the case where the extra queries cost the most. Any convergence on one inventory has to account for it: either the absent inventory is remembered for the rest of the run so the fallbacks do not ask again, or the fallbacks are dropped and the text and JSON say "not detected" instead.

What a user can see: on a Windows host with an NVIDIA card, the JSON lists it and sets `has_nvidia_gpu`, and the summary stack never mentions it, so an NVIDIA-only host reads "AMD GPU not detected yet" in the text beside `has_nvidia_gpu: true` in the JSON. The text prints no adapter list at all, only `detected_gfx_target` and the driver lines; the JSON's `gpus[]` is the only place adapters appear. The driver version and date appear twice in the JSON (`summary.driver.windows` and `adrenalin_*`) from two runs; the scenario `examine-windows-driver-version-date-and-age` holds the text equal to the JSON. The AMD test itself is written three ways (the PowerShell `-match`, `is_amd_windows_display`'s word match, and the substring test in `parse_gpus_windows_output`).

Proposal: the summary stack's inventory survives and `probe_gpus_windows` builds `gpus[]` from it. Two constraints decide how:

- NVIDIA has to survive in the JSON. Either the inventory rows carry a vendor and every AMD accessor filters on it (the last fallback of `preferred_amd_display` takes any named adapter, so an unfiltered NVIDIA row would be chosen as "the AMD adapter" when nothing else matches), or the examination keeps a small vendor-only check on the PNP id. The first is the larger change; the second leaves one query.
- `gpus[].name` and `pci_id` should come from the video controller source specifically, not from "whichever source answered first". `pnputil` supplies a cleaned device description and an instance id, which may not equal the WMI `Name` and `PNPDeviceID` (UNCONFIRMED), and the JSON name is user-visible.

`fill_unidentified_gpus_from_hipinfo` stays in the examination: it is enrichment of the document, not a second read of the same fact.

## Marketing name to gfx target

Verdict: confirmed duplicate, but not equivalent. The two cover different names.

| | Summary | Examination |
|---|---|---|
| Symbols | `AMD_MARKETING_GFX_TARGETS` in [gfx_target.rs](../crates/rocm-core/src/gfx_target.rs), through `gfx_target_from_amd_marketing_name` | `classify_amd_marketing_name` and `APU_KEYWORDS` in [linux_gpus.rs](../crates/rocm-core/src/examine/linux_gpus.rs) |
| Answers | a gfx target, for the discrete RDNA2 to RDNA4 parts and for APU names | a gfx target only for Strix Halo, Ryzen AI Max, the Radeon 8045S, 8050S and 8060S, 880M, 890M, Strix Point, Krackan, the Radeon 740M, 760M and 780M, Phoenix and Hawk Point; for the other APU codenames (Rembrandt, Raphael, Barcelo, Lucienne, Renoir, Cezanne) only "is an APU" with an empty target |
| Matching | whole tokens, after every non-alphanumeric character becomes a space | substring, after removing `(tm)`, `(r)`, `(c)` |
| APU verdict | none (the caller derives it from the target with `gfx_is_apu_family`) | yes, from the name |
| Used on | Windows adapters and the WSL host's display probe | Windows adapters after the PNP id, and the Linux `lspci` scan, which asks no table of its own |

The two agree where they overlap on `880m`, `780m` and `8060s`. They disagree on Krackan: `classify_amd_marketing_name` returns `gfx1150` for the token `krackan`, while the table maps `860m` and `840m` to `gfx1152`, and `gfx_target_from_amd_pci_device_id` maps device `1114` to `gfx1152`. Which is right for a given part is UNCONFIRMED here (no hardware); the open pull request [#521](https://github.com/ROCm/rocm-cli/pull/521) reaches the same disagreement and resolves it toward `gfx1152`.

What a user can see, on Windows without a HIP SDK: an RX 7900 XTX is not in the PCI-id table and not in `classify_amd_marketing_name`, so `gpus[].gfx_target` is empty while `summary.detected_gfx_target` in the same JSON document is `gfx1100`, from the name table. (Derived from the code; `hipInfo` fills the examination's value in when an SDK is installed.) On native Linux the summary reads sysfs and KFD rather than names, so only the examination's `lspci` path can show a name-derived empty target there; WSL uses the name table through the Windows host's display probe.

Proposal: the table survives. It is pure, data-driven and already the superset for target names. It gains the codenames `classify_amd_marketing_name` knows, and the APU verdict is derived from the target, falling back to a codename list only while some codename still has no target. [#521](https://github.com/ROCm/rocm-cli/pull/521) already proposes exactly this convergence and is open and conflicting with `main`; the cheapest route is to finish and rebase it rather than start a second change. Its Linux device-id work is a separate concern from this pair.

## PATH lookup

Verdict: confirmed duplicate.

| | Summary | Examination |
|---|---|---|
| Symbol | `tool_on_path` and `tool_path_candidates` in [capture.rs](../crates/rocm-core/src/host_gpu/capture.rs) | `which` in [probe_exec.rs](../crates/rocm-core/src/probe_exec.rs) |
| Splitting `PATH` | `split_paths`; an empty entry means the current directory | split on `;` or `:`; empty entries skipped |
| `PATH` that is not UTF-8 | still searched | not found |
| Windows names | the bare name, every `PATHEXT` entry in both cases, plus the fixed System32 locations for `cmd`, `pnputil`, `powershell`, `pwsh` | `.exe`, `.bat`, `.cmd` and the bare name |
| Callers | `detect_wsl_summary` (`rocminfo`, `cargo`) | about fifteen call sites in `examine/` and [runners.rs](../crates/rocm-core/src/fix/runners.rs) |

What a user can see: only on WSL, where both run on Linux: the summary's `wsl_global_rocminfo` and the examination's `wsl.rocminfo` answer the same question with `tool_on_path` and `which`, and differ for an empty `PATH` entry or a `PATH` that is not UTF-8. With a `rocminfo` file in the current directory and an empty `PATH` entry (a leading or trailing `:`, or `::`), `tool_on_path` reads it as found and `which` does not, so `wsl.rocminfo` is false where the summary's answer is true (derived from `split_paths` against `which`'s `is_empty` skip; not run). With a `PATH` that is not UTF-8, `which` returns false for every program through its `std::env::var` read, and `tool_on_path` still searches. Every other caller of `which` (`usermod`, `sudo`, `python`, `python3`, `mokutil`, `lspci`, `rocminfo`, `dpkg`, `rpm`, the llama.cpp binaries, `wsl.exe`) takes the same differences, so those are not confined to WSL. On Windows the examination's `framework` probe uses `which` for `python`, which would miss a `PATHEXT` extension outside its list, and `which` splits on `;` without the quote handling of `split_paths` (both UNCONFIRMED in practice, and how Windows `split_paths` treats an empty entry is UNCONFIRMED too). The System32 candidates of `tool_path_candidates` would also make `cmd`, `pnputil`, `powershell` and `pwsh` read as present without being on `PATH`, but no current `which` call site asks for them (verified by grep). A near copy of `tool_path_candidates` also exists in `apps/rocm` (`program_path_candidates` in [python.rs](../apps/rocm/src/therock/python.rs)); it is outside this pair and worth deciding with it.

Proposal: the behaviour of `tool_on_path` survives (it is the one that follows `PATHEXT`), housed where both stacks already import from, in `probe_exec.rs`, under the existing name `which` so the call sites do not change. `tool_path_candidates` moves with it, because the runner uses it to try each spelling. Keeping the call sites unchanged does not keep their answers unchanged: the differences listed above become the intended observable differences of this change (see the first item of the order below). The empty-entry case is a real choice, because either answer changes one side. Following `PATH` semantics matches what a shell or `execvp` does with an empty entry, and makes `which` read a file in the current directory as found, which widens what the examination calls installed (fifteen or so capability checks, including `sudo` and `usermod`). Skipping empty entries keeps the examination's answer and changes only the summary's on WSL, at the cost of disagreeing with `execvp` for a program the CLI later starts by bare name. The note does not pick one; the change has to, and its PR text states which.

## Subprocess runners

Verdict: confirmed duplicate, and a family rather than a pair.

| Helper | Streams | Result | Default timeout | Reader threads |
|---|---|---|---|---|
| `capture_optional_command_with_timeout` ([capture.rs](../crates/rocm-core/src/host_gpu/capture.rs)) | stdout only; stderr discarded | `Option<String>`: only on success and valid UTF-8 | 1.5 s, 5 s for inventory queries | joined with no limit; tries each candidate spelling in turn, including after a timeout |
| `run`, `run_raw` ([probe_exec.rs](../crates/rocm-core/src/probe_exec.rs)) | both | `(rc, stdout, stderr)`, `127` for spawn failure, `124` for timeout, lossy UTF-8 | 5 s, 8 s, 15 s | joined with no limit |
| `capture_diagnostic_command` ([diagnostics.rs](../crates/rocm-core/src/host_gpu/diagnostics.rs), Windows only) | both | status, text, error and a timed-out flag | per call | joined with no limit |
| `run_with_timeout` ([process.rs](../crates/rocm-core/src/process.rs)) | both | `Result<Output>`; a timeout is an `Err` that quotes bounded output | per call | abandoned after a short grace |

The first three join their reader threads with no limit, so a grandchild that keeps the pipe open can outlast their timeout; [process.rs](../crates/rocm-core/src/process.rs) documents why `run_with_timeout` does not. That defect is already tracked in [#623](https://github.com/ROCm/rocm-cli/issues/623), which lists the same helpers. Separately, the candidate loop in `capture_optional_command_with_timeout` keeps going after a timeout, so on Windows a hung `powershell` may be started again under several spellings (UNCONFIRMED; the loop does not stop on a timeout).

The capture type from the earlier `rocm` capture work is on `main`: `CommandCapture` in [command_capture.rs](../crates/rocm-core/src/command_capture.rs), built over `run_with_timeout`. As it stands it is not enough for the probes, and the reasons are specific. Its constructor runs the `rocm` binary only (the general `capture_program` is not public). It stores lossy text, so `run_utf16le`, which decodes `wsl.exe`'s UTF-16LE output from raw bytes, cannot use it. It reports `1` for a child killed by a signal, which is also a real exit code. And `run_with_timeout` signals a timeout only through the text of its error, while a spawn failure should be recognisable by downcasting to `std::io::Error` (not exercised here), so a probe cannot tell "not installed" (quiet) from "timed out" (worth a `probe_failures` entry) without parsing a message.

Proposal: `run_with_timeout` survives as the one bounded executor, with `CommandCapture` as the shared result carrier extended with a typed outcome (not found, timed out, exited with a code) rather than a second capture type. `run` and `capture_optional_command*` become thin adapters that keep their current return shapes, so the first change touches no call site. That is not the same as keeping every observable result: a timeout is where they differ (see the last item of the order below). The adapters can then be retired site by site as the pairs above remove call sites. This pair overlaps [#623](https://github.com/ROCm/rocm-cli/issues/623); extend that issue's scope with the typed outcome rather than opening a parallel one.

## Proposed order

One pair per change. Each begins with a commit that adds a characterisation test and is green against the code as it is: it pins what both copies return for the same input, and where they differ it pins each copy's answer, with a comment naming the pair. The convergence commit then flips the assertions that are meant to change and the PR text lists each one as a user-visible difference. Tests that need a program on `PATH` use the fake-executable helper in [fake_executable.rs](../crates/rocm-core/src/fake_executable.rs); Windows-only logic goes behind an injectable function so a unit test runs on the Windows job, as the inventory already does with `WindowsInventorySources`.

1. PATH lookup. Smallest, and it settles where `tool_path_candidates` lives before the runner work needs it. It is not output-preserving: the differences listed under "PATH lookup" (an empty `PATH` entry, a `PATH` that is not UTF-8, and on Windows the candidate spellings and the `;` splitting) change which tools are reported as present, on WSL in `wsl.rocminfo` and everywhere in the examination's capability checks. Its PR text lists each one as an intentional difference, with the characterisation test that pins it.
2. Kernel version. A single command; settles the timeout and the empty-versus-`None` question.
3. CPU model. Adds the model-and-vendor reader; needs the Windows inventory row, which already exists.
4. Marketing name to gfx target, by finishing [#521](https://github.com/ROCm/rocm-cli/pull/521). It must land before the next item, so that `gpus[]` is not rebuilt on top of two name tables.
5. Windows GPU query. The riskiest: it needs an answer on NVIDIA and on adapter names, and a Windows lane. Scenario `examine-windows-driver-version-date-and-age` already runs there; a new scenario should compare the text's `detected_gfx_target` with the JSON `gpus[]` on a named adapter.
6. Subprocess runners. Last, because items 1 to 5 reduce the number of call sites, but independent: the bounded-join fix in [#623](https://github.com/ROCm/rocm-cli/issues/623) does not wait for it and can land at any time. Its intended difference is the timeout path: `run` and `run_raw` join their reader threads and so wait for a grandchild that holds the pipe open, while `run_with_timeout` abandons them after a short grace and reports a timeout as an error that quotes bounded output. A probe behind a hung grandchild returns at its deadline instead of late, and the output it keeps after a timeout can differ (derived from the code, not run).

Items 2 and 3 should add a scenario that compares `summary.kernel` with `kernel_release` and `summary.cpu` with `cpu_model` in one `--json` run, since no existing scenario does.

No item is output-preserving without qualification, and the PR for each says which of its differences are intended instead of claiming nothing changed. Items 2 to 5 change visible output (a kernel or CPU string, a gfx target, an adapter name or vendor). Item 1 changes which tools are reported as present, per the list under item 1. Item 6 changes timing and timed-out output only, which a user sees as a probe that no longer hangs. For every listed difference the PR either adds a scenario that exercises it in `tests/e2e-cucumber/features/` or states why none can: the empty-entry and non-UTF-8 cases can be driven from a scenario that sets `PATH` and plants a file in the working directory, whereas the Windows candidate spellings cannot run on the Linux lane and need the injectable-function unit test on the Windows job (UNCONFIRMED that the harness can set a non-UTF-8 `PATH`; if it cannot, the PR says so).

## What this note leaves open

- How NVIDIA adapters are represented once the examination reads the inventory (a vendor on every row, or a separate check).
- Whether one `rocm examine --json` run shares answers between the stacks (a single WSL summary and a single `ldconfig -p`, one inventory including its failure) or keeps calling shared code that asks again.
- What an absent Windows inventory does once there is one inventory: remember the absence for the run, or drop the per-fact fallbacks.
- Whether an empty `PATH` entry counts as the current directory in the surviving lookup.
- Whether `gpus[].name` should come from `pnputil` or from WMI.
- Which timeout wins for the kernel and CPU reads, and whether `kernel_release` should be filled on Windows.
- Whether the typed outcome belongs on `CommandCapture` or beside it.
- Whether the near copy of `tool_path_candidates` in `apps/rocm` joins the PATH lookup change.
