<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# ROCm CLI v0.1.0 Release notes

## Release highlights

This is the first stable release, moving off preview versioning
(v0.1.0-preview.1 → v0.1.0). It adds ROCm 10 install support, vLLM wheel
auto-discovery, non-interactive installs, and dozens of reliability fixes
across install, serve, examine, diagnose, and the dashboard TUI.

- **ROCm 10 "next" install layout**: `rocm install` now supports it out of the
  box, without disturbing existing layouts ([#329]).
- **vLLM on ROCm 10 just works**: `vllm` auto-discovers matching flash-attn and
  amd-aiter wheels instead of requiring manual pinning ([#416]).
- **Non-interactive installs**: `rocm install --yes` skips prompts for
  scripted/CI setups ([#273]).
- **WSL2 support in `doctor`**: diagnoses and fixes WSL2-specific issues
  directly instead of telling you to look elsewhere ([#340]).
- **Clearer diagnostics**: `diagnose` now catches vLLM engine-startup import
  failures ([#374]) and shared-memory allowances too small to serve on ([#386]).
- **Visibility into downloads and installs**: live progress bars for
  multi-gigabyte ROCm downloads ([#347]) and spinners during tarball extraction
  and ComfyUI dependency install ([#397]).
- **`services`**: a supported way to remove non-live local server records, no
  more manual cleanup ([#411]).
- **`examine --json`** now reports `active_runtime_root` ([#431]).
- **`version`** now shows the release tag/branch and commit hash, useful for bug
  reports ([#244]).

### Fixed

- Lemonade's ROCm llama.cpp backend staying pinned to an old version
  instead of tracking the active ROCm SDK ([#404]).
- Dashboard not restoring the terminal on SIGTERM/SIGINT, and Ctrl-C not
  reliably quitting ([#326]).
- `serve` race conditions and incorrect GPU selection through visibility
  masks ([#267]).
- `examine` probing the ambient torch install instead of the active
  runtime's ([#405]).
- Ambiguous-runtime install errors in ComfyUI giving no actionable next
  step ([#328]).
- Driver-install reporting and the `runtimes uninstall --yes` gate being
  too loose ([#402]).
- `bench` missing the engine column and reporting unwindowed load
  latency ([#327]).
- `dash --replay` accepting an invalid replay path and crashing into the
  TUI ([#331]).
- `install` requiring sudo even when already running as root ([#305]).
- `install` failing to resolve canonical ROCm streams ([#308]).
- `install` occasionally realigning a working GPU-kernel torch onto a
  broken one ([#314]).
- `examine` misreading the KFD gfx target from the node properties file
  ([#343]).
- `chat` not accepting a piped one-shot prompt from stdin ([#339]), and
  not telling the assistant which machine it's running on ([#321]).
- `rocm fix` failures going to stdout instead of stderr, sometimes
  dropping the log guard ([#350]).
- Rollback path was silent and its single-level limit undocumented
  ([#345]).
- `runtimes list` legend drifting out of sync with its markers ([#344]).
- Dashboard TUI stale "update freshness" warnings, missing activity glyph
  key ([#357]), inconsistent marker/legend rendering ([#354]), and Section D
  navigation/job-completion visibility ([#361]).
- Dialog behavior and dimmed-backdrop UX being inconsistent across TUI
  overlays ([#358]).
- `diagnose` and `rocm fix` using different wording for the same
  remediation flags ([#413]).
- Unbounded local-service HTTP reads that could hang ([#348]), `http_get`
  connect phase not respecting the timeout ([#338]), and an HTTP read not
  retrying after signal interruption ([#252]).

### Breaking changes

None in this release. Starting with v0.1.0, the CLI's command surface is
considered stable — breaking changes in future releases will be called out here,
with migration notes where relevant.

### Known issues

- Managed Lemonade ROCm backend can hang on install, or use excessive disk
  space, on Windows ([#392]).
- WSL: dashboard ignores a working WSL `amd-smi` and looks for the wrong binary
  path ([#276]).
- Instinct serving on RHEL 9.6 can still require manual
  vLLM/OpenMPI/amdsmi workarounds ([#254]).

<!--
PR links. The v0.1.0 release notes on GitHub render "#NNN" as an automatic
cross-reference; here they need explicit targets.
-->

[#244]: https://github.com/ROCm/rocm-cli/pull/244
[#252]: https://github.com/ROCm/rocm-cli/pull/252
[#254]: https://github.com/ROCm/rocm-cli/pull/254
[#267]: https://github.com/ROCm/rocm-cli/pull/267
[#273]: https://github.com/ROCm/rocm-cli/pull/273
[#276]: https://github.com/ROCm/rocm-cli/pull/276
[#305]: https://github.com/ROCm/rocm-cli/pull/305
[#308]: https://github.com/ROCm/rocm-cli/pull/308
[#314]: https://github.com/ROCm/rocm-cli/pull/314
[#321]: https://github.com/ROCm/rocm-cli/pull/321
[#326]: https://github.com/ROCm/rocm-cli/pull/326
[#327]: https://github.com/ROCm/rocm-cli/pull/327
[#328]: https://github.com/ROCm/rocm-cli/pull/328
[#329]: https://github.com/ROCm/rocm-cli/pull/329
[#331]: https://github.com/ROCm/rocm-cli/pull/331
[#338]: https://github.com/ROCm/rocm-cli/pull/338
[#339]: https://github.com/ROCm/rocm-cli/pull/339
[#340]: https://github.com/ROCm/rocm-cli/pull/340
[#343]: https://github.com/ROCm/rocm-cli/pull/343
[#344]: https://github.com/ROCm/rocm-cli/pull/344
[#345]: https://github.com/ROCm/rocm-cli/pull/345
[#347]: https://github.com/ROCm/rocm-cli/pull/347
[#348]: https://github.com/ROCm/rocm-cli/pull/348
[#350]: https://github.com/ROCm/rocm-cli/pull/350
[#354]: https://github.com/ROCm/rocm-cli/pull/354
[#357]: https://github.com/ROCm/rocm-cli/pull/357
[#358]: https://github.com/ROCm/rocm-cli/pull/358
[#361]: https://github.com/ROCm/rocm-cli/pull/361
[#374]: https://github.com/ROCm/rocm-cli/pull/374
[#386]: https://github.com/ROCm/rocm-cli/pull/386
[#392]: https://github.com/ROCm/rocm-cli/pull/392
[#397]: https://github.com/ROCm/rocm-cli/pull/397
[#402]: https://github.com/ROCm/rocm-cli/pull/402
[#404]: https://github.com/ROCm/rocm-cli/pull/404
[#405]: https://github.com/ROCm/rocm-cli/pull/405
[#411]: https://github.com/ROCm/rocm-cli/pull/411
[#413]: https://github.com/ROCm/rocm-cli/pull/413
[#416]: https://github.com/ROCm/rocm-cli/pull/416
[#431]: https://github.com/ROCm/rocm-cli/pull/431