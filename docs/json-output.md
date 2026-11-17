<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# JSON output

Which `rocm` commands print JSON with `--json`, the rules they follow, and the
commands that do not have it yet.

## Rules

- **Text is the default.** `--json` is opt-in per command; leaving it out never
  changes the text a person reads.
- **JSON goes to stdout, and only JSON.** With `--json`, stdout holds one JSON
  document and nothing else, so a script can parse all of it. Diagnostics,
  progress and warnings go to stderr. A command that has no single document to
  print is listed below as unsupported rather than given a `--json` that mixes
  the two.

  One known exception, tracked as a follow-up: the startup update check that
  runs before most commands can print a progress line on stdout ahead of the
  document, such as `Python from PATH is not usable for this ROCm install`.
  It runs at most once per check interval, and prints only when the runtime it
  checks is a wheel install whose recorded Python is not usable and a Python
  it finds instead (on `PATH`, or a saved managed one) does not fit. Setting
  `ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK` turns the check off, and with it the
  line.
- **Errors are not JSON today.** A command that fails with `--json` prints its
  error to stderr, prints nothing on stdout, and exits 1, unless the command
  documents its own code, as `rocm fix` does. Command lines the argument parser
  rejects (an unknown flag, such as `--json` on a command without one, or a
  missing argument) exit 2 with a usage message on stderr, before the command
  runs. A script tells success from failure by the exit code, not by parsing an
  error document. A JSON error shape belongs to the shared envelope below. The
  one refusal printed as JSON today is `rocm diagnose --report --json`
  declining to prepare a report: that is the command working, so it exits 0
  and prints `refusal_envelope` from
  [report.rs](../crates/rocm-core/src/report.rs) (`schema`, `refused`,
  `explanation`, `architecture_matrix`).
- **Documented fields do not change silently.** A field may be added. Renaming
  or removing one, or giving it a new meaning, needs a versioned transition:
  publish the new shape under a new version and keep the old one readable while
  callers move.
- **Two shapes are frozen contracts** that other tools are built against:
  - `rocm examine --json` — the Examination document the diagnosis tooling
    reads. `examination_top_level_keys_match_examine_py_contract` in
    [examine.rs](../crates/rocm-core/src/examine.rs) fails if a top-level field
    is added, removed or renamed.
  - `rocm fix --json` — the fix catalog, carrying `contract_version`
    (`CATALOG_CONTRACT_VERSION` in
    [manifest.rs](../crates/rocm-core/src/fix/manifest.rs), 1 today). The
    rocm-doctor skill states the version it needs, and scenario `skill-07`
    holds the two equal.
- **New commands print through the shared helper.** A command whose output is
  one value implements `JsonOrText` and calls `print_json_or_text` in
  [cli_report.rs](../apps/rocm/src/cli_report.rs), which writes pretty JSON plus
  one newline, or the text, and nothing else.

## Which commands have `--json`

| Command | Today | Decision | Proven by |
|---|---|---|---|
| `rocm examine` | supported; pretty JSON | supported, frozen | `examine-07`, `examine-08`; `examination_top_level_keys_match_examine_py_contract` |
| `rocm diagnose` | supported; pretty JSON (`DiagnoseOutput` through `JsonOrText`); with `--report`, the prepared report or its refusal | supported | `diagnose-02`, `diagnose-03` |
| `rocm fix` | the catalog only: `--json` is refused with a fix id; `contract_version` 1 | catalog supported, frozen; a JSON result for applying one fix waits for the envelope | `skill-07` |
| `rocm update` | supported for the check; one compact JSON line, not pretty, and refused with `--apply` and `--dry-run` | supported; the compact form is a known difference from the others | `update-02`, `update-03` |
| `rocm storage report` | supported; pretty JSON through `JsonOrText` | supported | `json_or_text_impl_renders_the_report_and_serializes_it` |
| `rocm services list` | supported; pretty JSON. `rocm remote` reads it back over its control channel | supported | `services_json_round_trips_and_applies_the_same_liveness_filter_as_the_table`; exercised indirectly by `remote-15` (`@requires-docker`), whose `rocm remote` reads it back |
| `rocm runtimes list` | supported; pretty JSON through `JsonOrText` | supported, additive-only, no version field yet | `runtime-lifecycle-23` to `runtime-lifecycle-31` and `runtimes_list_json_has_exactly_the_documented_fields_newest_install_first`; what each covers is in [testing.md](testing.md#runtime-selection-and-activation) |
| `rocm runtimes activate`, `rollback`, `uninstall`, `import`, `adopt` | none | follow-up: they change state, so they need the envelope's result and refusal shapes | — |
| `rocm services logs` | none | intentionally unsupported: streamed text, and `-f` never ends | — |
| `rocm services stop`, `restart`, `remove`, `prune` | none | follow-up: needs the envelope, including the refusal path | — |
| `rocm install sdk`, `rocm install driver` | none | follow-up: long-running with progress; needs the envelope and an audit that progress stays off stdout | — |
| `rocm uninstall` | none (`--dry-run` prints the plan as text) | follow-up | — |
| `rocm daemon --status` | none | follow-up; a one-shot snapshot, so a good next command | — |
| `rocm daemon` | none | intentionally unsupported: it runs until stopped | — |

Commands not in the table (for example `rocm engines list`, `rocm config show`,
`rocm automations list`, `rocm comfyui status`, `rocm setup status`,
`rocm version`) have no `--json` and no decision recorded yet.

## `rocm runtimes list --json`

The same listing as the text form, as one object:

| Field | Value |
|---|---|
| `default_runtime_id` | the configured default runtime id as written in the config, or `null`. The text form labels it `active_runtime_id:`. Not necessarily the active runtime's `runtime_id`: an `active_runtime_key` that names a runtime takes precedence, and the id may differ from its runtime's in letter case or match no runtime at all. Read each runtime's `active` flag to see which one is in use |
| `active_runtime_key` | the configured active runtime key, or `null` |
| `previous_runtime_key` | the configured rollback key, or `null` |
| `registry`, `marker` | paths of the runtime registry folder and the active-runtime marker |
| `active_problem` | `null`, or what is wrong with the configured active runtime, as described under [which runtimes are marked](#which-runtimes-are-marked): `{"kind": "missing_manifest_for_key", "runtime_key"}`, `{"kind": "missing_manifest_for_id", "runtime_id"}`, or `{"kind": "ambiguous_runtime_id", "runtime_id", "runtime_keys": [...]}` |
| `runtimes` | one object per registered runtime, newest install first, ties broken by `runtime_key` |

Each runtime carries `runtime_key`, `runtime_id`, `version` (as recorded; the
text adds the build date), `format`, `family`, `mode` (`managed` or
`read-only`), `status` (`ready` when the runtime passes the manifest check
that activation applies, `unusable` when it fails it; `ready` does not promise
that an activation succeeds), `unusable_reason` (`null` when ready, otherwise
why the runtime fails that check: text for a person, not a code to match on),
`toolchain` (`included` or `excluded`, the words `rocm examine` uses), `active`,
`rollback_target`, `install_root` and `provenance`. `active` and `rollback_target` are the facts the text marks with
`*` and `-`; a runtime that is both is shown as `*` in the text. The text's
next-step hint for an empty registry is not part of the JSON.

`provenance` is an object with the four facts the text prints on the runtime's
`provenance:` line, under the same names: `channel`, `source` (the package
index, else the downloaded archive), `source_layout_generation` and
`device_target`. Each is the value the manifest recorded, trimmed, or `null`
where the text prints `unrecorded`: a fact recorded nowhere or recorded blank,
the source of an adopted runtime, the exact target of a tarball or family-wide
install. A script reads `null` as "not recorded"; nothing is guessed from other
facts.

`registry`, `marker` and `install_root` are strings. A path that is not valid
UTF-8 is written with each invalid sequence replaced by U+FFFD, as the text
form prints it, so such a path cannot be passed back to the filesystem from the
JSON as it stands.

### Which runtimes are marked

`active` marks the runtime that `rocm examine`, `rocm runtimes activate`,
`rocm runtimes rollback`, `rocm runtimes uninstall` and `rocm update` treat as
the active one: the listing asks the resolver they use
(`current_runtime_manifest` in
[runtime_registry.rs](../apps/rocm/src/runtime_registry.rs)) rather than
applying rules of its own. That resolver takes the first runtime in list order
whose key matches `active_runtime_key` with ASCII letter case ignored; when
`active_runtime_key` is unset or names no runtime, it takes the one runtime
whose id matches `default_runtime_id`, letter case ignored, and nothing when
that matches none or several.

`rollback_target` marks the runtime `rocm runtimes rollback` selects:
`previous_runtime_key` resolved the way rollback resolves it
(`select_runtime_manifest`, same file), which ignores surrounding whitespace,
takes an exact key first, then the one key differing only in letter case, then
the one runtime whose id matches. When that selection fails (the value names
nothing, or matches several runtimes), no runtime is the rollback target. A
selected runtime whose `status` is `unusable` is still marked; rollback then
refuses it at the manifest check.

`active_problem` reports the configuration, by the same matching:

- `missing_manifest_for_key` when `active_runtime_key` is set and no runtime's
  key matches it, letter case ignored. This is the one case where a problem is
  reported beside an `active` runtime: when `default_runtime_id` still resolves
  one runtime, the commands fall back to it and the listing marks it active,
  while `active_problem` still says the configured key names nothing. This
  happens when the active runtime's registry entry is deleted outside the CLI
  while another registered runtime has the same id (activation records the
  active runtime's id as `default_runtime_id`, and installs of one channel and
  family share an id).
- With no `active_runtime_key`: `missing_manifest_for_id` when
  `default_runtime_id` matches no runtime's id, `ambiguous_runtime_id` when it
  matches several (ids differing only in letter case count as several), and
  `null` when it matches one, which is then the active runtime. `rocm config
  set-default-runtime` reaches this state: it stores the id as typed, clears
  `active_runtime_key` and `previous_runtime_key`, and tries to remove the
  active-runtime marker file (best effort: a failed removal is ignored).

Two runtimes can have keys that differ only in letter case on a
case-sensitive filesystem, since `rocm runtimes import` and
`rocm runtimes adopt --runtime-key` store keys as given. For such a pair the
resolver takes the first in list order even when `active_runtime_key` equals
the other exactly, and the listing marks that first one, because it is the one
those commands act on. Each registry entry is a file named `<key>.json`, so on
a case-insensitive filesystem (the default on Windows volumes, and on drvfs
mounts under WSL) the second key names the first one's file: the command
refuses it, or with `--replace` overwrites the first runtime's entry, and this
state cannot occur.

Not every surface resolves the active runtime this way. Known differences:

- The runtime manager in `rocm dash` matches exactly: `active_runtime_key`
  equal to a runtime's key, or with no `active_runtime_key` the single runtime
  whose id equals `default_runtime_id`, and `previous_runtime_key` equal to a
  runtime's key. It shows nothing active where the listing marks a runtime
  through letter case or through the default-id fallback after a dead key,
  shows no rollback target for a `previous_runtime_key` in other letter case,
  with surrounding whitespace or given as a runtime id. Of two keys differing
  only in letter case it marks the one equal to `active_runtime_key`, where the
  listing marks the first in list order, and of two ids differing only in case
  it marks the one equal to `default_runtime_id`, where the listing reports
  `ambiguous_runtime_id`.
- `rocm setup status` looks `active_runtime_key` up ignoring letter case but
  does not fall back to `default_runtime_id`.
- `rocm engines install` without `--runtime-id`, for an engine that runs on the
  ROCm runtime, resolves `active_runtime_key` (or, when it is unset,
  `default_runtime_id`) the way rollback resolves a key: a dead key fails the
  install instead of falling back, and a key matching case twins only by letter
  case is refused as ambiguous.

The shape is additive-only: fields may be added, none will be renamed or
removed without a versioned transition. It carries no version field yet; one
arrives with the envelope if that is adopted.

## Open decision: a shared envelope

Every command above prints its own top-level value (an object, or for
`rocm services list` an array). The mutating commands that
are still follow-ups need more than a value: whether the action ran, was
refused or failed, and why, in a shape a script can rely on across commands. A
shared envelope (for example a version, an outcome, the command's result and an
error object) would give them that.

It is not adopted. Wrapping the existing outputs in it would change documented
shapes, including the two frozen contracts, so it can only arrive alongside
them, under a versioned transition, and the decision belongs with the first
mutating command that needs it.
