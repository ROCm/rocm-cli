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
- **Errors are not JSON today.** A command that fails with `--json` prints its
  error to stderr, prints nothing on stdout, and exits non-zero (1, unless the
  command documents its own code, as `rocm fix` does). A script tells success
  from failure by the exit code, not by parsing an error document. A JSON error
  shape belongs to the shared envelope below. The one refusal printed as JSON
  today is `rocm diagnose --report --json` declining to prepare a report: that
  is the command working, so it exits 0 and prints `refusal_envelope` from
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
| `rocm runtimes list` | supported; pretty JSON through `JsonOrText` | supported, additive-only, no version field yet | `runtime-lifecycle-19` to `runtime-lifecycle-23` and `runtimes_list_json_has_exactly_the_documented_fields_newest_install_first`; what each covers is in [testing.md](testing.md#runtime-selection-and-activation) |
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
| `active_runtime_id` | the configured default runtime id as written in the config, or `null`. Not necessarily the active runtime's `runtime_id`: `active_runtime_key` takes precedence when set, and the id may differ from its runtime's in letter case or match no runtime at all. Read each runtime's `active` flag to see which one is in use |
| `active_runtime_key` | the configured active runtime key, or `null` |
| `previous_runtime_key` | the rollback target's key, or `null` |
| `registry`, `marker` | paths of the runtime registry folder and the active-runtime marker |
| `active_problem` | `null`, or why no runtime is active (matched as described under the known limitation below): `{"kind": "missing_manifest_for_key", "runtime_key"}`, `{"kind": "missing_manifest_for_id", "runtime_id"}`, or `{"kind": "ambiguous_runtime_id", "runtime_id", "runtime_keys": [...]}` |
| `runtimes` | one object per registered runtime, newest install first, ties broken by `runtime_key` |

Each runtime carries `runtime_key`, `runtime_id`, `version` (as recorded; the
text adds the build date), `format`, `family`, `mode` (`managed` or
`read-only`), `status` (`ready` when the runtime passes the manifest check
that activation applies, `unusable` when it fails it; `ready` does not promise
that an activation succeeds), `unusable_reason` (`null` when ready, otherwise
why the runtime fails that check: text for a person, not a code to match on),
`toolchain` (`included` or `excluded`), `active`,
`rollback_target` and `install_root`. `active` and `rollback_target` are the facts the text marks with
`*` and `-`; a runtime that is both is shown as `*` in the text. The text's
next-step hint for an empty registry is not part of the JSON.

### Known limitation: the listing's matching rules

`rocm runtimes list` decides which runtime is active, which is the rollback
target and what `active_problem` says with rules of its own, which are not all
the ones `rocm examine`, `rocm runtimes activate` and `rocm runtimes rollback`
resolve with. Today:

- When `active_runtime_key` is set, `active` marks the runtime whose key
  equals it exactly, and `active_problem` is `missing_manifest_for_key` when
  no runtime's key equals it exactly. `default_runtime_id` is then not
  consulted.
- When no `active_runtime_key` is set, `default_runtime_id` is matched against
  each runtime's id with ASCII letter case ignored, by one rule for both
  fields: exactly one match is marked `active` with `active_problem` `null`; no
  match gives `missing_manifest_for_id`; several give `ambiguous_runtime_id`
  and nothing active. `rocm config set-default-runtime` reaches this state: it
  stores the id as typed, clears `active_runtime_key` and
  `previous_runtime_key`, and tries to remove the active-runtime marker file
  (best effort: a failed removal is ignored).
- `rollback_target` marks the runtime whose key equals `previous_runtime_key`
  exactly.

The commands acting on the active runtime also accept an `active_runtime_key`
that differs from a runtime's only in letter case, and fall back to
`default_runtime_id` when `active_runtime_key` names no runtime.
`rocm runtimes rollback` also ignores whitespace around `previous_runtime_key`
and accepts one that differs from a runtime's key in letter case or names a
runtime by its id. For the active runtime, the commands take the first
runtime in list order whose key matches `active_runtime_key` ignoring case,
even when a later one has that key exactly; rollback tries an exact match
first.

The CLI records the active and rollback keys exactly as the runtimes have
them, so most of these differences need a change made outside the CLI: a key
edited by hand in the config shows nothing active in the listing, or no
rollback target; and if the active runtime's registry entry is deleted while
exactly one other registered runtime has the same id as `default_runtime_id`
(activation records the active runtime's id there), the commands fall back to
that runtime while the listing reports `missing_manifest_for_key`. One
difference is reachable from the CLI alone, on a case-sensitive filesystem:
`rocm runtimes import` and `rocm runtimes adopt --runtime-key` store keys as
given, so two runtimes can have keys that differ only in letter case. When the
one activated by its exact key is not the first of the two in list order, the
listing marks it active while the commands act on the other. Each registry
entry is a file named `<key>.json`, so on a case-insensitive filesystem (the
default on Windows volumes, and on drvfs mounts under WSL) the second key names
the first one's file: the command refuses it, or with `--replace` overwrites
the first runtime's entry, and this state cannot occur.

`active` and `rollback_target` mean what the rules above say, and that meaning
is part of the documented shape. Changing those rules, for example to match the
resolution `rocm examine` uses, would change what the two flags report for the
same configuration, so it is a contract change under the versioning rule
above, not an additive one.

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
