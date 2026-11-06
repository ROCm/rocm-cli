<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

- [ ] If this PR fixes a bug, searched `tests/e2e-cucumber/expectations.toml` for the fixed ticket ID and removed/narrowed any now-stale xfail rows.
- [ ] If this PR adds a new subcommand or subsystem, its domain implementation lives in its own file per `docs/architecture.md` (the clap declaration in `cli.rs` and the dispatch wiring in `main.rs`/`lib.rs` are expected, not a violation).
- [ ] If this PR adds, renames or removes a production source file, `docs/architecture.md` has the matching entry (`cargo xtask check-architecture-doc` fails otherwise), and the prose of any entry whose file changed role still describes it, which the check cannot verify.
- [ ] Every new or changed user-facing message was read against the code path that runs after it, and its test asserts the resulting state — not only the wording — per AGENTS.md §3. Covers claims of an outcome, remediation advice naming a command, and promises that something will *not* happen.
