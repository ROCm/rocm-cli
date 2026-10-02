Feature: Runtime lifecycle state machine

  # `rocm runtimes activate/rollback/uninstall/import` move a runtime through its
  # registry state machine. Only install/adopt/list were covered before; the
  # activate/rollback/uninstall/import transitions were verified during the
  # walkthrough but unprotected. The scenarios plant read-only (externally-sourced)
  # runtimes in the isolated registry, so no SDK download or GPU is needed — they
  # run on the mock lane every PR. Related EAI-7404.

  # Also covers the `rocm runtimes rollback` recovery hint: absent on the first
  # activation (no previous runtime, rollback would hard-error), present once a
  # previous runtime exists to roll back to.
  @id:runtime-lifecycle-activate-records-previous
  Scenario: runtime-lifecycle-01 - Activating a runtime records where it changed from
    Given two registered runtimes and none active
    When the user activates the first runtime
    Then that runtime becomes active having changed from nothing
    When the user activates the second runtime
    Then that runtime becomes active having changed from the first

  @id:runtime-lifecycle-rollback-returns-to-previous
  Scenario: runtime-lifecycle-02 - Rolling back returns to the previously active runtime
    Given two registered runtimes with the second active after the first
    When the user rolls back
    Then the first runtime is active again

  @id:runtime-lifecycle-uninstall-keeps-external-folder
  Scenario: runtime-lifecycle-03 - Uninstalling an externally-sourced runtime keeps its folder
    Given a registered read-only runtime
    When the user uninstalls that runtime
    Then its registry entry is removed
    And its external folder is left in place

  @id:runtime-lifecycle-uninstall-requires-yes-noninteractive
  Scenario: runtime-lifecycle-04 - Uninstalling without --yes is refused outside a terminal
    Given a registered read-only runtime
    When the user tries to uninstall that runtime without confirming
    Then the CLI refuses and requires --yes

  @id:runtime-lifecycle-uninstall-dry-run
  Scenario: runtime-lifecycle-05 - Dry-running an uninstall makes no changes
    Given a registered read-only runtime
    When the user dry-runs an uninstall of that runtime
    Then the dry run reports the plan without confirming or changing anything

  @id:runtime-lifecycle-import-rejects-duplicate-unless-replacing
  Scenario: runtime-lifecycle-06 - Importing a runtime, then rejecting a duplicate unless replacing
    Given a runtime manifest to import
    When the user imports the runtime
    Then the runtime is registered as read-only
    When the user imports the same runtime again
    Then the CLI refuses because it already exists
    When the user imports it again allowing replacement
    Then the import succeeds

  # `rocm runtimes list` prefixes each row with `*`/`-`/a blank to mark active,
  # rollback-target, and neither, with nothing else on the page explaining what
  # they mean. This asserts the printed legend actually names both glyphs, so the
  # rendered marker and its explanation can't drift apart silently.
  @id:runtime-lifecycle-list-shows-marker-legend
  Scenario: runtime-lifecycle-07 - Listing runtimes explains the active and rollback markers
    Given two registered runtimes with the second active after the first
    When the user rolls back
    And the user lists the registered runtimes
    Then the listing explains the active and rollback markers
    And the first runtime is marked active
    And the second runtime is marked as the rollback target

  # Re-selecting the runtime you are already on changes nothing the user can
  # see, so it must not quietly take away the recovery path the previous
  # activation had just offered ("if this causes problems, run `rocm runtimes
  # rollback`"). It used to: the activation recomputed the rollback target from
  # the runtime being left, and when there was none it cleared the recorded one
  # instead of leaving it alone. The shortest command sequence that exposes this
  # is three activations in a row, which is why none of the hand-written
  # examples above caught it: every one- and two-command sequence conforms.
  @id:runtime-lifecycle-reactivating-keeps-rollback-target
  Scenario: runtime-lifecycle-08 - Re-selecting the runtime already in use keeps the rollback target
    Given two registered runtimes with the second active after the first
    When the user activates the second runtime again
    Then the second runtime is still the one in use
    And the first runtime is still the rollback target

  # A selector that matched installed runtime keys only by letter case used to
  # settle the tie silently, so the user could act on a runtime they never named.
  # It now refuses and names the candidates, and naming one exactly works. Two
  # keys that differ only in case are two registry files, so they can only
  # coexist on a case-sensitive filesystem: Linux-only.
  @id:runtime-lifecycle-case-twin-selector-refused @requires-os:linux
  Scenario: runtime-lifecycle-09 - A selector matching two runtimes only by letter case is refused with both named
    Given two registered runtimes whose keys differ only in letter case
    When the user activates a runtime with a selector that matches both only by letter case
    Then the CLI refuses and names both runtimes
    And no runtime is active
    When the user activates one of them by its exact key
    Then that exact runtime becomes active
