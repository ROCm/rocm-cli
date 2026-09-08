Feature: Runtime lifecycle state machine

  # `rocm runtimes activate/rollback/uninstall/import` move a runtime through its
  # registry state machine. Only install/adopt/list were covered before; the
  # activate/rollback/uninstall/import transitions were verified during the
  # walkthrough but unprotected. The scenarios plant read-only (externally-sourced)
  # runtimes in the isolated registry, so no SDK download or GPU is needed — they
  # run on the mock lane every PR. Related EAI-7404.

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

  @id:runtime-lifecycle-import-rejects-duplicate-unless-replacing
  Scenario: runtime-lifecycle-04 - Importing a runtime, then rejecting a duplicate unless replacing
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
  Scenario: runtime-lifecycle-05 - Listing runtimes explains the active and rollback markers
    Given two registered runtimes with the second active after the first
    When the user rolls back
    And the user lists the registered runtimes
    Then the listing explains the active and rollback markers
    And the first runtime is marked active
    And the second runtime is marked as the rollback target

  # `rollback` can only ever undo one step, not walk a history — rolling back
  # twice just re-toggles between the same two runtimes. A unit test on
  # `render_long_help()` proves clap renders the NOTE, but not that the built
  # binary prints it to a real user (examine.feature:15-18 sets this precedent
  # for `--help` text). No runtime state needed, so this runs on the mock lane.
  @id:runtime-lifecycle-rollback-help-states-single-level-limit
  Scenario: runtime-lifecycle-06 - Stating rollback's single-level limit in --help
    When the user asks for rollback help
    Then the help states that rollback has no history
