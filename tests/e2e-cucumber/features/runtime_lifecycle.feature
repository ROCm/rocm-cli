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

  # Activating used to close its report with one fixed sentence — "running
  # services keep their recorded runtime until they are restarted" — printed
  # whether or not a single local server existed, so it could neither name a
  # server that was really left behind nor stay quiet when none was. The report
  # is now derived from the on-disk service records: the same planted server is
  # silent (`services_on_previous_runtime: 0`) while the runtime it recorded is
  # the one being activated, and is named once activating moves past it.
  @id:runtime-lifecycle-activate-names-services-left-behind
  Scenario: runtime-lifecycle-08 - Activating names the local servers left on the previous runtime
    Given two registered runtimes and a local server recorded on the first
    When the user activates the first runtime
    Then the activation reports no local server left on a previous runtime
    When the user activates the second runtime
    Then the activation names the local server left on the first runtime
    And the activation does not print the old fixed note about running services

  # `--restart-services` stops and respawns live local servers, so it needs the
  # same explicit approval every other service mutation takes — and the refusal
  # has to land before the switch, or the user gets the new runtime without the
  # restarts they asked for in the same breath.
  @id:runtime-lifecycle-activate-restart-services-requires-yes
  Scenario: runtime-lifecycle-09 - Activating with --restart-services is refused without --yes
    Given two registered runtimes with the second active after the first
    When the user tries to activate the first runtime restarting services without confirming
    Then the CLI refuses the restart and the second runtime stays active

  # Full success path for `--restart-services --yes`: a running managed server is
  # stopped and respawned onto the runtime just activated, exits the command with
  # rc=0, and the report names it under `services_restarted` rather than
  # `services_on_previous_runtime`.
  #
  # The closing step is what makes this a test of the RESTART rather than of the
  # report. `restart_service_onto_runtime` re-pins the record before restarting,
  # because the restart rebuilds argv from the record on disk; transposing the
  # two brings the engine back up on the runtime it was already using and still
  # reports success, because the entry moves to `services_restarted` either way
  # and the endpoint answers either way. Re-activating reads the record back
  # through `refresh_from_engine_state`, which adopts the runtime the engine
  # really launched with — so the wrong runtime surfaces there and nowhere else.
  #
  # Why @requires-gpu: `restart_internal_managed_service` stops the engine process
  # and waits up to 45 s on a real HTTP readiness probe before returning, so no
  # mock engine or substitution hook exists that could satisfy it on the mock lane.
  # This scenario therefore only runs on the self-hosted GPU lanes defined in
  # `.github/workflows/e2e-selfhosted.yml`.
  #
  # Why @requires-engine:vllm: the premise is a server the activation can MOVE,
  # and only an engine whose runtime is a ROCm runtime can be moved. Lemonade
  # manages its own runtime, so `classify_service_runtime_state` exempts it by
  # design and it is never stale — on a lemonade-default host the report would
  # correctly say `services_on_previous_runtime: 0` and asserting a restart there
  # would be a guaranteed false failure. Same reasoning as `serve-11`.
  @id:runtime-lifecycle-activate-restart-services-succeeds @requires-gpu @requires-engine:vllm
  Scenario: runtime-lifecycle-10 - Activating with --restart-services --yes stops and respawns the running server
    Given a managed runtime is active
    And a model is being served on GPU
    And the running service is recorded on a different runtime
    When the user activates the current runtime restarting services with confirmation
    Then the activation exits 0 and names the service under services_restarted
    And the restarted service list is empty for services_on_previous_runtime
    And the model endpoint responds after the restart
    And re-activating that runtime reports the service already on it

  # The other half of this change: the live services are read BEFORE either of
  # the activation's two writes, so a services folder that cannot be read
  # refuses the switch while the previous runtime is still fully in place —
  # rather than failing partway and leaving the config and the marker naming
  # different runtimes. Unit tests cover the snapshot that undoes a half-applied
  # write; this covers the refusal a user actually meets, and that the runtime
  # they were on is still the one a serve would pick up.
  @id:runtime-lifecycle-activate-refused-when-services-unreadable
  Scenario: runtime-lifecycle-11 - An unreadable service record refuses the activation and changes nothing
    Given two registered runtimes and none active
    And the first runtime is active
    And the local service records cannot be read
    When the user tries to activate the second runtime
    Then the activation is refused and the first runtime is still active

  # `rollback` carries the same `--restart-services` / `--yes` pair as `activate`,
  # and the same ordering requirement: refuse before the switch, not after. It
  # had no scenario, so moving that guard below `rollback_runtime` — where the
  # equivalent call sat on `activate` before this change — would have switched
  # the runtime and then declined the restarts, with nothing failing.
  #
  # The refusal also has to name `rollback`, not `activate`. The hint is built
  # from a per-command string, so the copy-paste that names the wrong command is
  # exactly the error it exists to prevent.
  @id:runtime-lifecycle-rollback-restart-services-requires-yes
  Scenario: runtime-lifecycle-12 - Rolling back with --restart-services is refused without --yes
    Given two registered runtimes with the second active after the first
    When the user tries to roll back restarting services without confirming
    Then the CLI refuses the restart naming rollback and the second runtime stays active

  # `previous_runtime_key` is printed by `rocm runtimes list` and decides whether
  # `rocm runtimes rollback` works at all, so a no-op activation that discarded
  # it is observable — and the report's own note tells the user to run exactly
  # that command to move the servers left behind.
  @id:runtime-lifecycle-reactivate-keeps-rollback-target
  Scenario: runtime-lifecycle-13 - Re-activating the active runtime keeps the rollback target
    Given two registered runtimes with the second active after the first
    When the user activates the second runtime
    Then the first runtime is still the rollback target
    And rolling back still reaches the first runtime

  # The PR's headline claim — the activation is transactional across its two
  # writes — reaching the user. The config is written first and the marker
  # second, so a marker write that fails after a config write that succeeded is
  # the half-applied state the snapshot exists to undo. Unit tests drive
  # `activate_runtime` and read the marker back; this asserts the refusal a user
  # meets, and that the config was rolled back rather than left naming a runtime
  # the marker never recorded.
  @id:runtime-lifecycle-failed-marker-write-rolls-back
  Scenario: runtime-lifecycle-14 - A failed marker write rolls the activation back
    Given two registered runtimes and none active
    And the first runtime is active
    And the active runtime marker cannot be written
    When the user tries to activate the second runtime
    Then the activation is refused and the config still names the first runtime

  # `services_with_unrecorded_runtime:` is the one line of the report with no
  # lane behind it. A record that names no runtime says nothing about which
  # runtime its server loaded, so it must be counted apart rather than folded
  # into `services_on_previous_runtime` — which `runtime-lifecycle-08` cannot
  # catch, since it asserts that count is 0 while its own service matches.
  @id:runtime-lifecycle-counts-unrecorded-runtime-services
  Scenario: runtime-lifecycle-15 - A server recording no runtime is counted apart
    Given two registered runtimes and a local server recording no runtime
    When the user activates the second runtime
    Then the activation counts the server under services_with_unrecorded_runtime
