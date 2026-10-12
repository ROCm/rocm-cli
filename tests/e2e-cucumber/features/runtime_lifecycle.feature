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
  # The report also says nothing changed, rather than naming the retained rollback
  # target as the runtime the user came from.
  @id:runtime-lifecycle-reactivating-keeps-rollback-target
  Scenario: runtime-lifecycle-08 - Re-selecting the runtime already in use keeps the rollback target
    Given two registered runtimes with the second active after the first
    When the user activates the second runtime again
    Then the report says the runtime did not change
    And the second runtime is still the one in use
    And the first runtime is still the rollback target

  # A selector that matched installed runtime keys only by letter case used to
  # settle the tie silently, so the user could act on a runtime they never named.
  # It now refuses and names the candidates, and naming one exactly works. Two
  # keys that differ only in case are two registry files, so they can only
  # coexist on a case-sensitive filesystem — which Linux does not guarantee,
  # since the scenario's root follows $TMPDIR. Linux-only, and skipped where
  # that temp root folds case.
  @id:runtime-lifecycle-case-twin-selector-refused @requires-os:linux @requires-case-sensitive-fs
  Scenario: runtime-lifecycle-09 - A selector matching two runtimes only by letter case is refused with both named
    Given two registered runtimes whose keys differ only in letter case
    When the user activates a runtime with a selector that matches both only by letter case
    Then the CLI refuses and names both runtimes
    And no runtime is active
    When the user activates one of them by its exact key
    Then that exact runtime becomes active

  # The activation report is derived from the on-disk service records, not from
  # a fixed sentence about "running services" printed whether or not a single
  # local server existed: the same planted server is silent
  # (`services_on_previous_runtime: 0`) while the runtime it recorded is the one
  # being activated, and is named once activating moves past it.
  @id:runtime-lifecycle-activate-names-services-left-behind
  Scenario: runtime-lifecycle-10 - Activating names the local servers left on the previous runtime
    Given two registered runtimes and a local server recorded on the first
    When the user activates the first runtime
    Then the activation reports no local server left on a previous runtime
    When the user activates the second runtime
    Then the activation names the local server left on the first runtime
    And the activation does not print a fixed note about running services

  # `--restart-services` stops and respawns live local servers, so it needs the
  # same explicit approval every other service mutation takes — and the refusal
  # has to land before the switch, or the user gets the new runtime without the
  # restarts they asked for in the same breath.
  @id:runtime-lifecycle-activate-restart-services-requires-yes
  Scenario: runtime-lifecycle-11 - Activating with --restart-services is refused without --yes
    Given two registered runtimes with the second active after the first
    When the user tries to activate the first runtime restarting services without confirming
    Then the CLI refuses the restart and the second runtime stays active

  # Full success path for `--restart-services --yes`: a running managed server is
  # stopped and respawned onto the runtime just activated, exits the command with
  # rc=0, and the report lists it as restarted — under `services_restarted` once
  # it answers an inference request, under `services_restarted_not_yet_serving`
  # while the model is still loading, which the endpoint step then waits out —
  # rather than under `services_on_previous_runtime`.
  #
  # The closing steps re-activate the runtime, which reads the record back
  # through `refresh_from_engine_state` and so adopts the runtime the engine
  # really launched with, and assert the activation finds nothing left behind
  # and no server whose runtime went unrecorded — so a server that came back on
  # the wrong runtime is named there. What they do NOT do is police the
  # pin-before-restart order inside `restart_service_onto_runtime`: an engine
  # that resolved a runtime for itself reports the manifest's family id, and
  # `classify_service_runtime_state` resolves an unambiguous family id back to
  # the install that carries it — so with a single install of the family, which
  # is what this tree normally has, the transposed order reports `Matches` too.
  # That ordering is pinned instead by the unit test
  # `pinning_the_record_is_what_puts_the_new_runtime_in_the_restart_argv`,
  # which asserts the child's argv under both orders and needs no GPU. This
  # scenario's job is the user-visible half: rc=0, the report's counts, and a
  # server that answers afterwards.
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
  Scenario: runtime-lifecycle-12 - Activating with --restart-services --yes stops and respawns the running server
    Given a managed runtime is active
    And a model is being served on GPU
    And the running service is recorded on a different runtime
    When the user activates the current runtime restarting services with confirmation
    Then the activation exits 0 and reports the service as restarted
    And the activation reports no local server left on a previous runtime
    And the model endpoint responds after the restart
    When the user re-activates that runtime
    Then the activation reports no local server left on a previous runtime
    And the activation reports no local server with an unrecorded runtime

  # The live services are read BEFORE either of the activation's two writes, so
  # a services folder that cannot be read refuses the switch while the previous
  # runtime is still fully in place — rather than failing partway and leaving the
  # config and the marker naming different runtimes. The refusal says the switch
  # was not made. Unit tests cover the snapshot that undoes a half-applied write;
  # this covers the refusal a user actually meets, and that the runtime they were
  # on is still the one a serve would pick up.
  @id:runtime-lifecycle-activate-refused-when-services-unreadable
  Scenario: runtime-lifecycle-13 - An unreadable service record refuses the activation and changes nothing
    Given two registered runtimes and none active
    And the first runtime is active
    And the local service records cannot be read
    When the user tries to activate the second runtime
    Then the activation is refused, saying the switch was not made, and the first runtime is still active

  # `rollback` carries the same `--restart-services` / `--yes` pair as `activate`,
  # and the same ordering requirement: refuse before the switch, not after.
  # Moving that guard below `rollback_runtime` would switch the runtime and then
  # decline the restarts, so the scenario checks the runtime stayed where it was.
  #
  # The refusal also has to name `rollback`, not `activate`. The hint is built
  # from a per-command string, so the copy-paste that names the wrong command is
  # exactly the error it exists to prevent.
  @id:runtime-lifecycle-rollback-restart-services-requires-yes
  Scenario: runtime-lifecycle-14 - Rolling back with --restart-services is refused without --yes
    Given two registered runtimes with the second active after the first
    When the user tries to roll back restarting services without confirming
    Then the CLI refuses the restart naming rollback and the second runtime stays active

  # The PR's headline claim — the activation is transactional across its two
  # writes — reaching the user. The config is written first and the marker
  # second, so a marker write that fails after a config write that succeeded is
  # the half-applied state the snapshot exists to undo. Unit tests drive
  # `activate_runtime` and read the marker back; this asserts the refusal a user
  # meets, and that the config was rolled back rather than left naming a runtime
  # the marker never recorded.
  @id:runtime-lifecycle-failed-marker-write-rolls-back
  Scenario: runtime-lifecycle-15 - A failed marker write rolls the activation back
    Given two registered runtimes and none active
    And the first runtime is active
    And the active runtime marker cannot be written
    When the user tries to activate the second runtime
    Then the activation is refused and the config still names the first runtime

  # `services_with_unrecorded_runtime:` is the one line of the report with no
  # lane behind it. A record that names no runtime says nothing about which
  # runtime its server loaded, so it must be counted apart rather than folded
  # into `services_on_previous_runtime` — which `runtime-lifecycle-10` cannot
  # catch, since it asserts that count is 0 while its own service matches.
  @id:runtime-lifecycle-counts-unrecorded-runtime-services
  Scenario: runtime-lifecycle-16 - A server recording no runtime is counted apart
    Given two registered runtimes and a local server recording no runtime
    When the user activates the second runtime
    Then the activation counts the server under services_with_unrecorded_runtime

  # `--yes` approves a restart, so without `--restart-services` it has nothing to
  # approve. A silent no-op would let the user believe they had asked for a
  # restart they did not get, so both commands refuse it — which a user sees as
  # the exit code and the message, and which must leave the runtime alone.
  @id:runtime-lifecycle-yes-without-restart-services-refused
  Scenario: runtime-lifecycle-17 - --yes without --restart-services is refused on activate and rollback
    Given two registered runtimes with the second active after the first
    When the user tries to activate the first runtime with only --yes
    Then the CLI refuses the bare --yes and names --restart-services
    And the second runtime is still the one in use
    And the first runtime is still the rollback target
    When the user tries to roll back with only --yes
    Then the CLI refuses the bare --yes and names --restart-services
    And the second runtime is still the one in use
    And the first runtime is still the rollback target

  # A restart failure does not undo the switch, and it must not be reported as
  # a success or as a stopped server. A restart is refused before it stops
  # anything when the server is bound publicly with no endpoint key, so the
  # server is still up on the runtime it recorded: the command fails, the report
  # names the failure and still counts that server as left behind, the switch
  # stands, and the record is not rewritten to claim a runtime it never loaded.
  # Mock lane: the refusal comes before any engine is touched, so no GPU or
  # engine is needed. The restart that does stop and respawn a server is
  # `runtime-lifecycle-12`.
  @id:runtime-lifecycle-failed-service-restart-is-reported
  Scenario: runtime-lifecycle-18 - A refused server restart is reported and the server stays on its recorded runtime
    Given two registered runtimes and a publicly bound local server recorded on the first
    When the user activates the second runtime restarting services with confirmation
    Then the activation fails naming the server it could not restart
    And the server is still counted as left on the first runtime
    And the switch to the second runtime was kept
    And the server record still names the first runtime
