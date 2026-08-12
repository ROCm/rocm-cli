Feature: Diagnosing failures and listing fixes

  # `rocm diagnose` matches a symptom string against a closed catalog of known
  # ROCm/PyTorch/llama.cpp failure modes, and `rocm fix` lists or previews the
  # remediations. Both are black-box and GPU-independent (no serve, no download,
  # no mutation), so every scenario here runs on the mock lane / per-PR tier.
  #
  # The catalog is OS-gated (the checkers only run on linux/windows), so these
  # scenarios do NOT assert a specific fix-id — the top match is environment-
  # dependent. They assert the SHAPE of a diagnosis (a scored match with an id
  # and a plan) and the query/refusal contracts.

  # @requires-bare-metal: these two need the catalog to actually produce a match.
  # On WSL2 the catalog is deliberately not run at all — that platform uses
  # /dev/dxg and the Windows host driver, so bare-metal Linux diagnoses would be
  # false positives — which leaves these scenarios with no premise there. That is
  # designed behaviour with its own unit test, not a bug, so they are skipped
  # rather than xfail'd. `@requires-os:linux` would not do it: WSL2 is linux.
  @id:diagnose-matches-known-symptom @requires-bare-metal
  Scenario: 1 - Diagnosing a recognised failure reports a likely cause and a fix
    Given a user who hit a known ROCm failure
    When the user asks the CLI to diagnose that symptom
    Then the CLI reports a likely cause with a suggested fix
    And every reported cause comes with a command that applies it

  @id:diagnose-always-offers-a-way-forward
  Scenario: 2 - Diagnosing any failure always gives the user a way to escalate
    Given a user who hit a failure the CLI does not recognise
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then the CLI always points to somewhere the problem can be reported

  @id:diagnose-json-has-match-flag @requires-bare-metal
  Scenario: 3 - A diagnosis is available in machine-readable form for tooling
    Given a user who hit a known ROCm failure
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then the result is machine-readable and identifies the matched cause

  @id:fix-lists-known-recipes
  Scenario: 4 - The user can see every fix the CLI knows how to apply
    When the user asks the CLI which fixes it offers
    Then the CLI lists the fixes it can apply
    And each fix indicates whether the CLI can apply it automatically
    And the listing explains what those indicators mean

  @id:fix-dry-run-changes-nothing
  Scenario: 5 - Previewing a fix explains the change without making it
    Given a user who has chosen a known fix
    When the user previews that fix without applying it
    Then the CLI describes what the fix would change
    And nothing on the machine is changed

  @id:fix-unknown-id-rejected
  Scenario: 6 - Asking for a fix the CLI does not know is refused clearly
    Given a user who names a fix the CLI does not offer
    When the user asks the CLI to apply that fix
    Then the CLI refuses and explains that the fix is not recognised

  # A diagnosis ranks causes `#1`, `#2`; reaching for that number here is the
  # natural mistake, and it used to get the same bare "unknown id" as a typo.
  @id:fix-position-argument-rejected
  Scenario: 7 - Asking for a fix by its position in the diagnosis is corrected
    Given a user who refers to a cause by its position in the diagnosis
    When the user asks the CLI to apply that fix
    Then the CLI refuses and explains that a position is not a fix-id

  # The one gate standing between `rocm fix` and an edited machine, and until now
  # it had no end-to-end coverage. The scenario gives the CLI a home directory it
  # owns, so the file the fix would edit is one the scenario can read back: the
  # refusal must not depend on what is in the runner's dotfiles, and a regression
  # here must not be able to reach them.
  # Linux-only because the assertion is "the file is untouched": on Windows the
  # same recipe persists through `setx` into the user environment, which the
  # suite cannot plant or read back safely. The gate itself is shared code, so
  # this still guards it — just not the Windows persistence step.
  @id:fix-requires-agreement-before-changing-anything @requires-os:linux
  Scenario: 8 - A fix that changes the machine is not applied without agreement
    Given a user who has chosen a fix that would change the machine
    When the user asks the CLI to apply it without agreeing to the change
    Then the CLI refuses and explains that it needs agreement
    And the file the fix would have changed is untouched

  # Expected to FAIL. `diagnose` and `fix` are two views of the same remedy: the
  # first tells the user what will put the machine right, while the second is the
  # command they are told to run. They must not leave the user with two different
  # definitions of what proves the remedy worked.
  @id:diagnose-and-fix-agree-on-how-to-verify @requires-os:linux
  Scenario: 9 - Diagnosing a problem and previewing its fix agree on how to verify it
    Given a user who hit a device-permission failure
    When the user compares the diagnosis with the matching fix preview
    Then both give the same way to verify that the fix worked

  # Expected to FAIL on a bare-metal GPU host. When the device belongs to a
  # group that is not named in the local group database, the diagnosis prints
  # that lookup failure as if it were a group the user could join. A proposed
  # remedy has to name a group that actually exists on the machine.
  @id:diagnose-commands-name-a-real-group @requires-gpu @requires-bare-metal @requires-os:linux
  Scenario: 10 - Every group named in a diagnosis is one the machine recognises
    Given the GPU device belongs to a group the machine cannot name
    When the user asks the CLI to diagnose a device-permission failure
    Then every group named in the remedy exists on the machine

  # Expected to FAIL on a bare-metal GPU host whose device group is not the
  # hard-coded default. The diagnosis promises one command and its matching fix
  # previews another. This is distinct from scenario 9: even after the verify
  # text agrees, the actual change must agree too.
  @id:diagnose-and-fix-agree-on-the-remedy-command @requires-gpu @requires-bare-metal @requires-os:linux
  Scenario: 11 - Diagnosing a problem and previewing its fix agree on the remedy command
    Given the GPU device belongs to a recognised non-default group
    When the user compares the diagnosis with the matching fix preview
    Then both give the same command for applying the remedy

  # Expected to FAIL on a bare-metal GPU host where direct access already works.
  # Membership in one conventional group is only a means to access the device,
  # not the outcome. A diagnosis should credit the observable access the user
  # has instead of recommending a permission repair for a usable device.
  @id:diagnose-credits-a-usable-device @requires-gpu @requires-bare-metal @requires-os:linux
  Scenario: 12 - A usable GPU device is not diagnosed as a permission failure
    Given the user can already read and write the GPU device
    When the user asks the CLI to diagnose the machine
    Then adding the user to a device group is not the leading remedy
