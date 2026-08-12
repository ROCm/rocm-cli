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
