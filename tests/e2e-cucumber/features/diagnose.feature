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

  # The other half of scenario 3, and the half every host can prove. A caller
  # cannot read "did anything match?" off the size of the list: every checker
  # that fires at all is reported, including ones scoring too low to act on,
  # and several open with a nonzero score for a situation that is merely
  # POTENTIALLY relevant — being in a container, having an APU beside a
  # discrete GPU. So a healthy machine hands back a non-empty list of things
  # that are not wrong with it. A caller treating that as a diagnosis proposes
  # a fix for a machine with nothing wrong, and never routes the user onward.
  @id:diagnose-json-states-when-nothing-matched
  Scenario: 9 - A tool is told plainly when no cause was established
    Given a user who hit a failure the CLI does not recognise
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then the result states that no cause was established
    And the CLI always points to somewhere the problem can be reported

  # Host-agnostic on purpose: the scenario asks the CLI what it makes of this
  # platform and then holds it to the matching half of the contract. A caller
  # decides whether to diagnose at all from this verdict, and nothing pinned it
  # before — the suite only ever SKIPPED the bare-metal scenarios on WSL2, which
  # proves nothing about what gets reported there.
  #
  # Be precise about where each half runs, because the halves are not equal.
  # There is NO WSL2 lane in CI (every job pins a native runner), so the
  # route-out half is proven only by a developer running the suite on WSL2.
  # What CI gets is the covered half plus the cross-check against the host
  # report — both of which can fail, which is the bar an assertion has to clear
  # to be worth writing. An earlier version of this scenario returned early on a
  # covered platform and asserted nothing at all on any lane CI runs.
  @id:diagnose-states-whether-the-platform-is-covered
  Scenario: 10 - A platform the catalog does not cover says so and routes onward
    Given a user who hit a known ROCm failure
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then the result says whether this platform is covered
    And a platform that is not covered is given no diagnosis
    And a platform that is covered gets a verdict that follows the evidence
    And the CLI always points to somewhere the problem can be reported

  # A fix that cannot run here is a different outcome from one that failed, and
  # from one the user declined — a caller that cannot tell them apart reports a
  # broken machine when the truth is "wrong operating system". The scenario
  # picks whichever catalog entry belongs to the OTHER platform, so it carries
  # the same weight on the Linux and Windows lanes.
  @id:fix-inapplicable-here-is-declined-not-attempted
  Scenario: 11 - A fix meant for another operating system is declined, not attempted
    Given a user who has chosen a fix meant for a different operating system
    When the user asks the CLI to apply that fix
    Then the CLI declines because the fix does not apply to this machine
    And nothing on the machine is changed

  # Scenario 4 proves the listing works; this proves it is COMPLETE. Which
  # failure modes exist, and which of them the CLI will carry out itself, are
  # part of the published contract rather than private detail — so a mode added
  # or removed is a change to what callers were promised, and it should not be
  # possible to make it quietly. This is deliberately the brittle test that
  # breaks when the catalog changes; that break is the notification. Do not
  # loosen it.
  @id:fix-catalog-is-complete
  Scenario: 12 - The CLI offers every fix its catalog documents
    When the user asks the CLI which fixes it offers
    Then every fix the catalog documents is listed
    And only the fixes the CLI can carry out itself are marked as such

  # vLLM runs on Linux and WSL, but not native Windows. This scenario is
  # GPU-independent: it supplies the captured startup error as symptom text and
  # proves the public diagnosis output preserves both branches of the remedy.
  @id:diagnose-vllm-oom-is-conditional @requires-os:linux
  Scenario: 13 - A vLLM startup OOM receives conditional remediation
    Given a user whose vLLM server ran out of GPU memory
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then the diagnosis identifies the vLLM startup OOM
    And the OOM remedy distinguishes a busy GPU from a model that does not fit
