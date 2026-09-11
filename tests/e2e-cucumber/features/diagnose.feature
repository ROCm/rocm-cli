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
  Scenario: diagnose-01 - Diagnosing a recognised failure reports a likely cause and a fix
    Given a user who hit a known ROCm failure
    When the user asks the CLI to diagnose that symptom
    Then the CLI reports a likely cause with a suggested fix
    And every reported cause comes with a command that applies it

  @id:diagnose-always-offers-a-way-forward
  Scenario: diagnose-02 - Diagnosing any failure always gives the user a way to escalate
    Given a user who hit a failure the CLI does not recognise
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then the CLI always points to somewhere the problem can be reported

  @id:diagnose-json-has-match-flag @requires-bare-metal
  Scenario: diagnose-03 - A diagnosis is available in machine-readable form for tooling
    Given a user who hit a known ROCm failure
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then the result is machine-readable and identifies the matched cause

  @id:diagnose-fix-lists-known-recipes
  Scenario: diagnose-04 - The user can see every fix the CLI knows how to apply
    When the user asks the CLI which fixes it offers
    Then the CLI lists the fixes it can apply
    And each fix indicates whether the CLI can apply it automatically
    And the listing explains what those indicators mean

  @id:diagnose-fix-dry-run-changes-nothing
  Scenario: diagnose-05 - Previewing a fix explains the change without making it
    Given a user who has chosen a known fix
    When the user previews that fix without applying it
    Then the CLI describes what the fix would change
    And nothing on the machine is changed

  @id:diagnose-fix-unknown-id-rejected
  Scenario: diagnose-06 - Asking for a fix the CLI does not know is refused clearly
    Given a user who names a fix the CLI does not offer
    When the user asks the CLI to apply that fix
    Then the CLI refuses and explains that the fix is not recognised

  # A diagnosis ranks causes `#1`, `#2`; reaching for that number here is the
  # natural mistake, and it used to get the same bare "unknown id" as a typo.
  @id:diagnose-fix-position-argument-rejected
  Scenario: diagnose-07 - Asking for a fix by its position in the diagnosis is corrected
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
  @id:diagnose-fix-requires-agreement-before-changing-anything @requires-os:linux
  Scenario: diagnose-08 - A fix that changes the machine is not applied without agreement
    Given a user who has chosen a fix that would change the machine
    When the user asks the CLI to apply it without agreeing to the change
    Then the CLI refuses and explains that it needs agreement
    And the file the fix would have changed is untouched

  # The other half of diagnose-03, and the half every host can prove. A caller
  # cannot read "did anything match?" off the size of the list: every checker
  # that fires at all is reported, including ones scoring too low to act on,
  # and several open with a nonzero score for a situation that is merely
  # POTENTIALLY relevant — being in a container, having an APU beside a
  # discrete GPU. So a healthy machine hands back a non-empty list of things
  # that are not wrong with it. A caller treating that as a diagnosis proposes
  # a fix for a machine with nothing wrong, and never routes the user onward.
  @id:diagnose-json-states-when-nothing-matched
  Scenario: diagnose-09 - A tool is told plainly when no cause was established
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
  Scenario: diagnose-10 - A platform the catalog does not cover says so and routes onward
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
  @id:diagnose-fix-inapplicable-here-is-declined-not-attempted
  Scenario: diagnose-11 - A fix meant for another operating system is declined, not attempted
    Given a user who has chosen a fix meant for a different operating system
    When the user asks the CLI to apply that fix
    Then the CLI declines because the fix does not apply to this machine
    And nothing on the machine is changed

  # diagnose-04 proves the listing works; this proves it is COMPLETE. Which
  # failure modes exist, and which of them the CLI will carry out itself, are
  # part of the published contract rather than private detail — so a mode added
  # or removed is a change to what callers were promised, and it should not be
  # possible to make it quietly. This is deliberately the brittle test that
  # breaks when the catalog changes; that break is the notification. Do not
  # loosen it.
  @id:diagnose-fix-catalog-is-complete
  Scenario: diagnose-12 - The CLI offers every fix its catalog documents
    When the user asks the CLI which fixes it offers
    Then every fix the catalog documents is listed
    And only the fixes the CLI can carry out itself are marked as such

  # This failure mode is reachable ONLY from the error text. The fact that
  # decides it is the torch version inside the managed runtime, which the host
  # examination does not read — so unlike every other entry there is no
  # structural signal to fall back on, and a symptom that does not score is a
  # symptom that gets the render-group false lead instead. That makes "the text
  # scores" the whole behaviour, which is why it is asserted directly here.
  #
  # The assertion is that the entry CLEARS the report's own threshold, not that
  # it ranks first: a runner with a real fault of its own (a blacklisted amdgpu)
  # legitimately scores higher for any symptom, so a ranking assertion would be
  # a test of the runner's health. Clearing the threshold comes from the keyword
  # alone and holds on every host.
  #
  # @requires-os:linux because the checker is registered linux-only, and
  # @requires-bare-metal because WSL2 does not run the catalog at all — the two
  # are not interchangeable, WSL2 reports an os_family of linux.
  @id:diagnose-recognises-the-engine-import-failure @requires-bare-metal @requires-os:linux
  Scenario: diagnose-13 - A vLLM engine-startup import failure is recognised from its error text
    Given a user who hit the vLLM engine-startup import failure
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then the CLI reports the engine-startup import failure as an established cause

  # Every other recipe in the catalog is a flat sequence for one shell. This one
  # is not: `rocm engines shell vllm` opens an INTERACTIVE subshell, the two
  # probes are meant to run inside it, and the reinstall replaces the very
  # environment that subshell is standing in, so it must not run there. The CLI
  # renders every command line with the same `$` prefix, so ordering alone said
  # none of that, and a user pasting the block wholesale was left depending on
  # terminal stdin buffering to land each line in the right shell. What the user
  # can observe is the printed plan, so that is what is asserted.
  #
  # Deliberately not OS-gated even though the fix is linux-only: the plan is
  # printed before the fix's own platform gate is reached, so the text under test
  # is identical on both lanes and the Windows lane exercises it too. The step
  # therefore asserts the printed block and not the exit code, which does differ
  # (0 where the fix applies, 3 where it does not).
  @id:diagnose-fix-says-which-shell-each-step-runs-in
  Scenario: diagnose-14 - A fix whose steps span two shells says which shell each step runs in
    Given a user who has chosen the fix for the engine-startup import failure
    When the user previews that fix without applying it
    Then the printed plan says which shell each step runs in

  # diagnose-08/-11 cover the refusal branches (no agreement, wrong OS); this
  # covers the third failure shape a fix can hit -- an approved, applicable fix
  # whose underlying command itself fails (e.g. `usermod` exiting non-zero).
  # Until now that branch of `fix-4-render-group` had no e2e coverage: a
  # regression could move the explanation back to stdout, or off exit code 4,
  # while every other listed scenario kept passing. Linux-only because the
  # recipe itself is `applies_on: LINUX_ONLY`.
  @id:diagnose-fix-command-failure-reported-on-stderr @requires-os:linux
  Scenario: diagnose-15 - A fix whose helper command fails explains why, on stderr, with exit code 4
    Given a user who has approved a fix whose helper command will fail
    When the user asks the CLI to apply the approved fix
    Then the CLI reports the command failure on stderr with exit code 4

  # diagnose-08 proves the non-interactive refusal (piped stdin, `is_terminal()`
  # false); this proves the sibling branch on a real terminal — the CLI must
  # print the confirmation prompt, read the typed answer, and, on anything but
  # y/yes, decline the same way. That branch has no piped-stdin equivalent: a
  # real TTY is required to reach it at all, so this is the one scenario in the
  # suite driven through the pseudo-terminal harness instead of piped stdin.
  # Linux-only for the same reason diagnose-08 is: the recipe under test
  # (`fix-9-igpu-dgpu`) only appends a shell rc file on Linux.
  @id:diagnose-fix-interactive-decline-reported @requires-os:linux
  Scenario: diagnose-16 - Declining the confirmation prompt on a real terminal is reported the same way
    Given a user who has chosen a fix that would change the machine
    When the user is asked interactively to apply it and types no
    Then the CLI declines on the terminal and explains that it needs agreement
    And the file the fix would have changed is untouched

  # vLLM runs on Linux and WSL, but not native Windows. This scenario is
  # GPU-independent: it supplies the captured startup error as symptom text and
  # proves the public diagnosis output preserves both branches of the remedy.
  @id:diagnose-vllm-oom-is-conditional @requires-os:linux
  Scenario: diagnose-17 - A vLLM startup OOM receives conditional remediation
    Given a user whose vLLM server ran out of GPU memory
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then the diagnosis identifies the vLLM startup OOM
    And the OOM remedy distinguishes a busy GPU from a model that does not fit
