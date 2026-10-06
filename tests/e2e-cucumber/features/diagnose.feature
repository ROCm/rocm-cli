Feature: Diagnosing failures and listing fixes

  # `rocm diagnose` matches a symptom string against a closed catalog of known
  # ROCm/PyTorch/llama.cpp failure modes, and `rocm fix` lists or previews the
  # remediations. Both are black-box and GPU-independent (no serve, no download,
  # no mutation), so every scenario here runs on the mock lane / per-PR tier.
  #
  # The catalog is platform-gated (linux, windows and wsl each select their own
  # entries), so these scenarios do NOT assert a specific fix-id — the top match
  # is environment-dependent. They assert the SHAPE of a diagnosis (a scored match
  # with an id and a plan) and the query/refusal contracts.
  #
  # These two used to carry @requires-bare-metal, because WSL2 ran no catalog at
  # all and so had no premise for a match. WSL2 has its own entries now, and the
  # symptom-keyword checks that were always valid there run too, so both hold on
  # every supported platform and the tag is gone.
  @id:diagnose-matches-known-symptom
  Scenario: diagnose-01 - Diagnosing a recognised failure reports a likely cause and a fix
    Given a user who hit a known ROCm failure
    When the user asks the CLI to diagnose that symptom
    Then the CLI reports a likely cause with a suggested fix
    And every reported cause comes with a command that applies it
    And every reported cause states its remediation flags

  @id:diagnose-always-offers-a-way-forward
  Scenario: diagnose-02 - Diagnosing any failure always gives the user a way to escalate
    Given a user who hit a failure the CLI does not recognise
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then the CLI always points to somewhere the problem can be reported

  @id:diagnose-json-has-match-flag
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

  # This scenario exercises `rocm fix <id> --dry-run` (print_recipe's `Flags:`
  # line), not the `rocm diagnose` report itself -- see diagnose-01's "states
  # its remediation flags" step for the equivalent `flags:` line on that
  # surface.
  @id:diagnose-fix-dry-run-changes-nothing
  Scenario: diagnose-05 - Previewing a fix explains the change without making it
    Given a user who has chosen a known fix
    When the user previews that fix without applying it
    Then the CLI describes what the fix would change
    And nothing on the machine is changed
    And the preview states plainly that this fix is manual only

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
  #
  # @requires-bare-metal on top of that: the scenario needs a fix that both
  # applies here AND reaches the consent gate, and only fix-9 does that on a host
  # with nothing installed. fix-9 does not apply on WSL2 — a single device with
  # no topology cannot have an iGPU/dGPU collision — so there the run stops at
  # the wrong-platform refusal before the gate is ever reached. That is designed
  # behaviour, not a bug, so it is a skip rather than an xfail. The gate is
  # shared code and stays covered by the mock and Linux GPU lanes.
  @id:diagnose-fix-requires-agreement-before-changing-anything @requires-os:linux @requires-bare-metal
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
  # Every lane CI runs — mock, the GPU lanes, and the WSL2 lane on Strix Halo —
  # is a covered platform, so what CI proves is the covered half plus the
  # cross-check against the host report. Both of those can fail, which is the bar
  # an assertion has to clear to be worth writing. An earlier version of this
  # scenario returned early on a covered platform and asserted nothing at all.
  #
  # The uncovered half no longer means WSL2: that platform has its own entries
  # now. It means a host that is neither Linux, Windows nor WSL, which no lane
  # runs, so that half is exercised by the unit tests rather than here.
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
  #
  # @requires-bare-metal on top of that, same reasoning as diagnose-08:
  # `fix-4-render-group`'s `applies_on` does not include `wsl`, so on a WSL2
  # host the CLI refuses it as the wrong platform before ever invoking the
  # (faked) `usermod` — there is no command-failure branch to reach there.
  @id:diagnose-fix-command-failure-reported-on-stderr @requires-os:linux @requires-bare-metal
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
  #
  # @requires-bare-metal for the same reason as diagnose-08: `fix-9-igpu-dgpu`
  # does not apply on WSL2 (no per-device topology to collide over there), so
  # the run stops at the wrong-platform refusal before the confirmation prompt
  # is ever printed.
  @id:diagnose-fix-interactive-decline-reported @requires-os:linux @requires-bare-metal
  Scenario: diagnose-16 - Declining the confirmation prompt on a real terminal is reported the same way
    Given a user who has chosen a fix that would change the machine
    When the user is asked interactively to apply it and types no
    Then the CLI declines on the terminal and explains that it needs agreement
    And the file the fix would have changed is untouched

  # WSL2 reaches the GPU through /dev/dxg and the Windows host driver, so the
  # bare-metal questions — render group, /dev/kfd, modprobe amdgpu — have no
  # answer there and any finding naming one would be a false positive. This is
  # the guard on the platform split; it is what makes covering WSL2 safe rather
  # than merely louder. @requires-os:linux would not express it: WSL2 is linux.
  @id:diagnose-wsl-never-reports-bare-metal-causes @requires-wsl
  Scenario: diagnose-17 - A WSL machine is never given a bare-metal cause
    Given a user who hit a known ROCm failure
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then no reported cause is one that only exists on bare-metal Linux
    And the result says this platform is covered

  # The remedies for a WSL GPU problem mostly live on the Windows host or install
  # packages with sudo, so none of them are ones the CLI carries out. A caller
  # that could not tell "explained" from "attempted" would report a changed
  # machine when nothing was touched.
  @id:diagnose-wsl-fix-is-explained-not-attempted @requires-wsl
  Scenario: diagnose-18 - A WSL remedy is explained rather than carried out
    Given a user who has chosen a WSL remedy that belongs on the Windows host
    When the user asks the CLI to apply that fix
    Then the CLI explains the remedy instead of carrying it out
    And nothing on the machine is changed

  # `rocm diagnose` can be pointed at another machine — a WSL distribution, from
  # the Windows host. The dangerous failure is not an error, it is a SILENT
  # fallback: reporting on the local machine when the user asked about a
  # different one hands them a verdict about the wrong host, and nothing in the
  # output says so. This holds everywhere, because "that machine is not reachable
  # from here" is as true on Linux, where there is no wsl.exe at all, as it is on
  # a Windows host that has no such distribution.
  @id:diagnose-unreachable-machine-is-refused-not-substituted
  Scenario: diagnose-19 - Asking about a machine that cannot be reached is refused, not substituted
    Given a user who asks to diagnose a machine that does not exist
    When the user asks the CLI to diagnose that machine
    Then the CLI refuses and explains that it could not reach that machine
    And no diagnosis of this machine is reported

  # diagnose-05 only proves the manual/zero-optional-flags wording, because
  # PREVIEW_FIX_ID (fix-1-arch) needs none of sudo/reboot/re-login. The
  # sudo+re-login combination only exists on a fix gated to bare-metal Linux
  # (fix-4-render-group), so it needs its own scenario.
  #
  # Unlike diagnose-14, this one IS OS-gated. `print_recipe` still runs before
  # the fix's own platform gate (see `apply` in fix.rs), but the Flags: line it
  # prints comes from `class_here()`, which looks up the catalog entry for the
  # *running* host's OS. "AUTO" only renders where `fix-4-render-group` is
  # actually `Auto` -- bare-metal Linux. Everywhere else (`applies_on` has no
  # other member) `class_here()` falls back to PRINT-ONLY, the generic "this
  # fix does not apply here" answer diagnose-11 already covers -- not a second,
  # platform-specific behaviour worth asserting under this scenario's name.
  #
  # The step still asserts only the printed Flags: text, never the exit code --
  # `fix-4-render-group` gates its own dry-run on host state ($USER,
  # `usermod`/`sudo` on PATH), so unlike PREVIEW_FIX_ID its exit code is not
  # guaranteed to be 0 even on Linux.
  @id:diagnose-fix-preview-states-required-flags @requires-os:linux @requires-bare-metal
  Scenario: diagnose-20 - Previewing a fix that needs sudo and a re-login says so, and that it's auto-applicable here
    Given a user who has chosen a fix that needs sudo and a re-login
    When the user previews that fix without applying it
    Then the preview states that the fix requires sudo and a re-login
    And the preview states that the CLI can run it automatically
  # `--model` answers the question that comes before the other two: given this
  # machine and that model, will it run. The point is that it answers in seconds
  # and fetches nothing, so the user is not told by a download that failed.
  #
  # Host-agnostic in the same way diagnose-10 is, and for the same reason. The
  # verdict depends on what this machine can measure of its own GPU, which
  # differs per lane, so the scenario asks the CLI what it measured and then
  # holds it to the matching half of the contract. Each half can fail, which is
  # the bar an assertion has to clear: a lane that could not measure its GPU --
  # whether because there is no GPU at all, an engine this platform's gate
  # rules out, or a GPU whose memory the CLI cannot read -- exercises the
  # "told why, not that the model is incompatible" half, the GPU lanes the
  # "measured and it does not fit" half. What holds everywhere is that a model
  # no machine could serve is never called ready, and that asking costs no
  # download.
  @id:diagnose-model-too-large-is-refused-with-something-that-fits
  Scenario: diagnose-21 - A model this machine cannot serve is refused before anything is downloaded
    Given a user asking about a model no single machine could serve
    When the user asks the CLI whether that model would run, in machine-readable form
    Then the model is never reported as ready
    And a machine that measured its GPU is told the model will not run, and what would
    And a machine that could not measure its GPU is told why, rather than that the model is incompatible
    And the human-readable answer names what would run instead
    And no model weights were fetched

  # The other half of the verdict, and the one a user acts on: a model that does
  # fit has to say which engine would serve it, because that is what `rocm serve`
  # will pick and the user has no other way to know before starting it.
  @id:diagnose-model-that-fits-is-ready-and-names-the-engine
  Scenario: diagnose-22 - A model this machine can serve is reported ready, with the engine that would serve it
    Given a user asking about the smallest curated model
    When the user asks the CLI whether that model would run, in machine-readable form
    Then a machine with enough measured GPU memory is told the model is ready
    And the answer names the engine that would serve it
    And a machine that could not measure its GPU is told why, rather than that the model is incompatible

  # The failure this guards is not an error, it is a WRONG ANSWER that reads like
  # a real one. If a recipe catalog that cannot be read is scored as though it
  # had been, the user is told their machine cannot run a model when the truth is
  # that the CLI never found out what the model needs — and they go looking for
  # hardware they may already have. Deterministic on every lane: the catalog
  # source is pointed at a path that does not exist.
  @id:diagnose-model-unreachable-catalog-is-not-an-incompatible-model
  Scenario: diagnose-23 - A recipe catalog that cannot be read is not reported as an incompatible model
    Given a machine that cannot reach the model recipe catalog
    When the user asks the CLI whether that model would run, in machine-readable form
    Then the CLI reports that it could not determine the answer
    And the reason given is the unreachable catalog, not the model
    And nothing is claimed about whether the model fits this machine

  # A model outside the curated catalog is not a model this CLI has judged
  # incompatible -- it is one the CLI never had the metadata to judge at all.
  # Folding the two together would tell a user "this will not run" about a
  # model that might run fine, on the strength of nothing. Deterministic on
  # every lane: the catalog is read successfully, it simply carries no recipe
  # by this name.
  @id:diagnose-model-not-curated-is-undetermined-not-blocked
  Scenario: diagnose-24 - A model outside the curated catalog is undetermined, not blocked
    Given a user asking about a model the curated catalog does not carry
    When the user asks the CLI whether that model would run, in machine-readable form
    Then the CLI reports that it could not determine the answer
    And the reason given is that the model is not curated, not that it does not fit
    And nothing is claimed about whether the model fits this machine

  # `degraded` exists so a model that runs, but below what the recipe
  # recommends, is never folded into the same answer as one that will not run
  # at all -- a user who is about to accept slower loading deserves a different
  # word than one being turned away. The fixture recipe needs almost no GPU
  # memory (so it clears the fit check on any lane that measured a GPU) but
  # recommends more system RAM than any real test host has, so the RAM
  # softening is the only thing left to trigger. A machine with no GPU still
  # cannot serve it at all, so that half is asserted the same way diagnose-21
  # and diagnose-22 already do.
  @id:diagnose-model-below-recommended-ram-is-degraded-not-blocked
  Scenario: diagnose-25 - A model that runs below its recommended system RAM is degraded, not blocked
    Given a user asking about a model that recommends far more system RAM than this host has
    When the user asks the CLI whether that model would run, in machine-readable form
    Then a machine with enough measured GPU memory to run it is told the model is degraded
    And a machine that could not measure its GPU is told why, rather than that the model is incompatible

  # `--model` and `--distro` together used to be refused only when the probe
  # happened to come back looking remote, so the refusal tracked a derived
  # examination property rather than the flag itself. Keyed on the flag now:
  # the refusal fires before any probe runs at all, so this holds even with no
  # `wsl.exe` on PATH and no distribution installed -- every lane proves it,
  # not only a WSL host.
  @id:diagnose-model-with-distro-is-refused-not-answered-for-the-local-host
  Scenario: diagnose-26 - Asking --model with --distro is refused before answering for the wrong machine
    Given a user who asks --model together with --distro
    When the user asks the CLI to diagnose with both flags
    Then the CLI refuses and says --model answers for this machine, not the one --distro names
    And no model verdict is reported

  # One entry behaves differently depending on the machine: it persists the
  # change on Windows, and on Linux it only reports where the value is set,
  # because the code that would write it takes no options and never does.
  # The listing said "the CLI will run this" on both, so a user on Linux — and
  # an agent reading the same listing — was told a change was coming that never
  # came. Host-independent on purpose: the assertion is that the listing agrees
  # with the machine in front of it, whichever machine that is.
  @id:diagnose-fix-applicability-is-per-machine
  Scenario: diagnose-27 - A fix that only explains itself here is not advertised as one the CLI will run
    Given a fix the CLI carries out on one kind of machine and only explains on another
    When the user asks the CLI which fixes it offers
    Then that fix is shown as what it does on this machine

  # The other half of the same defect. This entry does have a fix and the CLI
  # will carry it out, but not until it is told which device to pin; asked
  # plainly it prints the query that identifies one and stops. It was marked as
  # a fix the CLI applies, so the report of a change that never happened looked
  # like success.
  # @requires-bare-metal because the entry under test is scoped to bare-metal
  # Linux and Windows. On WSL it is refused at the platform gate instead, which
  # is a different contract with its own scenario — and the right one, since the
  # catalog does not claim this remedy applies there.
  @id:diagnose-fix-needing-an-argument-says-so @requires-bare-metal
  Scenario: diagnose-28 - A fix that needs more information says what it needs and changes nothing
    Given a user who has chosen a fix that cannot run until it is told what to act on
    When the user asks the CLI to apply it without saying what to act on
    Then the CLI names what it still needs and reports no change

  # Nothing here sends a report -- transport does not exist yet -- so what these
  # two prove is the part that has to be right before it does: that the machine
  # can see exactly what would be published, and that asking produces either a
  # report or a stated refusal and never a silent send.
  #
  # Host-independent on purpose, and the branches land on different lanes. A
  # lane with an AMD GPU on the compatibility matrix exercises the prepared
  # report; a lane without one exercises the unreadable-architecture refusal,
  # which is the case the mock lane actually has. The WSL lane reaches neither:
  # `examine` returns before any GPU probe there, so it refuses because the
  # platform was never inspected, whatever hardware it holds. Saying "a lane
  # without an allowlisted GPU exercises the refusal" would be wrong for that
  # lane, and would record the guard as firing correctly when it fired for an
  # unrelated structural reason. Written so that whichever branch a lane
  # reaches is a real assertion rather than a skip.
  @id:diagnose-report-is-shown-and-not-sent
  Scenario: diagnose-29 - Asking what a report would say shows it and sends nothing
    When the user asks the CLI what a report would carry
    Then the CLI either shows the whole report or says why it will not prepare one
    And the CLI states that nothing has been sent

  # The rule this guards is that a report is assembled field by field, never by
  # copying a larger structure. The unit tests sweep for planted markers; this
  # asserts the same property against whatever this real machine happens to be,
  # which is the case a fixture cannot reproduce.
  @id:diagnose-report-carries-no-identifying-detail
  Scenario: diagnose-30 - What a report would carry never identifies the machine
    When the user asks the CLI what a report would carry in machine-readable form
    Then the answer names no user, no host, and no file path

  # `--send` promises the report is always read before its form is offered.
  # That promise only holds if asking for the form without asking to see the
  # report first is refused outright, before anything about this machine is
  # examined — so this is the same exit code any other argument mistake gets,
  # not a diagnosis outcome, and it is true on every host and every lane.
  @id:diagnose-send-without-report-is-refused
  Scenario: diagnose-31 - Asking the CLI for a way to send a report, without asking to see it first, is refused
    When the user asks the CLI for a way to send a report, without asking to see the report first
    Then the CLI refuses and explains that the report must be requested too

  # Forces the same headless shape a server or container presents: no display,
  # no forwarded display, no override asking for a browser anyway. Linux-only
  # because the CLI only reads the environment for this decision on Linux;
  # Windows and macOS always treat a user as present, so there is no
  # environment that forces this branch on those hosts.
  #
  # Host-independent beyond that, and for the same structural reason
  # diagnose-29 and diagnose-30 are: the WSL lane refuses before any GPU
  # probe, and most other lanes have no GPU on the compatibility matrix
  # either, so a report is prepared on some lanes and refused on others.
  # Written so whichever branch a lane reaches is a real assertion rather
  # than a skip.
  @id:diagnose-send-on-a-headless-machine-prints-instead-of-opening @requires-os:linux
  Scenario: diagnose-32 - Asking to send on a machine with no desktop prints the address and a link instead of starting a mail client
    When the user asks the CLI for a way to send a report, with no desktop available to open it on
    Then the CLI either shows the whole report or says why it will not prepare one
    And the CLI states that nothing has been sent
    And the CLI prints the address to mail and a link, and starts nothing

  # HIP compiles device code at run time through a library a machine can hold
  # more than one copy of. When the copy that loads belongs to a different
  # installation than the runtime, compilation fails with an error naming
  # neither. Both remedies — remove one stack, or reorder the search path — can
  # break a working Python environment, and which is right depends on which
  # stack the user means to keep. So the CLI states them and changes nothing.
  #
  # The conflict itself cannot be provoked here: the suite cannot install a
  # second ROCm stack, and the detection rule is proven by unit tests that build
  # the machine state directly. What this pins is the half that matters if the
  # entry ever stops being advisory — that asking for it changes nothing and
  # recommends neither option.
  #
  # `@requires-os:linux` because `fix-18-comgr-conflict` is registered for
  # `["linux", "wsl"]` (comgr and LD_LIBRARY_PATH are POSIX-loader concepts, not
  # Windows ones). Unlike diagnose-20's preview, this step applies the fix for
  # real, so it goes through the fix's own platform gate and would be refused
  # for the wrong reason -- "wrong OS", not "advisory" -- on a native Windows
  # lane. `@requires-os:linux` matches WSL2 too, which is where this fix does
  # apply.
  @id:diagnose-fix-comgr-conflict-is-advisory-only @requires-os:linux
  Scenario: diagnose-33 - The fix for a shadowed compilation library changes nothing and recommends nothing
    Given a user who has chosen the fix for a shadowed compilation library
    When the user asks the CLI to apply that fix
    Then the CLI explains that it will not make the change itself
    And the CLI offers both options without ranking them

  # vLLM runs on Linux and WSL, but not native Windows. This scenario is
  # GPU-independent: it supplies the captured startup error as symptom text and
  # proves the public diagnosis output preserves both branches of the remedy.
  @id:diagnose-vllm-oom-is-conditional @requires-os:linux
  Scenario: diagnose-34 - A vLLM startup OOM receives conditional remediation
    Given a user whose vLLM server ran out of GPU memory
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then the diagnosis identifies the vLLM startup OOM
    And the OOM remedy distinguishes a busy GPU from a model that does not fit
    # A report that names `rocm fix <id>` and a `rocm fix` that then refuses that
    # id leaves the user worse off than no fix path at all. The two halves are
    # selected by separate platform lists -- the checker's, and the recipe's
    # against the RUNNING os, where WSL2 is its own family -- so they can
    # disagree while each looks right alone. Every Linux lane runs this step; the
    # WSL one is where the two lists can differ.
    And the CLI can act on the fix the diagnosis named

  # `--symptom` is where a user pastes a raw terminal capture, and a terminal
  # capture is full of line advances that are not `\n`: a progress bar repaints
  # with `\r`, and curses- and `rich`-style progress UIs move down with `ESC E`,
  # `ESC D` or `CSI n B` (terminfo's `nel`, `ind` and `cud1`). If those are not
  # line boundaries the whole paste collapses into a single line, a `vllm`
  # mention anywhere in it anchors another engine's OOM, and the user is told
  # with high confidence that vLLM ran out of memory -- quoting the other
  # engine's error text back as the evidence for it. A confidently wrong cause is
  # worse than no cause, so this is pinned at the level the user sees it.
  @id:diagnose-vllm-oom-not-attributed-across-rendered-lines @requires-os:linux
  Scenario: diagnose-35 - Another engine's OOM is not blamed on a vLLM mention elsewhere in the paste
    Given a user who pasted a capture naming vLLM and another engine's OOM on separate rendered lines
    When the user asks the CLI to diagnose that symptom in machine-readable form
    Then no vLLM startup OOM is reported
