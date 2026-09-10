Feature: Runtime configuration

  @id:runtime-install-sdk-active @requires-gpu @nightly
  Scenario: runtime-01 - Installing the SDK makes it the active runtime
    Given a machine with no CLI-managed runtimes
    When the user installs the SDK
    Then a runtime is registered
    And the runtime is set as active
    And the runtime includes an inference engine

  # Dogfooding #17: re-provisioning was observed writing inside the previous
  # runtime, producing a recursively nested `runtimes/wheel/.../runtimes/wheel/`
  # path that bloats paths and breaks `services/*.log` globs. Assert the active
  # runtime's folder path has no such recursive segment. GPU-gated (needs a real
  # install so the folder path is populated).
  @id:runtime-path-not-nested @requires-gpu
  Scenario: runtime-02 - The managed runtime path is not nested inside another runtime
    Given a managed runtime is active
    When the user inspects the system
    Then the managed runtime folder path is not recursively nested

  # The SDK and the engine share one Python environment and both write torch into
  # it, so a second `install sdk` could leave a torch that one of the two cannot
  # use. Every health surface still reported `ready` and the install still exited
  # 0 — the first signal was a serve failure naming neither. The runtime now
  # settles on the SDK's build of the release the engine pins, so this asserts the
  # outcome that actually matters rather than the wording of a check: can the
  # runtime still reach the GPU afterwards. Needs a real SDK install, a real engine
  # install, and a second SDK install, so it runs on the nightly GPU lane.
  # `@requires-engine:vllm` because only vLLM shares the runtime environment;
  # Lemonade manages its own.
  #
  # The second Then is not a restatement of the first. A runtime the alignment never
  # touched can still open a device, so the device check alone cannot distinguish
  # "settled correctly" from "skipped entirely" — and skipping is the regression the
  # gate in front of the settle step would produce. Only the alignment block
  # separates them, and it is the one part of this path with no other e2e coverage.
  # It reads the block's verdict rather than one string, because a torch that has
  # already run a GPU kernel with this SDK is kept instead of rewritten and reports
  # a `retained_*` verdict — settled, with nothing installed.
  @id:runtime-sdk-reinstall-keeps-engine-consistent @requires-gpu @requires-engine:vllm @nightly
  Scenario: runtime-03 - Reinstalling the SDK leaves the installed engine able to use the GPU
    Given a managed runtime with an inference engine already installed
    When the user installs the SDK again
    Then the runtime can still use the GPU
    And the torch alignment settled rather than being skipped

  # `ROCM_CLI_DISABLE_TORCH_ALIGNMENT` is the exit for the machine where the stack
  # the alignment settles on — the SDK's build of the release the engine pins —
  # does not work. That stack is not validated against the supported matrix, and
  # the alignment runs on every path that installs an engine, so without the
  # opt-out a torch the user installed deliberately is replaced again by the next
  # command and the only remaining exit is to stop using the CLI.
  #
  # Nothing asserted it from the user's side. The unit tests reach the gate
  # directly, and a gate that is honoured in isolation but bypassed by the install
  # path around it looks identical to a working one from every surface a user can
  # see. This is the same reinstall as scenario 4 with the opt-out set, so what
  # differs between them is exactly the variable.
  #
  # Four claims across three Thens, because the opt-out is only coherent if all
  # four hold: torch was not rewritten; the skip is reported as its own verdict
  # rather than folded into the generic `not_applicable`, which would leave the
  # user unable to tell whether the variable did anything; the divergence the
  # opt-out deliberately leaves behind is not then sold back to that user as a
  # runtime to repair by reinstalling the engine — an instruction that would undo
  # what they asked for; and the checks the opt-out does not suppress still run,
  # because it suppresses the correction, not the diagnosis.
  #
  # Same lane as scenario 4 and for the same reasons: a real SDK install and a
  # real engine, on the serialized nightly GPU runners. `@requires-engine:vllm`
  # because only vLLM shares the runtime environment the alignment writes into.
  @id:runtime-torch-alignment-opt-out @requires-gpu @requires-engine:vllm @nightly
  Scenario: runtime-04 - Opting out of the torch alignment keeps the torch the user installed
    Given a managed runtime with an inference engine already installed
    And the user has opted out of realigning torch
    When the user installs the SDK again
    Then the torch alignment reports the opt-out instead of rewriting torch
    And the install does not offer to reinstall the engine over the kept torch
    And the runtime's device health is still reported

  # The GPU E2E lanes no longer install the shared runtime once and keep it
  # forever: `xtask e2e-prewarm` asks `rocm update` whether the channel index has
  # published a newer version, and installs it side-by-side when it has (EAI-8057).
  # That makes CI depend on the freshness line this scenario pins. A unit test on a
  # hand-written fixture cannot catch the renderer drifting away from the parser —
  # only running the real command can, which is why this is a scenario and not just
  # an xtask test. Cheap enough for the per-PR lanes: one CLI call against the
  # already-installed shared runtime. `status=error` is an ACCEPTED outcome, so an
  # offline runner reports honestly instead of flaking.
  @id:runtime-update-reports-freshness @requires-gpu
  Scenario: runtime-05 - The update check reports the active runtime's freshness
    Given a managed runtime is active
    When the user checks for runtime updates
    Then the report states the runtime's freshness against the channel index

  # The install used to record the path it was handed rather than the folder the
  # files land in, so reaching `data/runtimes` through a link made the runtime name
  # a folder that disappeared with the link — taking every console-script shebang
  # in the environment with it, while the files stayed where they were written
  # (rocm-cli#315). The E2E harness itself creates exactly that link when a scenario
  # opts into the shared pre-warmed runtime, so the shared tree on a runner was the
  # thing being poisoned. Previewing the install is enough to pin this and needs no
  # GPU and no download: the planned folder is resolved before the preview prints
  # it, so a regression shows up in the plan. `--family` is supplied because
  # without a GPU there is no target to detect.
  #
  # `@nightly` is not about this scenario's own cost — it runs in about eight
  # seconds, nearly all of it resolving the channel index. It is that the no-GPU
  # mock lane runs 64 scenarios at once, and that much concurrent network work is
  # enough to push `eai-7960-gen-tps-held-after-scrape-failure` and
  # `eai-7960-gen-tps-expiry-boundary` past the validity window they assert on
  # (measured: both fail 3/3 with this scenario on the mock lane, and pass with the
  # very same scenario once the suite is serialized). Those two are timing-fragile
  # under load, which is their own problem to fix; until then this runs on the
  # nightly lanes, where a GPU is present and scenarios are serialized.
  @id:runtime-install-records-the-real-folder @nightly
  Scenario: runtime-06 - Previewing an install through a linked runtimes folder names the real folder
    Given a machine whose runtimes folder is a link to somewhere else
    When the user previews an SDK install
    Then the planned runtime folder is inside the folder the link points at
    And the planned runtime folder is not expressed through the link

  # Linux-only: the step adopts a standard `/opt/rocm` install with a Unix python
  # path. On Windows those paths don't exist (the CLI resolves `/usr/bin/python3`
  # to a bogus `C:/usr/bin/python3` and errors on the missing path before it can
  # emit the install-type guidance), so the scenario's premise doesn't hold there.
  @id:runtime-adopt-preexisting-rejected @requires-os:linux
  Scenario: runtime-07 - Adopting a pre-existing ROCm install is rejected with guidance
    Given a machine with a standard ROCm install
    When the user tries to adopt the existing install
    Then the adoption is refused
    And the error explains which install types can be adopted

  # Both channels now resolve from one canonical aggregate index each, so the
  # provenance the preview prints is the whole of what the user can check before
  # committing to a multi-GiB install. This pins the nightly half: the source that
  # was selected, the version that came back from it, and the layout generation
  # that source was read as. `--family` is supplied so the scenario needs no GPU,
  # and `@nightly` because it resolves the real index over the network — the same
  # cost that keeps scenario runtime-06 off the unserialized mock lane.
  @id:runtime-resolve-canonical-nightly @nightly
  Scenario: runtime-08 - Previewing a nightly SDK install reports canonical provenance
    When the user dry-runs a nightly SDK install for a known family
    Then the SDK preview reports canonical nightly provenance

  # The release channel is where this went wrong in the field (rocm-cli#271). The
  # resolver read a per-family index, `repo.amd.com/rocm/whl/{family}`, which is
  # frozen at 7.13.0 and has no device payloads at all, so every release install
  # got a stale SDK and a bare `rocm[libraries,devel]` — no GPU backend in it.
  # Both halves are fixed by the same canonical model: one flat aggregate index
  # for the channel, and the device payload for the chip this host actually has.
  #
  # The obvious regression test for that history is the one that does not work.
  # Asserting the broken per-family URL is *absent* passes with the bug present:
  # the old resolver only ever printed a candidate it failed on, and the stale
  # per-family index succeeded, so the URL never reached stdout either way. These
  # Thens assert the positives instead — which source was selected, which version
  # came back from it, which device payload the plan asks for. On the old resolver
  # the first reports a per-family URL and the second finds no device payload.
  #
  # The two Thens are not one claim split in half. A preview can name the right
  # source and still ask for the wrong payload — that is exactly what the earlier
  # attempt at this fix did on Instinct hosts, resolving the aggregate correctly
  # and then requesting every device wheel ROCm publishes. Only the second Then
  # separates them.
  #
  # `@requires-gfx-target` is narrower than `@requires-gpu`: this preview only
  # needs a detected chip name and never opens the device. It therefore runs on
  # WSL hosts that can read the Windows-side target before ROCm passthrough is
  # ready, while mock hosts with no target skip it.
  @id:runtime-resolve-canonical-release @requires-gfx-target
  Scenario: runtime-09 - Previewing a release SDK install resolves the canonical aggregate for this GPU
    Given a machine with an AMD GPU
    When the user dry-runs a release SDK install for this host
    Then the SDK preview reports canonical release provenance
    And the SDK preview requests the device payload for this host's GPU

  # Installing over the active default managed runtime must not silently
  # displace it. Outside an interactive terminal (as every e2e invocation
  # is here), `install sdk` without `--yes` must refuse rather than proceed.
  # GPU-gated because the precondition needs a GPU to have a runtime active.
  # The refusal is not free: the gate reports the version relation, so
  # it runs after the Python launcher is resolved and the channel index is read.
  # Both are already warm here — the `Given` installed a runtime, so the launcher
  # resolves to the saved managed Python rather than bootstrapping uv, and the
  # index read is cached — but on a cold host the launcher step can still fetch.
  # What the refusal does bail before is the SDK and torch download and any
  # change on disk.
  @id:runtime-install-sdk-overwrite-requires-yes @requires-gpu
  Scenario: runtime-10 - Reinstalling the SDK over an existing runtime without --yes is refused
    Given a managed runtime is active
    When the user reinstalls the SDK without confirming
    Then the reinstall is refused
    And the error explains that --yes is required

  # Companion to Scenario runtime-10: with --yes the same reinstall proceeds and the
  # runtime stays registered and active afterward. Nightly-gated in addition to
  # GPU because, unlike Scenario runtime-10, this exercises a real second SDK install.
  # The registered/active Thens hold from the Given alone, so the approval Then
  # is what actually distinguishes this from a no-op: it fails if --yes ever
  # regresses to a refusal or silently takes the fresh-install path.
  @id:runtime-install-sdk-overwrite-with-yes @requires-gpu @nightly
  Scenario: runtime-11 - Reinstalling the SDK over an existing runtime with --yes proceeds
    Given a managed runtime is active
    When the user reinstalls the SDK with --yes
    Then the install reports that --yes approved replacing the existing runtime
    And a runtime is registered
    And the runtime is set as active

  # The case a family-and-channel-scoped gate waved through. Activation is
  # global — whatever finishes installing last becomes the active default, no
  # matter which family it was built for — so installing a family this host has
  # never held displaces the active runtime exactly as a same-family reinstall
  # does, and has to ask exactly as loudly. Scenario runtime-10 cannot catch
  # this: it reinstalls the same family, so it passes under both the old
  # family-scoped gate and this one.
  #
  # No `@nightly` despite the second family: like Scenario runtime-10 this is a
  # refusal, so it bails before the multi-GiB download and costs a resolve, not
  # an install. The third Then is what separates a correct refusal from an
  # unrelated failure (a bad family name would also exit non-zero and also
  # mention `--yes` in the usage text): only the real gate names the runtime it
  # would replace.
  @id:runtime-install-sdk-other-family-requires-yes @requires-gpu
  Scenario: runtime-12 - Installing a different GPU family while a runtime is active is refused without --yes
    Given a managed runtime is active
    When the user installs a different GPU family without confirming
    Then the reinstall is refused
    And the error explains that --yes is required
    And the error names the active default runtime it would replace
