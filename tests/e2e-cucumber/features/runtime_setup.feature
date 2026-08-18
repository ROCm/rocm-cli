Feature: Runtime configuration

  # This is the acceptance path for the runtime-only default: the fresh install
  # must omit the compiler toolchain and still support vLLM's runtime compilation
  # and inference from that same isolated environment.
  @id:runtime-install-sdk-active @requires-gpu @requires-engine:vllm @nightly
  Scenario: 1 - A runtime-only SDK install supports vLLM inference
    Given a machine with no CLI-managed runtimes
    When the user installs the SDK
    Then a runtime is registered
    And the runtime is set as active
    And the runtime excludes the compiler toolchain
    When the user serves a model on GPU from the installed runtime
    And the user sends a chat completion request
    Then the response contains a model reply
    And the response identifies the correct model

  # Dogfooding #17: re-provisioning was observed writing inside the previous
  # runtime, producing a recursively nested `runtimes/wheel/.../runtimes/wheel/`
  # path that bloats paths and breaks `services/*.log` globs. Assert the active
  # runtime's folder path has no such recursive segment. GPU-gated (needs a real
  # install so the folder path is populated).
  @id:runtime-path-not-nested @requires-gpu
  Scenario: 3 - The managed runtime path is not nested inside another runtime
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
  Scenario: 4 - Reinstalling the SDK leaves the installed engine able to use the GPU
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
  Scenario: 9 - Opting out of the torch alignment keeps the torch the user installed
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
  Scenario: 5 - The update check reports the active runtime's freshness
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
  Scenario: 8 - Previewing an install through a linked runtimes folder names the real folder
    Given a machine whose runtimes folder is a link to somewhere else
    When the user previews an SDK install
    Then the planned runtime folder is inside the folder the link points at
    And the planned runtime folder is not expressed through the link

  # Linux-only: the step adopts a standard `/opt/rocm` install with a Unix python
  # path. On Windows those paths don't exist (the CLI resolves `/usr/bin/python3`
  # to a bogus `C:/usr/bin/python3` and errors on the missing path before it can
  # emit the install-type guidance), so the scenario's premise doesn't hold there.
  @id:runtime-adopt-preexisting-rejected @requires-os:linux
  Scenario: 2 - Adopting a pre-existing ROCm install is rejected with guidance
    Given a machine with a standard ROCm install
    When the user tries to adopt the existing install
    Then the adoption is refused
    And the error explains which install types can be adopted
