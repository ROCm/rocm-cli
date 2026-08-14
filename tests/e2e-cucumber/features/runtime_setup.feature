Feature: Runtime configuration

  @id:runtime-install-sdk-active @requires-gpu @nightly
  Scenario: 1 - Installing the SDK makes it the active runtime
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
  Scenario: 3 - The managed runtime path is not nested inside another runtime
    Given a managed runtime is active
    When the user inspects the system
    Then the managed runtime folder path is not recursively nested

  # The SDK and the engine share one Python environment, so a second
  # `install sdk` wrote the SDK's torch stack over the build the engine pins. The
  # engine still resolved, so the install reported success and every health surface
  # kept saying `ready` — the first signal was a serve failure naming neither. Needs
  # a real SDK install, a real engine install, and a second SDK install, so it runs
  # on the nightly GPU lane. `@requires-engine:vllm` because only vLLM shares the
  # runtime environment; Lemonade manages its own.
  @id:runtime-sdk-reinstall-keeps-engine-consistent @requires-gpu @requires-engine:vllm @nightly
  Scenario: 4 - Reinstalling the SDK leaves the installed engine's requirements satisfied
    Given a managed runtime with an inference engine already installed
    When the user installs the SDK again
    Then the install reports the engine's requirements as satisfied
    And the engine is still ready to serve

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
