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
