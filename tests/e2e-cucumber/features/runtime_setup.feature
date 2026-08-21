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

  # Regression guard for the bug where the release-channel multi-arch pip index
  # was queried with a per-family path segment (.../whl-multi-arch/{family}/),
  # which 403s because that index is flat and 404/403-ed straight into the
  # stale classic index every time — silently pinning every install to
  # whatever version predated the migration. `--family` bypasses GPU
  # auto-detection so this needs no GPU, and `--dry-run` resolves the real
  # index without installing anything.
  @id:runtime-install-sdk-release-index-shape
  Scenario: 4 - Resolving the SDK from the release channel never uses the broken multi-arch URL shape
    When the user dry-runs installing the SDK for a known family
    Then the resolved package index is not the broken per-family multi-arch path
