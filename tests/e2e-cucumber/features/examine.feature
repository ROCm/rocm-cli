Feature: GPU detection and system inspection

  @id:examine-version
  Scenario: 1 - The CLI reports its version
    When the user asks for the version
    Then a version string is returned

  @id:examine-engines-list
  Scenario: 2 - The CLI lists all supported engines
    When the user lists available engines
    Then all supported engines are listed

  # Dogfooding #24: the `rocm help` subcommand list is in declaration order, not
  # alphabetical, which makes it harder to scan. Expected to FAIL until fixed —
  # surfaces the bug so it can be ticketed.
  @id:help-lists-subcommands-alphabetically
  Scenario: 5 - The help output lists subcommands in alphabetical order
    When the user asks for help
    Then the subcommands are listed in alphabetical order

  @id:examine-detects-gpu-and-driver @requires-gpu
  Scenario: 3 - System inspection detects the GPU and driver
    Given a machine with an AMD GPU
    When the user inspects the system
    Then the inspection reports which GPU is installed
    And the inspection reports that the driver is available

  # `examine` used to report a hardcoded platform constant as the default engine,
  # so on Instinct it named Lemonade while `serve` selected vLLM. The assertion is
  # host-agnostic: it compares what `examine` reports against the engine the
  # harness works out for this host from the GPU family and OS, so it resolves to
  # vLLM on Instinct and Lemonade on Strix Halo and on the no-GPU lane without
  # naming either. The harness derives its answer from the GPU probe rather than
  # from `examine`, so this is a cross-check and not a tautology.
  @id:examine-reports-host-default-engine
  Scenario: 6 - System inspection names the engine this machine serves on
    When the user inspects the system
    Then the inspection names the engine this host serves on by default

  # No GPU needed: the install is planted by the harness (`plant_unmanaged_rocm`,
  # written precisely so this does not depend on an ambient `/opt/rocm`), and
  # every assertion here is about how a detected install is reported, not about
  # hardware. Dropping `@requires-gpu` gains per-PR mock-lane coverage for the
  # reporting this scenario exists to pin.
  @id:examine-distinguishes-unmanaged-rocm
  Scenario: 4 - System inspection distinguishes CLI-managed from pre-existing ROCm
    Given a machine with a ROCm install that was not set up by the CLI
    When the user inspects the system
    Then the inspection reports the install as pre-existing
    And the inspection names that install's version
    And the inspection does not claim nothing is installed
    And the inspection suggests setting up a CLI-managed install
