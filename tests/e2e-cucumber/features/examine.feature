Feature: GPU detection and system inspection

  @id:examine-version
  Scenario: 1 - The CLI reports its version
    When the user asks for the version through every CLI surface
    Then matching traceable version strings are returned

  @id:examine-engines-list
  Scenario: 2 - The CLI lists all supported engines
    When the user lists available engines
    Then all supported engines are listed

  # EAI-7383: keep the top-level command list alphabetized so it remains easy to
  # scan as commands are added or reordered in the source declaration.
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

  # The machine-readable form is a separate code path, not a re-rendering of the
  # human one: it used to answer before the CLI had loaded its paths or config,
  # putting every CLI-side fact out of reach. Eleven things the human report
  # states had no field in it at all — among them which engine this host will
  # serve on and whether an existing ROCm install was found. Since fixed (those
  # facts now travel under `summary`), and this is what holds the two forms
  # level: tooling reads this one, and it must not drift back into being the
  # weaker of the two.
  @id:examine-machine-readable-report
  Scenario: 7 - What the inspection tells a tool matches what it tells a person
    When the user inspects the system both for reading and for scripting
    Then the machine-readable form states everything the readable one does

  # The harness parses the human text rather than this form because of a defect
  # this scenario caught, and says so in capability.rs — on a real MI300X the
  # machine-readable form reported no AMD GPU on a machine that has one, while
  # Strix Halo (gfx1151) agreed. That workaround makes the disagreement
  # load-bearing: every host capability the suite resolves comes from scraped
  # text, so if the two forms ever diverge again, every capability-keyed
  # expectation silently resolves against the wrong host. Since fixed; this is
  # the guard that keeps it fixed.
  @id:examine-both-forms-agree-on-gpu
  Scenario: 8 - Both forms of the inspection agree about the GPU
    When the user inspects the system both for reading and for scripting
    Then both reports agree on whether this machine has an AMD GPU
    And both reports agree on whether this platform is in scope

  # `examine` is an inspector: the outcome says whether it managed to look, not
  # whether it liked what it saw. Finding no GPU is a finding, not a failure.
  # This is the mock lane's to prove — it is the one lane with nothing to find.
  @id:examine-reports-without-failing
  Scenario: 9 - Inspecting a machine reports what it finds without failing
    When the user inspects the system
    Then the inspection completes successfully
    And it states a verdict for this machine

  # Leaving the frameworks out is the variant pinned here: the outcome is
  # identical on every host, whereas asserting that a *named* framework was
  # probed would depend on what happens to be installed.
  #
  # @requires-bare-metal because the probe never reaches its framework step on
  # WSL2 — it returns as soon as it recognises the platform — so `framework`
  # stays "unknown" there whatever the flag says. That the probe gives up that
  # early is its own defect, tracked separately; this scenario is about whether
  # the choice is reachable, and it cannot answer that where no choice is acted
  # on at all.
  @id:examine-can-skip-framework-probing @requires-bare-metal
  Scenario: 10 - The user can leave the frameworks out of the inspection
    When the user inspects the system without probing frameworks
    Then the inspection reports that it skipped the frameworks
    And it still states a verdict for this machine
