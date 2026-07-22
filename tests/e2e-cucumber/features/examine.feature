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

  @id:examine-distinguishes-unmanaged-rocm @requires-gpu
  Scenario: 4 - System inspection distinguishes CLI-managed from pre-existing ROCm
    Given a machine with a ROCm install that was not set up by the CLI
    When the user inspects the system
    Then the inspection reports the install as pre-existing
    And the inspection suggests setting up a CLI-managed install

  @id:examine-detects-wsl @requires-wsl
  Scenario: 6 - System inspection recognizes a WSL host
    Given the CLI is running in WSL
    When the user inspects the system
    Then the inspection reports Linux as the operating system
    And the inspection reports that the host is WSL
