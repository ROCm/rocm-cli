Feature: Configuration mutations

  # `rocm config <...>` edits the CLI's own config file (JSON under the isolated
  # config dir) and prints a one-line confirmation. These behaviours were verified
  # correct during the walkthrough but had no scenario protecting them. They touch
  # only the config file — no GPU, no network — so they run on the mock lane every
  # PR, catching a regression in the config surface before the field does.

  @id:config-default-engine-set-and-cleared
  Scenario: config-01 - Setting and clearing the default engine is confirmed
    Given a fresh CLI configuration
    When the user sets the default engine
    Then the CLI confirms the default engine was set
    When the user clears the default engine
    Then the CLI confirms the default engine was cleared

  @id:config-default-runtime-set-and-cleared
  Scenario: config-02 - Setting and clearing the default runtime is confirmed
    Given a fresh CLI configuration
    When the user sets the default runtime
    Then the CLI confirms the default runtime was set
    When the user clears the default runtime
    Then the CLI confirms the default runtime was cleared

  @id:config-telemetry-mode-confirmed
  Scenario: config-03 - Choosing a telemetry mode is confirmed with its policy
    Given a fresh CLI configuration
    When the user turns telemetry off
    Then the CLI confirms the telemetry mode and states the policy

  @id:config-permissions-mode-confirmed
  Scenario: config-04 - Choosing a permissions mode is confirmed
    Given a fresh CLI configuration
    When the user selects a permissions mode
    Then the CLI confirms the permissions mode

  @id:config-set-engine-requires-target
  Scenario: config-05 - Configuring an engine requires a target
    Given a fresh CLI configuration
    When the user configures an engine without saying what to change
    Then the CLI refuses and explains a target is required
    When the user configures an engine with a runtime to use
    Then the CLI confirms the engine configuration was updated

  @id:config-provider-enable-disable
  Scenario: config-06 - Enabling and disabling a cloud provider is confirmed
    Given a fresh CLI configuration
    When the user enables a cloud provider
    Then the CLI confirms the provider is enabled for prompt sending
    When the user disables that provider
    Then the CLI confirms the provider is disabled

  @id:config-local-provider-not-toggleable
  Scenario: config-07 - The always-on local provider cannot be toggled as a cloud provider
    Given a fresh CLI configuration
    When the user tries to enable the local provider
    Then the CLI refuses and explains the local provider is always enabled

  # @requires-os:linux: the premise is a secure store that cannot save. The Linux
  # Secret Service is reached over D-Bus, which the step forces unreachable so the
  # save deterministically fails; the Windows/macOS credential stores are always
  # present and cannot be disabled the same way, so the failure premise only holds
  # on Linux. The no-echo property it verifies is the security-relevant contract.
  @id:config-provider-key-no-secret-storage @requires-os:linux
  Scenario: config-08 - Saving a provider key without secure storage fails without leaking the key
    Given a machine with no secure secret storage
    When the user saves a provider API key
    Then the CLI reports it could not save the key securely
    And the key value never appears in the output

  # A settings file the CLI cannot read must not be mistaken for "no settings".
  # It records where the user's runtimes live; guessing the default folder
  # instead makes them disappear from every answer, and later writes land in
  # the wrong place. The CLI stops, says which file, which folder it still
  # names, and how to recover, and leaves the file alone so nothing more is
  # lost. Two of the ways out it offers are then taken as printed — naming the
  # data folder it suggests, and moving the file aside — and what it says each
  # one does is checked. Repairing the file is not exercised here.
  @id:config-unreadable-settings-not-guessed
  Scenario: config-09 - A settings file the CLI cannot read stops it instead of hiding the user's runtimes
    Given the CLI's settings file was cut short just after it recorded a custom ROCm folder
    When the user checks the CLI version without naming a data folder
    Then the CLI refuses, naming the settings file, the recorded folder and how to recover
    And the settings file is left exactly as it was
    When the user opens the CLI with the data folder the refusal suggested
    Then the CLI keeps its data in the recorded folder, on default settings
    And the settings file is left exactly as it was
    When the user changes a setting with the data folder the refusal suggested
    Then the CLI refuses to change the setting
    And the settings file is left exactly as it was
    When the user moves the settings file aside and opens the CLI without naming a data folder
    Then the CLI keeps its data in its default folder, on default settings

  # Saving settings replaces config.json in one step, but must not change what
  # the user set up around it: a file only its owner may read stays that way,
  # and a settings file that is a link (a dotfiles checkout, say) stays a link
  # while the file it points at gets the new settings. These are Unix
  # permissions and links, so the scenarios run on Linux only.
  @id:config-save-keeps-a-private-settings-file-private @requires-os:linux
  Scenario: config-10 - Changing a setting keeps a private settings file private
    Given a settings file that only its owner can read
    When the user turns telemetry off
    Then the CLI confirms the telemetry mode and states the policy
    And the settings file is still readable only by its owner
    And the settings file records telemetry as off

  @id:config-save-keeps-a-linked-settings-file-linked @requires-os:linux
  Scenario: config-11 - Changing a setting through a linked settings file keeps the link
    Given a settings file that is a link to a file kept elsewhere
    When the user turns telemetry off
    Then the CLI confirms the telemetry mode and states the policy
    And the settings file is still a link to that file
    And the linked-to file records telemetry as off
