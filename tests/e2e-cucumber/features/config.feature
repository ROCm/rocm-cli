Feature: Configuration mutations

  # `rocm config <...>` edits the CLI's own config file (JSON under the isolated
  # config dir) and prints a one-line confirmation. These behaviours were verified
  # correct during the walkthrough but had no scenario protecting them. They touch
  # only the config file — no GPU, no network — so they run on the mock lane every
  # PR, catching a regression in the config surface before the field does.

  @id:config-default-engine-set-and-cleared
  Scenario: 1 - Setting and clearing the default engine is confirmed
    Given a fresh CLI configuration
    When the user sets the default engine
    Then the CLI confirms the default engine was set
    When the user clears the default engine
    Then the CLI confirms the default engine was cleared

  @id:config-default-runtime-set-and-cleared
  Scenario: 2 - Setting and clearing the default runtime is confirmed
    Given a fresh CLI configuration
    When the user sets the default runtime
    Then the CLI confirms the default runtime was set
    When the user clears the default runtime
    Then the CLI confirms the default runtime was cleared

  @id:config-telemetry-mode-confirmed
  Scenario: 3 - Choosing a telemetry mode is confirmed with its policy
    Given a fresh CLI configuration
    When the user turns telemetry off
    Then the CLI confirms the telemetry mode and states the policy

  @id:config-permissions-mode-confirmed
  Scenario: 4 - Choosing a permissions mode is confirmed
    Given a fresh CLI configuration
    When the user selects a permissions mode
    Then the CLI confirms the permissions mode

  @id:config-set-engine-requires-target
  Scenario: 5 - Configuring an engine requires a target
    Given a fresh CLI configuration
    When the user configures an engine without saying what to change
    Then the CLI refuses and explains a target is required
    When the user configures an engine with a runtime to use
    Then the CLI confirms the engine configuration was updated

  @id:config-provider-enable-disable
  Scenario: 6 - Enabling and disabling a cloud provider is confirmed
    Given a fresh CLI configuration
    When the user enables a cloud provider
    Then the CLI confirms the provider is enabled for prompt sending
    When the user disables that provider
    Then the CLI confirms the provider is disabled

  @id:config-local-provider-not-toggleable
  Scenario: 7 - The always-on local provider cannot be toggled as a cloud provider
    Given a fresh CLI configuration
    When the user tries to enable the local provider
    Then the CLI refuses and explains the local provider is always enabled

  # @requires-os:linux: the premise is a secure store that cannot save. The Linux
  # Secret Service is reached over D-Bus, which the step forces unreachable so the
  # save deterministically fails; the Windows/macOS credential stores are always
  # present and cannot be disabled the same way, so the failure premise only holds
  # on Linux. The no-echo property it verifies is the security-relevant contract.
  @id:config-provider-key-no-secret-storage @requires-os:linux
  Scenario: 8 - Saving a provider key without secure storage fails without leaking the key
    Given a machine with no secure secret storage
    When the user saves a provider API key
    Then the CLI reports it could not save the key securely
    And the key value never appears in the output
