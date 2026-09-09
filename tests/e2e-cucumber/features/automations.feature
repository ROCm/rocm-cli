Feature: Automation watchers

  # `rocm automations enable/disable <watcher> --mode <mode>` toggles a built-in
  # automation watcher in the CLI config and prints a confirmation. This feature
  # pins only the enable/disable/mode-confirmation slice verified during the
  # walkthrough; the broader automation behaviour is covered elsewhere. Config-only
  # (no GPU, no network), so it runs on the mock lane every PR.

  @id:automations-enable-confirms-mode
  Scenario: automations-01 - Enabling a watcher confirms its mode
    Given a fresh CLI configuration
    When the user enables an automation watcher in observe mode
    Then the CLI confirms the watcher is enabled in observe mode
    When the user re-enables the same watcher in propose mode
    Then the CLI confirms the watcher is enabled in propose mode

  @id:automations-disable-confirmed
  Scenario: automations-02 - Disabling a watcher is confirmed
    Given an enabled automation watcher
    When the user disables the watcher
    Then the CLI confirms the watcher is disabled

  @id:automations-enable-unknown-refused
  Scenario: automations-03 - Enabling an unknown watcher is refused
    Given a fresh CLI configuration
    When the user tries to enable a watcher that does not exist
    Then the CLI refuses and names it as unknown
