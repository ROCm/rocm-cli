Feature: Automation watchers

  # `rocm automations enable/disable <watcher> --mode <mode>` toggles a built-in
  # automation watcher in the CLI config and prints a confirmation. This feature
  # pins only the enable/disable/mode-confirmation slice verified during the
  # walkthrough; the broader automation behaviour is covered elsewhere. Config-only
  # (no GPU, no network), so it runs on the mock lane every PR.

  @id:automations-enable-confirms-mode
  Scenario: 1 - Enabling a watcher confirms its mode
    Given a fresh CLI configuration
    When the user enables an automation watcher in observe mode
    Then the CLI confirms the watcher is enabled in observe mode
    When the user re-enables the same watcher in propose mode
    Then the CLI confirms the watcher is enabled in propose mode

  @id:automations-disable-confirmed
  Scenario: 2 - Disabling a watcher is confirmed
    Given an enabled automation watcher
    When the user disables the watcher
    Then the CLI confirms the watcher is disabled

  @id:automations-enable-unknown-refused
  Scenario: 3 - Enabling an unknown watcher is refused
    Given a fresh CLI configuration
    When the user tries to enable a watcher that does not exist
    Then the CLI refuses and names it as unknown

  # Scenarios 1-3 enable a watcher by an id the test already knows. This one pins
  # the complementary contract: the listing is the only place a user learns which
  # background checks exist, so whatever it shows has to be enough to act on.
  # Deriving each check's identifier from the listing the way a reader would IS
  # the contract under test — the identifiers are deliberately not written into
  # the scenario or the steps, because hard-coding them would pass against a
  # listing that exposes nothing. No fixtures, no GPU, no network.
  @id:automations-listed-checks-can-be-enabled
  Scenario: 4 - Every background check that is listed can be turned on
    Given a machine with no background checks turned on
    When the user lists the background checks
    Then every listed check can be turned on by name
