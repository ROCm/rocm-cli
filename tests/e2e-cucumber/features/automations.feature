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

  # Autostart daemon singleton (EAI-7194): the first `automations enable` (or
  # `serve`) that finds no running daemon spawns one, but the child does not
  # publish its `running` runtime state until well after `spawn()` (clap parse,
  # runtime build, config load, banner flush). A second invocation that lands in
  # that spawn→publish window re-reads "not running" and — before the fix — spawned
  # a duplicate daemon that orphaned the first. The autostart claim (child PID +
  # spawn time, written under the lock before it drops) closes the window: a caller
  # that sees a live, recent claim defers instead of spawning. This plants an
  # in-flight claim (no published runtime state yet) and asserts the enable defers
  # rather than launching a second daemon. Config-only — no GPU or network — so it
  # runs on the mock lane every PR, which is why the race half of the PR can be
  # covered without hardware.
  @id:automations-autostart-defers-during-spawn-window @requires-os:linux
  Scenario: automations-04 - Enabling during an in-flight daemon spawn starts no second daemon
    Given a daemon spawn is already in flight
    When the user enables an automation watcher in observe mode
    Then the CLI does not start a second background daemon
