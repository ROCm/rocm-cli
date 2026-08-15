Feature: Driver installation planning

  # Driver plans are host-sensitive, but this scenario supplies the documented
  # WSL detection signal and uses --dry-run, so it exercises the released CLI
  # without privileged commands or host mutation.
  @id:driver-install-wsl-dry-run-plan @requires-os:linux
  Scenario: Previewing the WSL driver install produces an actionable packaged plan
    When the user previews driver installation with a WSL detection signal
    Then the driver plan is supported and mutating
    And the dry-run driver plan requires no approval and previews no execution
    And the driver plan does not direct the user to the removed WSL setup script

  @id:driver-install-wsl-review-requires-approval @requires-os:linux
  Scenario: Reviewing the WSL driver install requires approval before execution
    When the user reviews driver installation with a WSL detection signal without approval
    Then the unapproved WSL driver plan is actionable but not executed
    And the driver plan does not direct the user to the removed WSL setup script
