Feature: Driver installation planning

  # Driver plans are host-sensitive, but these scenarios supply the documented
  # WSL detection signal and use --dry-run, so they exercise the released CLI
  # without privileged commands or host mutation.
  @id:driver-install-wsl-dry-run-plan @requires-os:linux
  Scenario: driver-install-01 - Previewing the WSL driver install produces an actionable packaged plan
    When the user previews driver installation with a WSL detection signal
    Then the driver plan is supported and mutating
    And the dry-run driver plan requires no approval and previews no execution
    And the driver plan verifies the download before installing it
    And the driver plan does not direct the user to the removed WSL setup script

  @id:driver-install-wsl-review-requires-approval @requires-os:linux
  Scenario: driver-install-02 - Reviewing the WSL driver install requires approval before execution
    When the user reviews driver installation with a WSL detection signal without approval
    Then the unapproved WSL driver plan is actionable but not executed
    And the driver plan does not direct the user to the removed WSL setup script

  # The package is installed as root, so an unrecognised release must stop
  # rather than fall back to installing something nothing has authenticated.
  @id:driver-install-wsl-unverified-release-refused @requires-os:linux
  Scenario: driver-install-03 - Installing a ROCDXG release with no known digest is refused
    When the user previews driver installation for a ROCDXG release with no known digest
    Then the driver plan refuses rather than installing an unverified package
