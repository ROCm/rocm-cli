Feature: Driver installation planning

  # These use --dry-run and an unapproved review, so they exercise the released
  # CLI without privileged commands or host mutation.
  #
  # The plan branch under test is chosen by `is_wsl_host()`, which trusts only
  # `/dev/dxg` and the kernel build string in `/proc/version`, never
  # `$WSL_DISTRO_NAME`. The WSL2 machine is simulated with exactly those, so
  # these run on every Linux lane, not only the WSL one.
  @id:driver-install-wsl-dry-run-plan @requires-os:linux
  Scenario: driver-install-01 - Previewing the WSL driver install produces an actionable packaged plan
    Given a WSL machine with an AMD GPU passed through
    When the user previews driver installation on this WSL host
    Then the driver plan is supported and mutating
    And the dry-run driver plan requires no approval and previews no execution
    And the driver plan verifies the download before installing it
    And the driver plan does not direct the user to the removed WSL setup script

  @id:driver-install-wsl-review-requires-approval @requires-os:linux
  Scenario: driver-install-02 - Reviewing the WSL driver install requires approval before execution
    Given a WSL machine with an AMD GPU passed through
    When the user reviews driver installation on this WSL host without approval
    Then the unapproved WSL driver plan is actionable but not executed
    And the driver plan does not direct the user to the removed WSL setup script

  # The package is installed as root, so an unrecognised release must stop
  # rather than fall back to installing something nothing has authenticated.
  @id:driver-install-wsl-unverified-release-refused @requires-os:linux
  Scenario: driver-install-03 - Installing a ROCDXG release with no known digest is refused
    Given a WSL machine with an AMD GPU passed through
    When the user previews driver installation for a ROCDXG release with no known digest
    Then the driver plan refuses rather than installing an unverified package
