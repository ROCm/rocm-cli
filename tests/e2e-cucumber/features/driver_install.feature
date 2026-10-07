Feature: Driver installation planning

  # These use --dry-run and an unapproved review, so they exercise the released
  # CLI without privileged commands or host mutation.
  #
  # `@requires-wsl` rather than `@requires-os:linux`: the plan branch under test
  # is chosen by `is_wsl_host()`, which trusts only `/dev/dxg` and the kernel
  # build string in `/proc/version`. `$WSL_DISTRO_NAME` is deliberately not
  # trusted, so there is no way to stand a WSL host up on a native Linux runner
  # — the scenario has no premise there and must skip rather than assert against
  # a bare-metal plan.
  @id:driver-install-wsl-dry-run-plan @requires-wsl
  Scenario: driver-install-01 - Previewing the WSL driver install produces an actionable packaged plan
    When the user previews driver installation on this WSL host
    Then the driver plan is supported and mutating
    And the dry-run driver plan requires no approval and previews no execution
    And the driver plan verifies the download before installing it
    And the driver plan does not direct the user to the removed WSL setup script

  @id:driver-install-wsl-review-requires-approval @requires-wsl
  Scenario: driver-install-02 - Reviewing the WSL driver install requires approval before execution
    When the user reviews driver installation on this WSL host without approval
    Then the unapproved WSL driver plan is actionable but not executed
    And the driver plan does not direct the user to the removed WSL setup script

  # The package is installed as root, so an unrecognised release must stop
  # rather than fall back to installing something nothing has authenticated.
  @id:driver-install-wsl-unverified-release-refused @requires-wsl
  Scenario: driver-install-03 - Installing a ROCDXG release with no known digest is refused
    When the user previews driver installation for a ROCDXG release with no known digest
    Then the driver plan refuses rather than installing an unverified package
