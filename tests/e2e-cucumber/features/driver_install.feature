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

  # Every bare-metal Linux lane in this repo runs Ubuntu, but not always the
  # same release -- the self-hosted GPU lane's real /etc/os-release reports
  # 26.04 ("resolute"), which `build_driver_install_plan` routes to the
  # `ubuntu_native_archive` arm instead of the apt one (no repo.radeon.com/
  # amdgpu suite published for "resolute" yet; see the `("ubuntu", "26.04")`
  # arm in `driver_install.rs`). `--dry-run` previews the plan without writing
  # anything, so this is safe on a shared CI host either way: on an apt-arm
  # release it asserts the real `apt_driver_plan` command text this binary
  # would use to write /etc/apt/sources.list.d/amdgpu.list and import AMD's
  # signing key, the same commands the unit fixtures in `driver_install.rs`
  # exercise with a hand-written os-release; on 26.04 it asserts the
  # native-archive refusal instead, so the scenario proves the CLI's actual
  # output on whichever release the lane happens to run rather than
  # hard-coding an apt-arm premise a lane can silently outgrow.
  @id:driver-install-apt-repo-file-and-key-content @requires-os:linux @requires-bare-metal
  Scenario: driver-install-04 - Previewing the driver install on a real bare-metal host plans release-appropriate driver setup
    When the user previews driver installation on this host
    Then the driver plan matches this host's actual Ubuntu release
