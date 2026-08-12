Feature: System dependency installs preserve ROCm

  # vLLM needs the OpenMPI runtime, which the CLI installs through the host
  # package manager. `apt-get install -y` assumes yes for *removals* as well as
  # installs, so when apt chose to satisfy OpenMPI by evicting the ROCm stack it
  # did so unattended — reported in the field as `rocm install sdk` removing
  # rocm, rocm-hip, rocm-hip-runtime-dev, mivisionx-dev and rpp-dev while pulling
  # an older toolchain, breaking a working ROCm install.
  #
  # The package manager is planted rather than real, so the scenario owns the
  # dependency solution apt reports and needs no GPU, no engine install and no
  # network. Linux-only: the dependency-setup path does not run on Windows.
  #
  # It also assumes an apt host, which every Linux lane is — the guard is
  # apt-specific by design, because only `apt-get install -y` assumes yes for
  # removals. On a Linux host whose /etc/os-release selects dnf/zypper/pacman
  # the CLI plans a different, additive command and there is no refusal to
  # assert; move this behind a package-manager capability gate if such a lane
  # is ever added.
  @id:deps-guard-refuses-rocm-removal @requires-os:linux
  Scenario: 1 - Installing a dependency never silently removes ROCm
    Given a machine with a registered ROCm runtime
    And installing OpenMPI would remove the ROCm packages
    When the user installs the vLLM engine and approves system changes
    Then the CLI refuses instead of removing them
    And it does so before changing anything on the system
    And it lists every ROCm package that would have been removed
