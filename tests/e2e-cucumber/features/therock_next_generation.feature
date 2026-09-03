Feature: TheRock ROCm 10 ("next") install layout

  # ROCm 10 ("next") ships from a different, flat pip index (GPU targeting via
  # pip extras) and a tarball index that mixes in non-release `-tests-` sibling
  # files with a later mtime than the real dist archive. Both scenarios point
  # the new ROCM_CLI_THEROCK_NEXT_RELEASE_*_BASE overrides at a loopback fixture
  # server so the dispatch/selection logic is exercised hermetically, with no
  # live network dependency and no change to how older ROCm versions resolve.

  @id:therock-next-wheel-uses-device-extras
  Scenario: Installing the SDK against the ROCm 10 pip layout uses device extras
    Given a ROCm 10 pip index fixture for family gfx1200
    When the user previews a wheel SDK install for family gfx1200
    Then the dry-run output selects the ROCm 10 pip index
    And the dry-run output requests the gfx1200 device extras

  # Linux-only: `--format tarball` is rejected outright on Windows (native
  # tarball installs aren't supported there), so this scenario's premise
  # doesn't hold on that platform.
  @id:therock-next-tarball-skips-tests-artifact @requires-os:linux
  Scenario: Installing the SDK against the ROCm 10 tarball layout skips the tests sibling
    Given a ROCm 10 tarball index fixture for family gfx1200 with a tests sibling
    When the user previews a tarball SDK install for family gfx1200
    Then the dry-run output selects the real tarball artifact
    And the dry-run output does not select the tests artifact
