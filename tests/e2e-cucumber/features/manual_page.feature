Feature: The rocm manual page

  # `man rocm` is offline help, so it must describe the binary it came with:
  # the commands `rocm --help` lists and examples that actually run. These
  # scenarios read the page the binary under test produces for the Linux
  # installer, from its roff source, so they need nothing on the runner but
  # the binary. Where the installer puts it, and that man finds and shows it
  # there, is lifecycle-36 in install_lifecycle.feature. Linux only, the one
  # platform the page is installed on.

  @id:manual-page-lists-the-commands @requires-os:linux
  Scenario: manual-page-01 - The manual lists the same commands as the help
    When the user reads the rocm manual page
    Then the manual describes every command the help lists, and none it hides

  @id:manual-page-examples-are-accepted @requires-os:linux
  Scenario: manual-page-02 - Every example in the manual is a command rocm accepts
    When the user reads the rocm manual page
    Then rocm accepts every example the manual gives
