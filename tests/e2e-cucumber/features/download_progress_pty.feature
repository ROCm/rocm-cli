Feature: Download-progress spinner under a real terminal

  # `cli_progress::AnimatedSpinner` only has in-process unit coverage today —
  # it never runs under a spawned subprocess, so a regression that broke its
  # TTY detection, throttling, or line-clearing on `Drop` could ship
  # unnoticed. This proves it end to end: a real `rocm` binary, under a real
  # PTY, downloading from a server paced slowly enough to observe an
  # intermediate progress frame, and confirms the spinner line is gone once
  # the process exits.

  @id:download-progress-pty-01-therock-tarball-spinner-renders @requires-os:linux
  Scenario: download-progress-pty-01 - The tarball download spinner renders progress and clears on completion
    Given a paced canonical release tarball fixture
    When the user installs the tarball SDK for family gfx120X-all under a real terminal
    Then the terminal shows an intermediate download progress frame
    And the tarball install exits cleanly
    And the final terminal screen shows neither spinner line
