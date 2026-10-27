Feature: Download-progress spinner under a real terminal

  # `cli_progress::AnimatedSpinner` only has in-process unit coverage today —
  # it never runs under a spawned subprocess, so a regression that broke its
  # TTY detection, throttling, or line-clearing on `Drop` could ship
  # unnoticed. This proves it end to end: a real `rocm` binary, under a real
  # PTY, downloading from a server paced slowly enough to observe an
  # intermediate progress frame, and confirms the spinner line is gone once
  # the process exits.

  # @serial: this scenario's progress frames depend on real wall-clock pacing
  # between paced HTTP chunks and the PTY's polling cadence. Running alongside
  # up to 63 other scenarios (the mock lane's default concurrency) starves it
  # of CPU at unpredictable moments, letting the whole paced transfer (or the
  # `tar` extraction) complete between polls with no intermediate frame ever
  # observed — reproduced locally by running the full suite, never by running
  # this scenario alone. Serial execution removes that contention.
  @id:download-progress-linux-tarball-install-shows-live-progress @requires-os:linux @serial
  Scenario: download-progress-01 - The tarball download spinner renders progress and clears on completion
    Given a paced canonical release tarball fixture
    When the user installs the tarball SDK for family gfx120X-all under a real terminal
    Then the terminal shows an intermediate download progress frame
    And the terminal shows the archive being extracted
    And the tarball install exits cleanly
    And the final terminal screen shows neither spinner line
