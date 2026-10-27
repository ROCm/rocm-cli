Feature: Engine shell

  # `rocm engines shell` always did activate the engine environment, but nothing
  # said so: the prompt marker was passed through the PS1 *environment variable*,
  # which bash reassigns from its startup files on every interactive shell. Users
  # landed in a correctly activated shell that looked exactly like the one they
  # left and reported the command as doing nothing.
  #
  # Driven through a real pseudo-terminal, because the defect is only visible in
  # what the terminal renders — a piped run would have passed throughout. The
  # engine environment is planted rather than installed, so this needs no GPU and
  # no engine install and runs on every lane.
  #
  # Linux-only and pinned to bash: the prompt marker is not implemented on
  # Windows, and the runner's own $SHELL varies, which would otherwise decide
  # whether a marker appears at all.
  @id:engine-shell-marks-the-prompt @requires-os:linux
  Scenario: engine-shell-01 - Entering an engine shell is visibly different from the shell you left
    Given a machine with an installed engine environment
    When the user opens a shell for that engine
    Then the shell is visibly marked as that engine's shell
    And the engine environment's interpreter is the one that runs
    When the user leaves the engine shell
    Then the engine shell exits successfully
