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
  Scenario: 1 - Entering an engine shell is visibly different from the shell you left
    Given a machine with an installed engine environment
    When the user opens a shell for that engine
    Then the shell is visibly marked as that engine's shell
    And the engine environment's interpreter is the one that runs
    When the user leaves the engine shell
    Then the engine shell exits successfully

  # The engine environment here is planted in the shape the vLLM install actually
  # records — its manifest names the directory holding the engine command, not the
  # environment root (engines/vllm/src/lib.rs derives it as `command.parent()`).
  # That distinction is the whole scenario: the shell composes its activation hint
  # by appending the interpreter directory to whatever the manifest recorded, so a
  # manifest that already points inside it yields a path nobody can source.
  #
  # Phrased as "the file it names is there", not as "the path is not doubled": the
  # first survives whichever way the mismatch is repaired, the second would go
  # stale the moment it is.
  @id:engine-shell-activation-hint-is-usable @requires-os:linux
  Scenario: 2 - The activation hint an engine shell prints refers to a real file
    Given a machine with an engine environment installed the way the engine records it
    When the user opens a shell for that engine
    Then the printed activation hint names a file that exists
    When the user leaves the engine shell
    Then the engine shell exits successfully
