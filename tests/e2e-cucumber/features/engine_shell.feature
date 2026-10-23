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

  # The env_path is planted bin-shaped, the way the vLLM install RESPONSE reports it
  # (`command.parent()` in engines/vllm/src/install.rs; users reach it through the
  # auto-install branch of resolve_engine_env). No install writes such a manifest
  # to disk — planting one is only the cheapest way to hand the shell that value
  # without a multi-GiB install. The shell composes its activation hint by
  # appending the interpreter directory to whatever env_path it is given, so a
  # value that already points inside it yields a path nobody can source.
  #
  # Consequence for whoever fixes this: a fix in engines/vllm (report the env root)
  # closes EAI-8049 but leaves this fixture unchanged, so the scenario must then be
  # rewritten or deleted along with its expectation row. Only a fix to the hint
  # composition itself makes this row go stale.
  #
  # Phrased as "the file it names is there", not as "the path is not doubled": the
  # first survives whichever way the mismatch is repaired, the second would go
  # stale the moment it is.
  @id:engine-shell-activation-hint-is-usable @requires-os:linux
  Scenario: engine-shell-02 - The activation hint an engine shell prints refers to a real file
    Given a machine with an engine environment installed the way the engine records it
    When the user opens a shell for that engine
    Then the printed activation hint names a file that exists
    When the user leaves the engine shell
    Then the engine shell exits successfully
