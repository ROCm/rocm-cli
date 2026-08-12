Feature: Engine shell and engine installation

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

  # Expected to FAIL on a GPU host. Installing an engine is a plain command a
  # user runs from wherever they happen to be — a container, a bare `ssh`
  # session, a service unit — and none of those necessarily give a process its
  # own per-session scratch directory. The CLI already copes with that for the
  # directories it owns; the engine it launches on the user's behalf must not be
  # left to fail for the same reason.
  #
  # Only the missing-scratch-directory failure is pinned. Anything else the
  # install runs into on a given host (no supported GPU backend, a download
  # problem) is that host's business and does not make this scenario fail, so it
  # goes green the day the CLI stops leaving the engine without one.
  #
  # @requires-gpu because the engine's own launcher needs the AMD userspace
  # libraries to get far enough to look for a scratch directory at all: on a
  # plain container it dies earlier, for an unrelated reason, and would prove
  # nothing.
  @id:engines-install-without-a-session-runtime-dir @requires-gpu @requires-os:linux
  Scenario: 2 - Installing an engine works from a session with no scratch directory
    Given a session that provides no scratch directory of its own
    When the user installs the Lemonade engine
    Then the install does not fail for want of a scratch directory
