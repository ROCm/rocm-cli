Feature: Native driver installation

  # EAI-8053: `rocm install driver --dkms` builds its command plan with a literal
  # `sudo ` prefix on every step, unconditionally. When the CLI already runs as
  # root the prefix is unnecessary, and on a root host that has no `sudo` binary it
  # is actively harmful: the very first command dies with `sudo: not found` before
  # any driver work happens. The contract is that being root — the state where the
  # commands could otherwise succeed — must not be the thing that breaks the run.
  #
  # Root-gated: the fix is uid-aware (prepend `sudo` only when NOT root), so the
  # "no sudo prefix" contract only has a premise where the runner is actually root.
  # Off root the sudo prefix is correct, so @requires-root skips there rather than
  # letting the row falsely pass. Linux-only: the plan and its `sh -c` execution
  # are the Linux DKMS path.
  @id:driver-install-as-root-does-not-require-sudo @requires-os:linux @requires-root
  Scenario: 1 - Installing the driver as root does not depend on sudo being present
    Given a root machine with no sudo command available
    When the user installs the native driver with dkms
    Then the install does not fail merely because sudo is missing
