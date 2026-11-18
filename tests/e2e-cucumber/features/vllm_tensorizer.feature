Feature: vLLM's tensorizer dependency must import before an install is reported usable

  # vLLM's wheel pins tensorizer 2.10.1, which raises a KeyError at import under
  # Python 3.14 (ROCm 10.1.0's vLLM is published for that Python only). Before the
  # install checked for it, `rocm engines install vllm` reported its dependencies
  # satisfied and the first `rocm serve` died in the engine core.
  #
  # The runtime's Python and `uv` are planted shell stand-ins: the Python cannot
  # import tensorizer until the stand-in `uv` has installed the fixed release, and
  # the stand-in `uv` answers `pip check` the way the real one does once a release
  # other than vLLM's exact pin is present. That makes the installed tensorizer a
  # file the scenario can read, so each message is asserted together with the
  # state it describes. No GPU, network or real vLLM is needed. Linux only: the
  # stand-ins are shell scripts and the install path is not reached on Windows.
  @id:vllm-tensorizer-upgraded-when-pin-cannot-import @requires-os:linux
  Scenario: vllm-tensorizer-01 - An unimportable tensorizer pin is upgraded and reported as a deliberate divergence
    Given a managed vLLM runtime whose pinned tensorizer cannot be imported
    When the user installs the vLLM engine from that runtime
    Then the install succeeds
    And the user is told tensorizer was upgraded to 2.12.1
    And the dependency check reports the tensorizer pin as an expected divergence
    And tensorizer 2.12.1 is what the runtime now holds

  @id:vllm-tensorizer-second-install-leaves-upgrade-alone @requires-os:linux
  Scenario: vllm-tensorizer-02 - Installing again keeps the upgraded tensorizer and does not reinstall vLLM
    Given a managed vLLM runtime whose pinned tensorizer cannot be imported
    And the user has installed the vLLM engine from that runtime
    When the user installs the vLLM engine from that runtime
    Then the install succeeds
    And the dependency check reports the tensorizer pin as an expected divergence
    And tensorizer was upgraded only once and vLLM was not reinstalled

  # An upgrade can fail to help. On a runtime this run did not install vLLM into
  # that must not turn a previously successful install into a failure, but the
  # warning says what will happen at serve time.
  @id:vllm-tensorizer-warns-when-upgrade-does-not-help @requires-os:linux
  Scenario: vllm-tensorizer-03 - An upgrade that does not help only warns on a runtime this run did not install into
    Given a managed vLLM runtime whose tensorizer cannot be imported even after an upgrade
    When the user installs the vLLM engine from that runtime
    Then the install succeeds
    And the install warns that tensorizer still cannot be imported and shows the KeyError

  # Python 3.13+ writes a coloured traceback under FORCE_COLOR even when piped,
  # with escapes around the quoted path. The probe still has to see tensorizer's
  # frame, or a broken tensorizer reads as fine and `rocm serve` crashes.
  @id:vllm-tensorizer-coloured-traceback-still-upgraded @requires-os:linux
  Scenario: vllm-tensorizer-04 - A coloured traceback does not hide a broken tensorizer
    Given a managed vLLM runtime whose Python writes its tracebacks in colour
    When the user installs the vLLM engine from that runtime
    Then the install succeeds
    And the user is told tensorizer was upgraded to 2.12.1
    And tensorizer 2.12.1 is what the runtime now holds

  # Offline, the upgrade cannot be fetched. On a runtime this run did not install
  # vLLM into that must not turn a previously successful install into a failure,
  # but the warning says what will happen at serve time and nothing was changed.
  @id:vllm-tensorizer-failed-upgrade-warns-on-existing-runtime @requires-os:linux
  Scenario: vllm-tensorizer-05 - A tensorizer that cannot be upgraded only warns on an existing runtime
    Given a managed vLLM runtime whose pinned tensorizer cannot be imported and cannot be upgraded
    When the user installs the vLLM engine from that runtime
    Then the install succeeds
    And the install warns that tensorizer could not be upgraded

  # The check fails hard whenever the run installed vLLM (a first install,
  # `--reinstall` or a repair), because the runtime would be left unable to serve.
  # `--reinstall` on an existing runtime is the form this scenario runs.
  @id:vllm-tensorizer-failed-upgrade-refuses-reinstall @requires-os:linux
  Scenario: vllm-tensorizer-06 - A tensorizer that cannot be upgraded fails an install that reinstalls vLLM
    Given a managed vLLM runtime whose pinned tensorizer cannot be imported and cannot be upgraded
    When the user reinstalls the vLLM engine from that runtime
    Then the install is refused because tensorizer could not be upgraded

  # A ROCm 10.x discover runtime launches the server with a PYTHONPATH, and
  # `python -I` would drop it: the probe would then stop on a missing module
  # before tensorizer and call a broken tensorizer fine.
  @id:vllm-tensorizer-probe-keeps-the-server-pythonpath @requires-os:linux
  Scenario: vllm-tensorizer-07 - The probe runs with the PYTHONPATH the server launches with
    Given a managed vLLM runtime that only imports with the server's PYTHONPATH
    When the user installs the vLLM engine from that runtime
    Then the install succeeds
    And the user is told tensorizer was upgraded to 2.12.1
    And tensorizer 2.12.1 is what the runtime now holds

  # An import that fails in torch, below a tensorizer frame, on a missing library
  # is for the device check to report. Upgrading tensorizer would not help and
  # refusing the install over it would be wrong.
  @id:vllm-tensorizer-library-failure-is-left-alone @requires-os:linux
  Scenario: vllm-tensorizer-08 - A missing library below tensorizer does not trigger an upgrade
    Given a managed vLLM runtime whose import fails on a missing library below tensorizer
    When the user installs the vLLM engine from that runtime
    Then the install succeeds without changing tensorizer

  @id:vllm-tensorizer-library-error-in-tensorizer-is-left-alone @requires-os:linux
  Scenario: vllm-tensorizer-09 - A missing-library error raised by tensorizer's own code does not trigger an upgrade
    Given a managed vLLM runtime whose tensorizer code raises a missing-library error
    When the user installs the vLLM engine from that runtime
    Then the install succeeds without changing tensorizer

  @id:vllm-tensorizer-non-library-error-below-tensorizer-is-left-alone @requires-os:linux
  Scenario: vllm-tensorizer-10 - A non-library exception raised below tensorizer does not trigger an upgrade
    Given a managed vLLM runtime whose import raises a non-library error in torch below tensorizer
    When the user installs the vLLM engine from that runtime
    Then the install succeeds without changing tensorizer

  @id:vllm-tensorizer-still-broken-after-reinstall-is-refused @requires-os:linux
  Scenario: vllm-tensorizer-11 - A reinstall whose tensorizer still cannot be imported says vLLM was installed
    Given a managed vLLM runtime whose tensorizer cannot be imported even after an upgrade
    When the user reinstalls the vLLM engine from that runtime
    Then the install is refused
    And the refusal says the install ran and tensorizer still cannot be imported after the upgrade

  # The remedy the CLI names beside a tensorizer divergence is a vLLM reinstall.
  # A reinstall puts vLLM's own tensorizer pin back, so it only clears the
  # condition if the install then upgrades tensorizer again. This starts from a
  # genuine violation beside the already upgraded tensorizer and checks the final
  # state: the violation gone, and tensorizer upgraded again after vLLM.
  @id:vllm-tensorizer-repair-clears-violation-and-keeps-upgrade @requires-os:linux
  Scenario: vllm-tensorizer-12 - Repairing a genuine violation beside the upgraded tensorizer keeps the upgrade
    Given a managed vLLM runtime with a genuine violation beside the upgraded tensorizer
    When the user installs the vLLM engine from that runtime
    Then the violation is cleared and tensorizer is upgraded again after vLLM was reinstalled

  # The remedy the CLI names beside a tensorizer divergence is a vLLM reinstall,
  # and the command run here is the one it printed. The reinstall puts vLLM's own
  # tensorizer pin back, so it only clears the condition if the install then
  # upgrades tensorizer again: the final state is the violation gone and
  # tensorizer upgraded after vLLM was reinstalled.
  @id:vllm-tensorizer-advised-reinstall-clears-violation @requires-os:linux
  Scenario: vllm-tensorizer-13 - The advised reinstall clears a violation beside the upgraded tensorizer and keeps the upgrade
    Given a managed vLLM runtime whose violation beside the upgraded tensorizer only a reinstall clears
    When the user installs the vLLM engine from that runtime
    Then the install reports the violation and advises reinstalling vLLM
    When the user runs the advised command
    Then the advised reinstall cleared the violation and tensorizer was upgraded after it

  # The first install: vLLM is not in the runtime yet, so this run installs it,
  # and a tensorizer that cannot be upgraded fails the install, saying so.
  @id:vllm-tensorizer-failed-upgrade-refuses-first-install @requires-os:linux
  Scenario: vllm-tensorizer-14 - A tensorizer that cannot be upgraded fails the first install of vLLM
    Given a managed runtime without vLLM whose pinned tensorizer cannot be imported and cannot be upgraded
    When the user installs the vLLM engine from that runtime
    Then the install is refused because tensorizer could not be upgraded
