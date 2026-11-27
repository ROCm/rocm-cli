Feature: vLLM companion recovery preserves the selected wheel through installation

  # Shell stand-ins model an index with both plain and cross-line local builds.
  # The real install writes its selected versions to files: checking those files
  # together with the recovery warning detects weakening === back to ==.
  #
  # The two SDK lines are spelled out as separate scenarios rather than as one
  # outline over an Examples table: the harness keys its report grid and its
  # lane expectations on `@id`, and every row an outline expands to would carry
  # the same one.
  @id:vllm-companion-strict-recovery-rocm-10-1 @requires-os:linux
  Scenario: vllm-companion-01 - Plain companions stay selected and the SDK 10.1 torch stack is retained
    Given a ROCm 10.1.0 runtime whose companion index also serves plain wheels
    When the user reinstalls vLLM against the mixed companion index
    Then the plain companion wheels are installed and the recovery is reported
    And the SDK torch stack is realigned from its recorded index

  @id:vllm-companion-strict-recovery-rocm-10-2 @requires-os:linux
  Scenario: vllm-companion-02 - Plain companions stay selected and the SDK 10.2 torch stack is retained
    Given a ROCm 10.2.0 runtime whose companion index also serves plain wheels
    When the user reinstalls vLLM against the mixed companion index
    Then the plain companion wheels are installed and the recovery is reported
    And the SDK torch stack is realigned from its recorded index

  @id:vllm-companion-missing-plain-refused @requires-os:linux
  Scenario: vllm-companion-03 - A cross-line companion without a plain wheel is refused before installation
    Given a ROCm 10.2.0 runtime whose companion index has no plain wheels
    When the user reinstalls vLLM against the mixed companion index
    Then the cross-line companion is refused and no packages are installed
