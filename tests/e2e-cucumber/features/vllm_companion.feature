Feature: vLLM companion recovery preserves the selected wheel through installation

  # Shell stand-ins model an index with both plain and cross-line local builds.
  # The real install writes its selected versions to files: checking those files
  # together with the recovery warning detects weakening === back to ==.
  @id:vllm-companion-strict-recovery @requires-os:linux
  Scenario Outline: vllm-companion-01 - Plain companions stay selected and the SDK torch stack is retained
    Given a ROCm <sdk> runtime whose companion index also serves plain wheels
    When the user reinstalls vLLM against the mixed companion index
    Then the plain companion wheels are installed and the recovery is reported
    And the SDK torch stack is realigned from its recorded index

    Examples:
      | sdk    |
      | 10.1.0 |
      | 10.2.0 |

  @id:vllm-companion-missing-plain-refused @requires-os:linux
  Scenario: vllm-companion-02 - A cross-line companion without a plain wheel is refused before installation
    Given a ROCm 10.2.0 runtime whose companion index has no plain wheels
    When the user reinstalls vLLM against the mixed companion index
    Then the cross-line companion is refused and no packages are installed
