# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

Feature: ComfyUI runtime selection is actionable

  # `rocm comfyui install` picks the ROCm runtime to install into. When more than
  # one managed runtime is ready and none is activated as the default, the CLI
  # refuses to guess — the same all-or-nothing policy `serve` uses. That refusal
  # has to be actionable on both surfaces it reaches: the CLI, where
  # `--runtime-id` and `rocm runtimes activate` apply, and the TUI, where
  # `/comfyui install` renders this text into the chat and the fix is the
  # `/runtimes` picker.
  #
  # No GPU is needed: runtime readiness is filesystem + manifest state, so the
  # scenario plants two ready wheel runtimes and asserts the refusal names every
  # remediation and lists both keys. Linux-only because the planted rocm_sdk stub
  # uses `.so` library names; the selection logic it exercises is platform-agnostic.
  @id:comfyui-ambiguous-runtime-actionable @requires-os:linux
  Scenario: ComfyUI install refuses ambiguously and names every remediation
    Given two ready ROCm runtimes and no active default
    When the user installs ComfyUI without choosing a runtime
    Then ComfyUI install is refused as ambiguous
    And the refusal offers the /runtimes picker
    And the refusal names the --runtime-id flag
    And the refusal names rocm runtimes activate
    And the refusal lists both runtime keys
