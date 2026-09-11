# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

Feature: ComfyUI runtime selection is actionable

  # `rocm comfyui install` picks the ROCm runtime to install into. When more than
  # one managed runtime is ready and none is activated as the default, the CLI
  # refuses to guess — the same all-or-nothing policy `serve` uses. That refusal
  # is surfaced in `rocm comfyui install`'s command output, where `--runtime-id`
  # and `rocm runtimes activate` apply and the `/runtimes` pointer is for the same
  # text read from a terminal. It is CLI-only today — and not because the TUI has
  # no error path: `/comfyui install` is approval-gated, and a non-zero `rocm` exit
  # is *captured* by `run_rocm_capture_for_paths`, not raised, so the seam yields
  # `RocmToolOutcome::Result` holding an `isError: true` envelope. The `Error` arm
  # that would print a message verbatim is never reached, and `summarize_json_value`
  # collapses the envelope to `content: [1 items]`, so the refusal text does not
  # reach the chat. Pinned by `approved_command_failure_stays_a_collapsed_envelope`
  # in `crates/rocm-dash-tui/src/app/mod.rs`. This scenario asserts the CLI surface
  # only.
  #
  # No GPU is needed: runtime readiness is filesystem + manifest state, so the
  # scenario plants two ready wheel runtimes and asserts the refusal names every
  # remediation and lists both keys. Linux-only because the planted rocm_sdk stub
  # uses `.so` library names; the selection logic it exercises is platform-agnostic.
  @id:comfyui-ambiguous-runtime-actionable @requires-os:linux
  Scenario: comfyui-01 - ComfyUI install refuses ambiguously and names every remediation
    Given two ready ROCm runtimes and no active default
    When the user installs ComfyUI without choosing a runtime
    Then ComfyUI install is refused as ambiguous
    And the refusal offers the /runtimes picker
    And the refusal names the --runtime-id flag
    And the refusal names rocm runtimes activate
    And the refusal lists both runtime keys
