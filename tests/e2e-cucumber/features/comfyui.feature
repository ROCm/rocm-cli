# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

Feature: ComfyUI install reports progress and makes failures actionable

  # `rocm comfyui install` shells out to `uv` to resolve ComfyUI's Python
  # dependencies. A failed resolve (a version conflict, a yanked release, a
  # flaky index) is the most likely real-world break, and it is deterministic
  # without a GPU or a terminal -- so it is the scenario to cover instead of the
  # unit test that used to be the only coverage for this failure message: it
  # proves the CLI's stderr names the exact install log a user needs to read
  # next, not just that the string got built correctly in isolation.
  #
  # The runtime, its rocm_sdk probe, the Python interpreter and `uv` itself are
  # all planted so the scenario is hermetic: no GPU, no network, no real
  # package resolution. Linux-only because the planted shims are POSIX shell
  # scripts.
  @id:comfyui-uv-install-failure-names-log @requires-os:linux
  Scenario: comfyui-01 - A failed dependency install names the log it wrote
    Given a ready ROCm install with a ComfyUI checkout pending dependencies
    And the ComfyUI dependency install with uv fails
    When the user installs ComfyUI
    Then the CLI fails and names the install log it wrote

  # The install progress spinner is TTY-gated and a no-op off a terminal, so a
  # piped/CI install relies on `uv`'s own output being streamed through
  # instead -- without it, a multi-minute dependency resolve would print
  # nothing at all, indistinguishable from a hang. This scenario's harness
  # runs non-interactively by construction, so it can assert that streaming
  # directly, on the success path (the failure path is `comfyui-01` above).
  @id:comfyui-uv-install-progress-streamed @requires-os:linux
  Scenario: comfyui-02 - A dependency install streams progress output
    Given a ready ROCm install with a ComfyUI checkout pending dependencies
    And the ComfyUI dependency install with uv prints progress and succeeds
    When the user installs ComfyUI
    Then the CLI succeeds and shows the install progress

  # `rocm comfyui install` picks the ROCm runtime to install into. When more than
  # one managed runtime is ready and none is activated as the default, the CLI
  # refuses to guess — the same all-or-nothing policy `serve` uses. That refusal
  # is surfaced in `rocm comfyui install`'s command output, where `--runtime-id`
  # and `rocm runtimes activate` apply and the `/runtimes` pointer is for the same
  # text read from a terminal. This scenario asserts that CLI surface only: the
  # refusal does not reach the TUI chat, which is machine-checked by
  # `approved_command_failure_stays_a_collapsed_envelope`
  # (`crates/rocm-dash-tui/src/app/mod.rs`) and explained at the seam that decides
  # it (`execute_approved` in `apps/rocm/src/dash_seam.rs`).
  #
  # No GPU is needed: runtime readiness is filesystem + manifest state, so the
  # scenario plants two ready wheel runtimes and asserts the refusal names every
  # remediation and lists both keys. Linux-only because the planted rocm_sdk stub
  # uses `.so` library names; the selection logic it exercises is platform-agnostic.
  @id:comfyui-ambiguous-runtime-actionable @requires-os:linux
  Scenario: comfyui-03 - ComfyUI install refuses ambiguously and names every remediation
    Given two ready ROCm runtimes and no active default
    When the user installs ComfyUI without choosing a runtime
    Then ComfyUI install is refused as ambiguous
    And the refusal offers the /runtimes picker
    And the refusal names the --runtime-id flag
    And the refusal names rocm runtimes activate
    And the refusal lists both runtime keys
