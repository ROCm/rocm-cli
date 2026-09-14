Feature: ComfyUI dependency install failures name the log to read

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
