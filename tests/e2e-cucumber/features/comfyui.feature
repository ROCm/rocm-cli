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
    Given a ready ROCm runtime with a ComfyUI checkout pending dependencies
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
    Given a ready ROCm runtime with a ComfyUI checkout pending dependencies
    And the ComfyUI dependency install with uv prints progress and succeeds
    When the user installs ComfyUI
    Then the CLI succeeds and shows the install progress

  # `rocm comfyui install` picks the ROCm runtime to install into. When more than
  # one managed runtime is ready and none is activated as the default, the CLI
  # refuses to guess — the same all-or-nothing policy `serve` uses. That refusal
  # is surfaced in `rocm comfyui install`'s command output, where `--runtime-id`
  # and `rocm runtimes activate` apply and the `/runtimes` pointer is for the same
  # text read from a terminal. This scenario asserts that CLI surface only: the
  # refusal does not reach the TUI chat. Both halves of that are machine-checked
  # — that a non-zero `rocm` exit arrives as a captured `isError: true` envelope
  # rather than as an error, by
  # `seam_execute_approved_captures_a_failing_command_as_a_result`
  # (`apps/rocm/src/dash_seam.rs`, which replays a real refusing subprocess), and
  # that such an envelope is then collapsed out of the chat, by
  # `approved_command_failure_stays_a_collapsed_envelope`
  # (`crates/rocm-dash-tui/src/app/mod.rs`). The reasoning is written out at the
  # seam that decides it (`execute_approved` in `apps/rocm/src/dash_seam.rs`).
  #
  # No GPU is needed: runtime readiness is filesystem + manifest state, so the
  # scenario plants two ready wheel runtimes and asserts the refusal names every
  # remediation and lists both keys. Linux-only because the planted rocm_sdk stub
  # uses `.so` library names; the selection logic it exercises is platform-agnostic.
  @id:comfyui-ambiguous-runtime-actionable @requires-os:linux
  Scenario: comfyui-03 - ComfyUI install refuses an ambiguous runtime choice and names every remediation
    Given two ready ROCm runtimes and no active default
    When the user installs ComfyUI without choosing a runtime
    Then ComfyUI install is refused as ambiguous
    And the refusal offers the /runtimes picker
    And the refusal names the --runtime-id flag
    And the refusal names rocm runtimes activate
    And the refusal lists both runtime keys

  # `download_and_extract_source` reports its download the same way TheRock's
  # tarball install does (`cli_progress::AnimatedSpinner`), but its extraction
  # is an in-process `GzDecoder`/`tar` unpack with no separate progress phase —
  # unlike TheRock's subprocess `tar -xf`, it never renders its own frame. This
  # proves the download half end to end: a real `rocm` binary, under a real
  # PTY, fetching from a server paced slowly enough to observe an intermediate
  # progress frame, and confirms the spinner line is gone once the process
  # exits. See `download_progress_pty.feature` for the TheRock counterpart.
  # @serial: same reasoning as `download_progress_pty.feature`'s
  # `download-progress-01` — this scenario's intermediate progress frame
  # depends on real wall-clock pacing that CPU contention from up to 63
  # concurrently-running scenarios can starve away entirely.
  # Note: "Fetching ComfyUI source archive…" is short enough to never
  # truncate at 80 columns, so this scenario does not exercise the
  # label-truncation fix in `cli_progress::assemble_status_line` — the
  # tarball scenario in `download_progress_pty.feature` is the regression
  # test for that.
  @id:comfyui-source-download-shows-live-progress @requires-os:linux @serial
  Scenario: comfyui-04 - The source-archive download spinner renders progress and clears on completion
    Given a paced ComfyUI source archive fixture
    When the user installs ComfyUI under a real terminal
    Then the terminal shows an intermediate ComfyUI download progress frame
    And the ComfyUI install exits cleanly
    And the final terminal screen shows no ComfyUI download spinner line

  # `--reinstall` used to delete ComfyUI's whole `source/` folder before it
  # downloaded anything. That folder is where ComfyUI keeps the user's models
  # (the `models path:` the CLI prints), saved workflows (`user/`), generated
  # images (`output/`), uploads (`input/`), installed custom nodes and an
  # `extra_model_paths.yaml`, because `rocm comfyui start` runs ComfyUI from it
  # without redirecting any of them. A reinstall now replaces only the code
  # around those. These scenarios plant a used install, then reinstall from a
  # loopback archive server: the success path, the path where the download
  # fails, the dry run, and a reinstall attempted while ComfyUI runs. Each
  # asserts what the CLI prints together with what is actually on disk
  # afterwards. Linux-only because the planted runtime uses `.so` names and a
  # POSIX-shell Python stand-in.
  @id:comfyui-reinstall-keeps-user-data @requires-os:linux
  Scenario: comfyui-05 - Reinstalling ComfyUI replaces its code and keeps the user's own files
    Given a ComfyUI install holding the user's models, workflows, images and custom nodes
    And a newer ComfyUI release is available to download
    When the user reinstalls ComfyUI
    Then the reinstall reports the user's folders as kept and they still hold the user's files
    And ComfyUI's code is the newer release

  @id:comfyui-reinstall-failed-download-changes-nothing @requires-os:linux
  Scenario: comfyui-06 - A ComfyUI reinstall whose download fails leaves the existing install untouched
    Given a ComfyUI install holding the user's models, workflows, images and custom nodes
    And the ComfyUI release download fails
    When the user reinstalls ComfyUI
    Then the reinstall fails
    And the existing ComfyUI code and the user's files are untouched

  @id:comfyui-reinstall-dry-run-names-kept-folders @requires-os:linux
  Scenario: comfyui-07 - A ComfyUI reinstall dry run says what it replaces and keeps
    Given a ComfyUI install holding the user's models, workflows, images and custom nodes
    When the user previews reinstalling ComfyUI
    Then the preview says the ComfyUI code is replaced and names the kept folders
    And the preview's install command includes --reinstall
    And the existing ComfyUI code and the user's files are untouched

  # The running ComfyUI keeps the old code loaded and keeps writing into the
  # folder a reinstall changes, so the reinstall is refused until it is
  # stopped. The refusal names `rocm comfyui stop`; the scenario runs exactly
  # that and proves the same reinstall then goes through.
  @id:comfyui-reinstall-refused-while-running @requires-os:linux
  Scenario: comfyui-08 - A ComfyUI reinstall waits until the running ComfyUI is stopped
    Given a ComfyUI install holding the user's models, workflows, images and custom nodes
    And a newer ComfyUI release is available to download
    And the ComfyUI that rocm-cli started is running from that install
    When the user reinstalls ComfyUI
    Then the reinstall is refused and names rocm comfyui stop
    And the existing ComfyUI code and the user's files are untouched
    When the user stops ComfyUI
    And the user reinstalls ComfyUI
    Then the reinstall reports the user's folders as kept and they still hold the user's files
    And ComfyUI's code is the newer release

  # A reinstall killed while it replaces the code leaves a half-replaced
  # folder. `start` must not launch it; it names the install command that
  # finishes the reinstall, and the scenario runs that exact command (read from
  # the CLI's own message) and proves the condition is cleared.
  @id:comfyui-interrupted-reinstall-blocks-start @requires-os:linux
  Scenario: comfyui-09 - ComfyUI will not start a half-replaced install until the reinstall is finished
    Given a ComfyUI install holding the user's models, workflows, images and custom nodes
    And a newer ComfyUI release is available to download
    When the user reinstalls ComfyUI
    Then the reinstall reports the user's folders as kept and they still hold the user's files
    Given the reinstall was cut short while replacing ComfyUI's code
    When the user starts ComfyUI
    Then start refuses and names the command that finishes the reinstall
    And ComfyUI status reports the interrupted reinstall
    When the user runs the command start named
    Then ComfyUI status no longer reports an interrupted reinstall
    And ComfyUI's code is the newer release
    And the user's own files are untouched
