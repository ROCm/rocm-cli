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
  # package resolution.
  @id:comfyui-uv-install-failure-names-log
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
  #
  # This scenario also runs the install through to a saved manifest (unlike
  # comfyui-01, which fails before one is written), so it doubles as the e2e
  # coverage for `folder:` naming the ComfyUI source checkout consistently in
  # both `install` and `status` output -- a user-observable line that AGENTS.md
  # requires a Gherkin scenario for, not just the unit tests that assert each
  # command's `folder:` line in isolation.
  @id:comfyui-uv-install-progress-streamed
  Scenario: comfyui-02 - A dependency install streams progress output
    Given a ready ROCm runtime with a ComfyUI checkout pending dependencies
    And the ComfyUI dependency install with uv prints progress and succeeds
    When the user installs ComfyUI
    Then the CLI succeeds and shows the install progress
    And the install output's folder: line names the ComfyUI source checkout
    When the user checks ComfyUI status
    Then status names the same ComfyUI source folder as the install

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
  # remediation and lists both keys.
  @id:comfyui-ambiguous-runtime-actionable
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
  # Linux-only: this scenario drives `rocm` under a real PTY, the same
  # suite-wide limitation `dash.feature`'s header comment documents
  # (portable-pty compiles on Windows via ConPTY, but that path isn't yet
  # promoted to a blocking contract) — unrelated to this file's other
  # scenarios, whose own Linux gate is explained below.
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
  # fails, the dry run, a reinstall attempted while ComfyUI runs (also one
  # started during the download), an interrupted reinstall, a user's entry set
  # aside, a failure after the new code is in place, and a download that is
  # not a ComfyUI release. Each asserts what the CLI prints together with what
  # is actually on disk afterwards. Only three of these still need the POSIX
  # shell: comfyui-11's GPU-check failing transition plants a `#!/bin/sh`
  # script (its passing-again half uses the Windows-portable `fake-python`
  # stand-in, like every other scenario here), and comfyui-08/comfyui-12 each
  # start their stand-in running ComfyUI process through `sh -c` (see
  # `spawn_stand_in`). comfyui-05/-06/-07/-09/-10/-13 plant the same
  # Windows-portable `fake-python` stand-in `comfyui-01`/`-02`/`-03` use and
  # have no shell dependency of their own, but stay `@requires-os:linux` too:
  # their reinstall behavior has not been run on Windows yet. Ungating them
  # is tracked separately (#697), to keep this fix scoped to #629's
  # comfyui-01/02/03 ask.
  #
  # rocm-cli's saved record of a ComfyUI it started is unreadable here. `stop`
  # cannot read it either, so it must not block the reinstall; the reinstall
  # says it could not tell.
  @id:comfyui-reinstall-keeps-user-data @requires-os:linux
  Scenario: comfyui-05 - Reinstalling ComfyUI replaces its code and keeps the user's own files
    Given a ComfyUI install holding the user's models, workflows, images and custom nodes
    And a newer ComfyUI release is available to download
    And rocm-cli's saved record of the ComfyUI it started cannot be read
    When the user reinstalls ComfyUI
    Then the reinstall reports the user's folders as kept and they still hold the user's files
    And ComfyUI's code is the newer release
    And the reinstall lists the code it replaced
    And the reinstall says it could not tell whether ComfyUI was running

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
  # stopped. The dry run says so first. The refusal names `rocm comfyui stop`;
  # the scenario runs exactly that and proves the same reinstall then goes
  # through. Once it is reinstalled, a ComfyUI started from it that has not
  # answered on its port yet is "starting" to a plain-words status question,
  # which must not advise starting a second one. (A ComfyUI that is still
  # starting also blocks a reinstall; the unit test
  # `reinstall_is_refused_while_comfyui_starts_until_it_is_stopped` covers it.)
  #
  # Steps that set up state in the middle of a scenario are `Given`, as in
  # comfyui-09 and comfyui-11.
  @id:comfyui-reinstall-refused-while-running @requires-os:linux
  Scenario: comfyui-08 - A ComfyUI reinstall waits until the running ComfyUI is stopped
    Given a ComfyUI install holding the user's models, workflows, images and custom nodes
    And a newer ComfyUI release is available to download
    And the ComfyUI that rocm-cli started is running from that install and answering on its port
    When the user previews reinstalling ComfyUI
    Then the preview says the reinstall is refused until ComfyUI is stopped
    And the existing ComfyUI code and the user's files are untouched
    When the user reinstalls ComfyUI
    Then the reinstall is refused and names rocm comfyui stop
    And the reinstall was refused before it downloaded anything
    And the existing ComfyUI code and the user's files are untouched
    When the user stops ComfyUI
    And the user reinstalls ComfyUI
    Then the reinstall reports the user's folders as kept and they still hold the user's files
    And ComfyUI's code is the newer release
    Given a ComfyUI started by rocm-cli is still loading from that install
    When the user asks in plain words for ComfyUI's status
    Then the answer says ComfyUI is installed and starting and does not advise starting it

  # A reinstall killed while it replaces the code leaves a half-replaced
  # folder. `start` must not launch it; it names the install command that
  # finishes the reinstall, and the scenario runs that exact command (read from
  # the CLI's own message) and proves the condition is cleared. `status` and
  # the answer to a natural-language status question both name that command
  # rather than `start` until then.
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
    And the plain-words status answer names that command instead of start
    When the user runs the command start named
    Then ComfyUI status no longer reports an interrupted reinstall
    And the plain-words status answer no longer reports an interrupted reinstall
    And ComfyUI's code is the newer release
    And the user's own files are untouched

  # A newer release can start shipping a name the user already uses at the top
  # of `source/`. The install recorded what its release shipped, so the
  # reinstall knows that entry is the user's: it renames it and says where,
  # instead of deleting it as old code.
  @id:comfyui-reinstall-sets-aside-colliding-entry @requires-os:linux
  Scenario: comfyui-10 - Reinstalling ComfyUI renames a folder of the user's that the new release also ships
    Given a ComfyUI install holding the user's models, workflows, images and custom nodes
    And the user keeps an app folder there that the installed ComfyUI release did not ship
    And a newer ComfyUI release that ships an app folder is available to download
    When the user reinstalls ComfyUI
    Then the reinstall reports the user's folders as kept and they still hold the user's files
    And the reinstall reports the user's app folder as set aside and it still holds the user's files
    And ComfyUI's code is the newer release

  # A reinstall can fail after the new code is in place: here the runtime's
  # GPU check fails. The success report that would have listed what was
  # replaced is never printed, so the error names it, together with the
  # command that finishes the install. The scenario runs that command, read
  # from the error, once the GPU check passes again.
  @id:comfyui-reinstall-failing-after-swap-names-the-finish @requires-os:linux
  Scenario: comfyui-11 - A ComfyUI reinstall that fails after replacing the code says what it replaced and how to finish
    Given a ComfyUI install holding the user's models, workflows, images and custom nodes
    And a newer ComfyUI release is available to download
    And the runtime's AMD GPU check fails
    When the user reinstalls ComfyUI
    Then the reinstall fails and names what it replaced and the command that finishes the install
    And ComfyUI's code is the newer release
    And the user's own files are untouched
    Given the runtime's AMD GPU check passes again
    When the user runs the command the reinstall named
    Then ComfyUI is installed and its AMD GPU check is ready
    And the user's own files are untouched

  # The reinstall checks that no ComfyUI runs from the folder before it
  # downloads, and again just before it replaces the code: a ComfyUI started
  # during a long download would otherwise be running the code the swap
  # replaces. Here it is started once the download has begun, so only the
  # second check can catch it. `rocm comfyui stop`, which the refusal names,
  # then lets the same reinstall through.
  @id:comfyui-reinstall-rechecks-running-before-swap @requires-os:linux
  Scenario: comfyui-12 - A ComfyUI started while a reinstall downloads stops the reinstall before it replaces any code
    Given a ComfyUI install holding the user's models, workflows, images and custom nodes
    And a newer ComfyUI release is available, and ComfyUI is started from that install while it downloads
    When the user reinstalls ComfyUI
    Then the reinstall is refused and names rocm comfyui stop
    And the newer release had downloaded before the reinstall was refused
    And the existing ComfyUI code and the user's files are untouched
    When the user stops ComfyUI
    And the user reinstalls ComfyUI
    Then the reinstall reports the user's folders as kept and they still hold the user's files
    And ComfyUI's code is the newer release

  # A download that unpacks but has no main.py is not installed: the reinstall
  # would otherwise remove the installed code and put nothing startable in its
  # place. It is not kept either, so the next attempt downloads again.
  @id:comfyui-reinstall-rejects-download-without-main-py @requires-os:linux
  Scenario: comfyui-13 - A ComfyUI reinstall whose download is not a ComfyUI release leaves the install untouched
    Given a ComfyUI install holding the user's models, workflows, images and custom nodes
    And the newer ComfyUI release offered for download has no main.py
    When the user reinstalls ComfyUI
    Then the reinstall fails saying the download is not a ComfyUI release, and does not keep it
    And the existing ComfyUI code and the user's files are untouched
