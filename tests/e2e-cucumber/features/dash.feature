Feature: Interactive dashboard

  # These scenarios drive the real interactive TUI through a pseudo-terminal —
  # the crossterm raw-mode event loop that a piped command can't reach. Linux
  # only for now: portable-pty compiles on Windows via ConPTY, but that path is
  # not yet promoted to a blocking contract (tracked as a follow-up).

  # #534: the ROCm view's `Runs:` line is a command a user can copy into a
  # shell, and it named `rocm doctor`, a subcommand that does not exist. The
  # unit test beside the action list checks each label against what its screen
  # runs; the diagnose steps here check the line the user actually sees. They
  # extend this journey rather than open another dashboard session, which would
  # add load to the no-GPU lane's timing-sensitive dash scenarios.
  @id:dash-opens-and-navigates @requires-os:linux
  Scenario: dash-01 - A user opens the dashboard and navigates to ROCm setup
    When the user opens the dashboard with demo data
    Then the dashboard home view is displayed
    When the user opens the ROCm view
    Then ROCm setup actions are displayed
    When the user selects the ROCm diagnose action
    Then the diagnose action says it runs rocm examine
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-chat-offline-reply @requires-os:linux
  Scenario: dash-02 - A user receives a response in interactive chat
    Given interactive chat uses an offline assistant
    When the user opens interactive chat
    And the user sends a message about GPU health
    Then the assistant's GPU status response is displayed
    When the user quits interactive chat
    Then interactive chat exits successfully

  @id:dash-loading-service-status @requires-os:linux
  Scenario: dash-03 - The dashboard reports a model that is still loading as loading
    Given a managed model is still loading
    When the user opens the dashboard
    And the user opens the Observe view
    Then the managed model is shown as loading rather than ready
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-managed-service-metrics @requires-os:linux
  Scenario: dash-04 - Observe displays metrics from a managed model
    Given a managed model exposes serving metrics
    When the user opens the dashboard
    And the user opens the Observe view
    Then live serving metrics are displayed for the managed model
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-help-guidance @requires-os:linux
  Scenario: dash-05 - A user can discover dashboard help and next-step guidance
    When the user opens the dashboard with demo data
    And the user opens dashboard help
    Then navigation and next-step guidance are displayed
    And the backdrop behind the popup is dimmed
    When the user closes dashboard help
    And the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-command-palette-navigation @requires-os:linux
  Scenario: dash-06 - A user navigates to Serving through the command palette
    When the user opens the dashboard with demo data
    And the user opens the command palette
    Then dashboard destinations are displayed
    When the user chooses Serving
    Then Serving actions are displayed
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-managed-service-visible @requires-os:linux
  Scenario: dash-07 - A managed model is visible in the dashboard
    Given a running managed model is available locally
    When the user opens the dashboard
    And the user opens the Observe view
    Then the managed model is displayed
    When the user opens the managed services overlay
    Then the Services overlay lists the managed model with its port
    When the user asks to stop the listed service
    Then a stop approval prompt names the managed model and its port
    When the user declines the approval prompt
    And the user closes the managed services overlay
    And the user quits the dashboard
    Then the dashboard exits successfully


  @id:dash-gen-tps-held-after-scrape-failure @requires-os:linux
  Scenario: dash-08 - Gen throughput stays visible for the validity window after a scrape failure
    # EAI-7960 principal regression: after establishing a positive gen_tps
    # baseline through the scripted mock, a single /metrics transport failure
    # must NOT immediately clear the displayed "tok/s" value. Observation time is
    # held before the failure is injected, because a free-running logical clock
    # still advances with the daemon's wall-clock-paced cycles: a scenario
    # descheduled on a loaded runner would otherwise reach the assertion below
    # after the window had honestly expired and call that a regression.
    Given a managed model exposes scripted serving metrics
    And dashboard observation time is deterministic
    When the user opens the dashboard
    And the user opens the Observe view
    Then positive generation throughput is displayed for the managed model
    When dashboard observation time is held
    And the metrics endpoint fails transiently
    Then generation throughput remains visible within the validity window
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-gen-tps-expiry-boundary @requires-os:linux
  Scenario: dash-09 - Gen throughput expires after the validity window following sustained failure
    # EAI-7960 expiry-boundary scenario: immediately after the first failed
    # scrape, gen_tps remains visible as Held. Stepping the held clock past
    # clamp(3 × instance_tick, 6 s, 30 s) then makes the daemon publish an
    # expired value. Neither boundary is defined by wall time: the clock is held
    # across the first, and only this scenario's explicit step crosses the second.
    Given a managed model exposes scripted serving metrics
    And dashboard observation time is deterministic
    When the user opens the dashboard
    And the user opens the Observe view
    Then positive generation throughput is displayed for the managed model
    When dashboard observation time is held
    And the metrics endpoint fails transiently
    Then generation throughput remains visible within the validity window
    When the validity window has elapsed
    Then generation throughput is no longer displayed
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-launcher-shows-live-serving-instance @requires-os:linux
  Scenario: dash-10 - The launcher front door shows a live serving model rather than idle
    # EAI-8190 regression: bare `rocm` opens the launcher front door, which
    # reads the managed-service registry (`launcher_serving_instances`) the same
    # way `rocm services` does. A model already serving must surface as
    # "Serving <model>", not the "Idle — nothing serving" state the front door
    # showed before the fix, which drove this whole PR.
    Given a running managed model is available locally
    When the user opens the launcher
    Then the launcher shows the model serving
    When the user quits the launcher
    Then the launcher exits successfully

  # Characterization coverage: this scenario observes that Escape closes the
  # manager on a non-domain tab, but the services manager's own event-loop arm
  # would close it on root Esc even without the tab-independent back-out path
  # this PR generalized, so a revert of that change would not turn this red.
  # The discriminating regression test for that change is the unit test
  # `back_out_requires_an_open_manager_on_any_tab` in crates/rocm-dash-tui's
  # app/mod.rs, which does fail on revert.
  @id:dash-manager-escape-closes-on-any-tab @requires-os:linux
  Scenario: dash-11 - Escape closes a manager overlay on any tab
    When the user opens the dashboard with demo data
    And the user opens the Observe view
    And the user opens the services manager
    Then the services manager is displayed
    When the user presses Escape
    Then the services manager is closed
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-chat-approval-defaults-to-deny @requires-os:linux
  Scenario: dash-12 - A surfaced tool call defaults to Deny and confirming without moving denies it
    Given interactive chat uses an offline assistant
    When the user opens interactive chat
    And the user sends a message that triggers a tool approval
    Then a tool approval prompt is displayed
    When the user confirms the approval prompt without moving the cursor
    Then the tool call is shown as declined
    When the user quits interactive chat
    Then interactive chat exits successfully

  @id:dash-instance-detail-dims-backdrop @requires-os:linux
  Scenario: dash-13 - Opening instance detail dims the screen behind the popup
    When the user opens the dashboard with demo data
    And the user opens the Observe view
    And the user opens instance detail
    Then instance details are displayed
    And the backdrop behind the popup is dimmed
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-chat-idle-escape-opens-menu @requires-os:linux
  Scenario: dash-14 - Escape opens the menu when idle on the Chat tab
    When the user opens the dashboard with demo data
    And the user opens the Chat view
    When the user presses Escape
    Then the dashboard menu is displayed
    When the user presses Escape
    Then the dashboard menu is closed
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-theme-picker-dims-backdrop @requires-os:linux
  Scenario: dash-15 - Opening the theme picker dims the screen behind it
    When the user opens the dashboard with demo data
    And the user opens the theme picker
    Then the backdrop behind the popup is dimmed
    When the user quits the dashboard
    Then the dashboard exits successfully

  # EAI-8366: `--replay <missing>` must fail fast — validate the path BEFORE the
  # dashboard takes over the terminal, printing a clear error and exiting
  # non-zero. Driven through a PTY (like the rest of this file): the fail-fast
  # property is unobservable through a pipe, and under a real terminal the pre-fix
  # binary enters the alt-screen and hangs, which this scenario pins.
  @id:dash-replay-missing-file-fails-fast @requires-os:linux
  Scenario: dash-16 - Replaying a missing recording fails before entering the dashboard
    When the user replays a recording that does not exist
    Then the dashboard is refused before taking over the terminal
    And the user is told the replay file was not found

  @id:dash-sigterm-restores-terminal @requires-os:linux
  Scenario: dash-17 - A SIGTERM restores the terminal and exits 143
    # Core regression for this PR: a SIGTERM to a running dashboard (e.g. a
    # supervisor stopping it) must run the restore path — leave the alternate
    # screen and show the cursor — and report the conventional 128+15 exit code,
    # rather than dying on the default disposition and leaving a broken terminal.
    When the user opens the dashboard with demo data
    Then the dashboard home view is displayed
    When the dashboard receives a SIGTERM
    Then the dashboard exits from the signal with code 143
    And the terminal is restored to the normal screen

  @id:dash-sigint-restores-terminal @requires-os:linux
  Scenario: dash-18 - A SIGINT restores the terminal and exits 130
    # An externally delivered SIGINT (`kill -INT` from another process) takes the
    # same restore path and reports the conventional 128+2 exit code. This is NOT
    # the typed Ctrl-C gesture: raw mode clears ISIG, so that keystroke never
    # becomes a signal — dash-20 covers it as the key event it actually is.
    When the user opens the dashboard with demo data
    Then the dashboard home view is displayed
    When the dashboard receives a SIGINT
    Then the dashboard exits from the signal with code 130
    And the terminal is restored to the normal screen

  @id:dash-launcher-sigterm-restores-terminal-across-a-session @requires-os:linux
  Scenario: dash-19 - A SIGTERM to the launcher hub restores the terminal after a session
    # EAI-7194 launcher-hub regression: bare `rocm` is a persistent hub whose
    # process outlives each session's Tokio runtime. Tokio never unregisters the
    # libc signal handler it installs, so a per-session watcher goes deaf the
    # moment its runtime is dropped — leaving the synchronous launcher menu
    # (itself in raw mode) unable to restore the terminal on a SIGTERM delivered
    # after the user's first flow: an unkillable, worse form of the bug this PR
    # fixes. A single process-lifetime watcher, installed once for the whole hub,
    # must keep every window killable. The process under test here is the
    # launcher hub, not a dashboard, so the outcome steps are the launcher's.
    # This drives a full session round-trip — open the dashboard, quit back to
    # the menu — before signalling, so it exercises the across-session path a
    # single-session scenario cannot. (The startup ordering — listeners
    # registered before raw mode — is covered by construction:
    # `spawn_termination_watcher` is called before `enable_raw_mode`; a 20
    # ms-polled PTY scenario cannot observe that microsecond window, so none is
    # claimed for it.)
    When the user opens the launcher
    Then the launcher front door is displayed
    When the user opens the dashboard from the launcher
    Then the dashboard home view is displayed
    When the user quits back to the launcher
    Then the launcher front door is displayed
    When the launcher receives a SIGTERM
    Then the launcher exits from the signal with code 143
    And the terminal is restored to the normal screen

  @id:dash-ctrl-c-restores-terminal @requires-os:linux
  Scenario: dash-20 - Typing Ctrl-C in the dashboard restores the terminal and exits 130
    # The gesture a user actually performs, and the one nothing covered. While
    # the TUI holds the terminal in raw mode the driver's ISIG translation is off
    # (ENABLE_PROCESSED_INPUT on Windows), so this keystroke is delivered to the
    # process as the byte 0x03 — an ordinary key event — and never becomes a
    # SIGINT. The signal watcher therefore cannot see it: before this was handled
    # as a key, pressing Ctrl-C in the dashboard did nothing at all and left the
    # user in a raw-mode terminal. The step sends the literal byte, not a signal,
    # so it fails if the handling regresses to relying on the signal path.
    When the user opens the dashboard with demo data
    Then the dashboard home view is displayed
    When the user presses Ctrl-C in the dashboard
    Then the dashboard exits from the keystroke with code 130
    And the terminal is restored to the normal screen

  @id:dash-launcher-ctrl-c-restores-terminal @requires-os:linux
  Scenario: dash-21 - Typing Ctrl-C at the launcher front door restores the terminal and exits 130
    # The same keystroke at the hub's synchronous menu. That loop is a separate
    # key loop from the dashboard's and had no Ctrl-C handling at all, so the
    # gesture left bare `rocm` sitting at the front door in raw mode. Both loops
    # must route it through the one restore path, or the key means different
    # things in the two windows of the same process.
    When the user opens the launcher
    Then the launcher front door is displayed
    When the user presses Ctrl-C in the launcher
    Then the launcher exits from the keystroke with code 130
    And the terminal is restored to the normal screen

  @id:dash-instance-detail-scroll-hint @requires-os:linux
  Scenario: dash-22 - Instance detail shows a scroll hint when content overflows
    # Opening the Observe view already enlarges the terminal (so other
    # journeys can assert the detail popup's full layout); even the demo
    # fixtures' launch_args/env_vars overflow that size, but this scenario
    # needs the args/env panes to overflow specifically once shrunk further,
    # so the assertion right before the shrink pins that starting point.
    When the user opens the dashboard with demo data
    And the user opens the Observe view
    And the user opens instance detail
    Then the instance detail footer does not show the scroll hint
    When the user shrinks the terminal until the detail body overflows
    Then the instance detail footer shows the scroll hint
    And the instance detail body shows a scrollbar
    When the user quits the dashboard
    Then the dashboard exits successfully

  # #75/#441: the Configure step's Folder row (`Tab` opens the shared
  # FolderBrowser) is a unit/snapshot-tested render, but nothing before this
  # drove it through the real crossterm event loop. The final Then is not a
  # restatement of "the browser opened" — that only proves the popup rendered,
  # not that confirming a folder actually changes what Configure stages, which
  # is the behaviour #75 restores.
  @id:dash-onboarding-configure-folder-browse @requires-os:linux
  Scenario: dash-23 - The onboarding Configure step opens the install-folder browser
    When the user opens the dashboard with demo data
    And the user opens the Observe view
    And the user opens onboarding setup
    Then the onboarding welcome screen is displayed
    When the user continues past the onboarding welcome screen
    Then the onboarding setup choices are displayed
    When the user chooses to install the ROCm SDK
    Then the SDK Configure step is displayed
    When the user browses for an install folder
    Then the install-folder browser is displayed
    When the user chooses the current folder
    Then the Configure step shows the chosen folder instead of the placeholder
    When the user closes onboarding setup
    And the user quits the dashboard
    Then the dashboard exits successfully

  # EAI-8450: the amd-smi device pre-flight only ever checked bare-metal
  # `/dev/kfd`, so under WSL2 (no such device node — GPU access goes through
  # WSLg's `/dev/dxg` instead) the dashboard always reported no GPU, even with
  # a real one passed through. `@requires-wsl` rather than `@requires-gpu`
  # alone: the premise under test is the WSL detection path specifically.
  # Currently skipped on every CI lane — no WSL lane carries a real GPU yet
  # (tracked separately in #223) — but still runs locally on WSL hardware.
  @id:dash-reports-wsl-gpu-telemetry @requires-wsl @requires-gpu
  Scenario: dash-24 - Dashboard displays GPU telemetry through WSL
    When the user opens the dashboard
    Then the dashboard reports live GPU telemetry
    When the user quits the dashboard
    Then the dashboard exits successfully

  # #501: proves TuiSession::spawn_in (see its doc comment) threads the
  # given directory into the live child. Spawns into a scenario-owned
  # folder, plants a child directory with a space in its name, and asserts
  # the real folder browser lists that child by its fixed-length entry
  # label rather than the host-length-dependent header path (see
  # ADOPT_SCENARIO_CHILD's doc comment). The space is in the child's name,
  # not the scenario's own cwd — it exercises the browser's whitespace
  # rendering; a whitespace path through spawn_in itself remains follow-up
  # work tracked in #693.
  @id:dash-onboarding-adopt-honors-spawn-cwd @requires-os:linux
  Scenario: dash-25 - Adopt existing opens the folder browser rooted at the session's own working directory
    When the user opens the dashboard in a scenario-owned folder
    And the user opens the Observe view
    And the user opens onboarding setup
    Then the onboarding welcome screen is displayed
    When the user continues past the onboarding welcome screen
    Then the onboarding setup choices are displayed
    When the user chooses to adopt an existing folder
    Then the existing-folder browser lists the scenario folder's contents
    When the user closes onboarding setup
    And the user quits the dashboard
    Then the dashboard exits successfully

  # #145: quitting used to leave a served model running in the background
  # with no warning at all. Declining first, then confirming, in the same
  # scenario pins both halves of the fix with one launch.
  @id:dash-quit-confirm-when-serving @requires-os:linux
  Scenario: dash-26 - Quitting while a model is being served asks for confirmation first
    Given a running managed model is available locally
    When the user opens the dashboard
    And the user opens the Observe view
    Then the managed model is displayed
    When the user tries to quit the dashboard
    Then the dashboard asks whether to quit while a model is still being served
    And the quit prompt shows the real stop command for the managed model
    When the user declines the quit prompt
    Then the managed model is displayed
    When the user tries to quit the dashboard
    Then the dashboard asks whether to quit while a model is still being served
    And the quit prompt offers to open Services for the serving model
    When the user presses s to manage the serving model
    Then the services overlay opens showing the managed model
    When the user closes the managed services overlay
    And the user quits the dashboard
    Then the dashboard exits successfully
    And the managed model is still listed as running

  # #145: the launcher front door is a second, independent quit entry point
  # from the dashboard's own — same bug (quitting silently left a served
  # model running), same fix. Mirrors dash-26 for the launcher's copy of the
  # prompt: declining first, then confirming, pins both halves with one launch.
  @id:dash-launcher-quit-confirm-when-serving @requires-os:linux
  Scenario: dash-27 - Quitting the launcher while a model is being served asks for confirmation first
    Given a running managed model is available locally
    When the user opens the launcher
    Then the launcher shows the model serving
    When the user tries to quit the launcher
    Then the launcher asks whether to quit while a model is still being served
    And the quit prompt shows the real stop command for the managed model
    When the user declines the quit prompt
    Then the launcher shows the model serving
    When the user quits the launcher
    Then the launcher exits successfully
    And the managed model is still listed as running

  # AGENTS.md §3: the quit prompt's remediation line names a real, fillable
  # command (`dash-26` already pins the exact string) — prove it actually
  # works, not just that the text is right. This runs the literal command a
  # user would copy from the prompt and checks the registry `rocm services`
  # itself reads, independent of whether the TUI's own live view has caught up
  # to the external change by the time the scenario ends.
  @id:dash-quit-prompt-stop-command-actually-works @requires-os:linux
  Scenario: dash-28 - Running the quit prompt's printed stop command actually stops the model
    Given a running managed model is available locally
    And the managed model's record points at a disposable process
    When the user opens the dashboard
    And the user opens the Observe view
    Then the managed model is displayed
    When the user tries to quit the dashboard
    Then the dashboard asks whether to quit while a model is still being served
    And the quit prompt shows the real stop command for the managed model
    When the user declines the quit prompt
    And the user runs the quit prompt's stop command
    Then the managed model is no longer listed as running
    And the disposable process has exited
    When the user quits the dashboard
    Then the dashboard exits successfully

  # #145: `/quit` in chat shares the exact same `request_quit` gate as the
  # dashboard's `q` and the launcher's own `q`, but had no scenario coverage of
  # its own before this — only the `q` path did. `rocm chat` is the same
  # dashboard with the Chat tab focused (`run_chat` calls the same
  # `run_async`), so the model being served here is exactly as real as it is
  # in dash-26.
  @id:dash-chat-quit-confirm-when-serving @requires-os:linux
  Scenario: dash-29 - Quitting chat with /quit while a model is being served asks for confirmation first
    Given a running managed model is available locally
    And interactive chat uses an offline assistant
    When the user opens interactive chat
    And the user tries to quit chat
    Then the dashboard asks whether to quit while a model is still being served
    When the user declines the quit prompt
    And the user sends a message about GPU health without refocusing the input
    Then the assistant's GPU status response is displayed
    When the user quits interactive chat
    Then interactive chat exits successfully
    And the managed model is still listed as running

  # #145: the launcher is a persistent hub — escalating into the full
  # dashboard ("Open full dashboard") and quitting FROM THERE used to ask
  # "May still be serving — quit anyway?" a second time, and approving it only
  # returned to the launcher menu, which then immediately asked the identical
  # question again on its own next `q` — a prompt whose "yes" didn't actually
  # exit the process at all. Only the launcher's own copy of the prompt may
  # ever govern the real exit; the escalated dashboard's own `q` must return
  # straight to the launcher menu with no prompt of its own.
  @id:dash-launcher-hosted-dashboard-quit-is-not-asked-twice @requires-os:linux
  Scenario: dash-30 - Quitting a dashboard opened from the launcher does not ask twice
    Given a running managed model is available locally
    When the user opens the launcher
    Then the launcher shows the model serving
    When the user opens the dashboard from the launcher
    And the user quits back to the launcher
    Then the launcher shows the model serving
    When the user tries to quit the launcher
    Then the launcher asks whether to quit while a model is still being served
    When the user quits the launcher
    Then the launcher exits successfully
    And the managed model is still listed as running

  # #145: `run_chat`'s own `launcher_hosted` flag must get the exact same
  # double-prompt suppression `run`'s does (dash-30) — escalating into Chat
  # from the launcher (the other hosted route, `launcher_route_hosts_session`)
  # and quitting FROM THERE must return straight to the launcher menu with no
  # quit-confirm prompt of its own; only the launcher's own copy may ever
  # govern the real exit.
  @id:dash-launcher-hosted-chat-quit-is-not-asked-twice @requires-os:linux
  Scenario: dash-31 - Quitting chat opened from the launcher does not ask twice
    Given a running managed model is available locally
    When the user opens the launcher
    Then the launcher shows the model serving
    When the user opens chat from the launcher
    And the user quits chat back to the launcher
    Then the launcher shows the model serving
    When the user tries to quit the launcher
    Then the launcher asks whether to quit while a model is still being served
    When the user quits the launcher
    Then the launcher exits successfully
    And the managed model is still listed as running

  # README's "First run" section promises `Ctrl-C` always quits immediately
  # without asking, even while a model is still being served — unlike
  # `q`/Esc/`/quit`, which all route through the confirm-before-quit gate
  # (dash-26). dash-20 already covers the keystroke with nothing serving;
  # this is the "will not happen" half AGENTS.md §3 asks for: with a model
  # serving, Ctrl-C must still exit on the spot rather than opening the
  # prompt dash-26 pins — both before that prompt has been triggered, and
  # (the production event loop puts the Ctrl-C arm ahead of the quit-confirm
  # arm in its match, a relationship only re-opening the dashboard and
  # triggering the prompt first can actually exercise) while it is already
  # open. `press_ctrl_c_and_wait` waits for process exit, so a quit-confirm
  # prompt intercepting the keystroke instead would time either half out, not
  # silently pass it.
  @id:dash-ctrl-c-skips-the-quit-confirm-while-serving @requires-os:linux
  Scenario: dash-32 - Ctrl-C quits immediately even while a model is being served
    Given a running managed model is available locally
    When the user opens the dashboard
    And the user opens the Observe view
    Then the managed model is displayed
    When the user presses Ctrl-C in the dashboard
    Then the dashboard exits from the keystroke with code 130
    And the terminal is restored to the normal screen
    And the managed model is still listed as running
    When the user opens the dashboard
    Then the dashboard home view is displayed
    When the user tries to quit the dashboard
    Then the dashboard asks whether to quit while a model is still being served
    When the user presses Ctrl-C in the dashboard
    Then the dashboard exits from the keystroke with code 130
    And the managed model is still listed as running

  # `quit_and_wait`'s own retry loop answers "y" to any quit-confirm prompt it
  # sees (`tui_driver.rs`), so every other scenario that quits through it
  # would silently swallow a regression that opened the prompt when nothing
  # is being served — a corrupted `AppState::has_live_instance` falsely
  # reporting live would just get auto-confirmed away. This is the one
  # scenario that bypasses that helper (`the user tries to quit the
  # dashboard` sends a bare `q` and resolves nothing) and asserts the
  # negative directly: with `--demo` serving no model at all, `q` must exit
  # on the very first keystroke, with the quit-confirm prompt never opening
  # to be auto-answered.
  @id:dash-quit-skips-confirm-with-nothing-serving @requires-os:linux
  Scenario: dash-33 - Quitting with nothing serving exits immediately, without asking
    When the user opens the dashboard with demo data
    Then the dashboard home view is displayed
    When the user tries to quit the dashboard
    Then the dashboard exits immediately with no quit confirmation

  # #145 / AGENTS.md §3: the Esc-menu "Quit" row shares `request_quit` with
  # every other quit gesture (`actions.rs`'s `KeyAction::MenuActivate` arm),
  # but until now that sharing was only proven at the unit level — mutating
  # the row's own `return request_quit(state)` back to `return true` fails no
  # scenario, only unit tests (`quitting_from_the_esc_menu_opens_the_same_confirm_prompt`
  # et al). This scenario reaches the row the way a user would: Esc opens the
  # menu, and navigating to and activating "Quit" asks the same quit-confirm
  # prompt every other entry point does.
  @id:dash-esc-menu-quit-row-confirms-when-serving @requires-os:linux
  Scenario: dash-34 - Choosing Quit from the Esc menu while a model is being served asks for confirmation first
    Given a running managed model is available locally
    When the user opens the dashboard
    And the user opens the Observe view
    Then the managed model is displayed
    When the user presses Escape
    Then the dashboard menu is displayed
    When the user selects Quit from the dashboard menu
    Then the dashboard asks whether to quit while a model is still being served
    When the user declines the quit prompt
    Then the managed model is displayed
    When the user quits the dashboard
    Then the dashboard exits successfully
    And the managed model is still listed as running

  # gpu_compact_line (the one-line-per-GPU row the Hardware/Observe GPU section
  # renders once it's too tight for the full bordered panels — see
  # `plan_gpu_rows` in hardware.rs) divided vram_used_mb/vram_total_mb by 1024
  # (binary arithmetic) but printed the result labeled "GB" (implying SI/
  # decimal units) instead of "GiB" (#673 — the same mislabeling class as #190,
  # pinned for a different formatter in automations-05). The demo session's 8
  # GPUs already force this row onto the compact layout at any terminal size
  # the other dash scenarios use, but the Observe tab nests the GPU section
  # inside a 35%-height split of the tab body, which stays too short to show
  # any GPU rows at all until the terminal is grown well past the sizes those
  # other scenarios need — hence the extra resize here.
  @id:dash-gpu-vram-uses-binary-units @requires-os:linux
  Scenario: dash-35 - The compact GPU row shows VRAM in binary units
    When the user opens the dashboard with demo data
    And the user opens the Observe view
    And the user grows the terminal until the GPU section shows compact rows
    Then the GPU section reports VRAM in binary units
    When the user quits the dashboard
    Then the dashboard exits successfully

  # #693: dash-25 proves `spawn_in` threads a scenario-owned folder into the
  # real browser, but its whitespace case lives in a CHILD entry under that
  # folder (`ADOPT_SCENARIO_CHILD`), not in the `spawn_in` cwd itself — so it
  # never drives #500's `quote_display_arg` fix through the real
  # approval-preview render a user actually sees. Here the scenario's own
  # root IS the whitespace path: `spawn_in`'s cwd has a space in its own
  # name, onboarding's Adopt flow resolves its browser start from that exact
  # cwd (`std::env::current_dir()` in `activate_choice`), and choosing
  # "[ use this folder ]" stages an approval whose `--root` argument is that
  # same whitespace path. #500 already has widget-level `TestBackend`
  # snapshot coverage of the quoting in isolation
  # (`approval_snapshot_quotes_args_containing_spaces` in
  # `crates/rocm-dash-tui/tests/wave0_job_bridge.rs`); this is the missing
  # black-box proof that the real wizard renders it quoted too.
  @id:dash-onboarding-adopt-whitespace-cwd-quotes-approval-preview @requires-os:linux
  Scenario: dash-36 - Adopting a folder whose own path contains a space quotes it in the approval preview
    When the user opens the dashboard in a scenario folder whose own name has a space
    And the user opens the Observe view
    And the user opens onboarding setup
    Then the onboarding welcome screen is displayed
    When the user continues past the onboarding welcome screen
    Then the onboarding setup choices are displayed
    When the user chooses to adopt an existing folder
    Then the existing-folder browser is displayed
    When the user chooses the current folder
    Then the approval preview quotes the whitespace-containing root path
    When the user denies the adopt approval and closes onboarding setup
    And the user quits the dashboard
    Then the dashboard exits successfully

  # The Diagnose screen offers the same Doctor report the CLI prepares, and the
  # review comes first. Whether a report can be prepared is a property of the
  # machine, not of the dashboard: hardware that is not on AMD's published list,
  # an architecture that cannot be read, and a WSL host all get a refusal, and
  # only an approved architecture gets a report. So this scenario does not pin
  # which outcome this machine produces. It reads whether the CLI refused from
  # the preview, then checks the card against that: a report is offered for
  # sending exactly when the CLI printed one, and a refusal shows the CLI's own
  # explanation. The card puts that outcome first. This scenario runs at the
  # detail size that "the user opens the Observe view" selects (120x40); the
  # cramped 80x24 case is covered by the unit test in
  # `crates/rocm-dash-tui/src/ui/examine_manager.rs`.
  #
  # The scenario opens the send review with `s` and then declines it. It never
  # approves, so no mail client is ever opened from a test. What the review must
  # say before anyone can approve is checked here: the mail goes from the user's
  # own account and carries their address, stated above the report so it is on
  # screen with Approve. On a lane where the CLI refused, `s` opens nothing, and
  # that is checked instead. The approve path, and the promise that no send
  # starts before an approve, are asserted where the key sequence can be driven
  # directly: the unit tests in `crates/rocm-dash-tui/src/ui/examine_manager.rs`.
  @id:dash-doctor-report-is-reviewed-before-sending @requires-os:linux
  Scenario: dash-37 - A user reviews a Doctor report in the dashboard before anything is sent
    When the user opens the dashboard with demo data
    And the user opens the Observe view
    And the user opens the Diagnose screen
    And the user prepares a Doctor report
    Then the report is displayed for review
    When the user leaves the report review
    Then the dashboard offers to send a report only if the CLI prepared one
    When the user asks to review the report for sending
    Then the review says the mail carries the user's own address before showing the report
    When the user declines the send
    And the user closes the Diagnose screen
    And the user quits the dashboard
    Then the dashboard exits successfully

  # #286: the header's warning badge counted warnings but gave no way to read
  # them, and while a GPU is reported the text is drawn nowhere else in the
  # dashboard. A recording supplies one warning next to a working GPU so the
  # scenario needs no daemon and no particular host hardware; the user reads
  # the text from the badge with `!`. The click on the badge, and its refusal
  # over an open modal, a manager overlay or focused chat input, are asserted by
  # the unit tests in `crates/rocm-dash-tui/src/app/` and `ui/mod.rs`, where a
  # mouse event can be sent directly. The key needs no such gate: it is only
  # reached when nothing else owns the keyboard, which is the event loop's
  # existing arm order and is not re-tested here.
  @id:dash-warning-badge-opens-the-warning @requires-os:linux
  Scenario: dash-38 - A user reads the warning behind the header badge
    When the user opens the dashboard replaying a recording that reports a warning
    Then the header shows that one warning is reported
    When the user opens the warnings
    Then the warning text is displayed
    And the backdrop behind the popup is dimmed
    When the user presses Escape
    Then the warning text is no longer displayed
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-lemonade-managed-inference-journey @requires-os:linux @requires-gpu @requires-engine:lemonade
  Scenario: dash-39 - A real managed Lemonade GPU model responds to inference and appears in the dashboard
    Given a managed runtime is active
    And the dashboard watches only managed services
    And a GGUF model is being served on lemonade
    Then the model responds to inference requests
    When the user opens the dashboard
    And the user opens the Observe view
    Then the real managed Lemonade model is displayed
    When the user opens instance detail
    Then the managed model's instance detail shows it ready on the port it was served on
    When the user closes instance detail
    And the user selects engine status in the Serving view
    Then Lemonade is shown as a managed runtime installation
    When the user quits the dashboard
    Then the dashboard exits successfully
    And the model responds to inference requests

  @id:dash-lemonade-external-protocol-discovery @requires-os:linux
  Scenario: dash-40 - External Lemonade protocol discovery displays every loaded model without adopting the service
    Given an external Lemonade server has two models loaded
    When the user opens the dashboard
    And the user opens the Observe view
    Then both loaded models are displayed with throughput shown as unknown
    When the user opens the managed services overlay
    Then the Services overlay has no managed servers
    When the user attempts to stop and restart the external models
    Then the Services overlay closes without an approval prompt or job
    And both loaded models are still displayed
    And the external server received no lifecycle request and no managed record exists
    When the user selects engine status in the Serving view
    Then Lemonade is shown as externally running without a managed installation
    When the user quits the dashboard
    Then the dashboard exits successfully
    And the external server received no lifecycle request and no managed record exists

  @id:dash-lemonade-external-protocol-transitions @requires-os:linux
  Scenario: dash-41 - External Lemonade protocol models disappear on unload, return on reload, and disappear again when the server stops responding
    Given an external Lemonade server has two models loaded
    When the user opens the dashboard
    And the user opens the Observe view
    Then both loaded models are displayed with throughput shown as unknown
    When the external Lemonade server unloads its second model
    Then only the first model is displayed, with its throughput
    When the external Lemonade server unloads every model
    Then no loaded model is displayed and none are reported as discovered
    When the external Lemonade server loads both models again
    Then both loaded models are displayed with throughput shown as unknown
    And the external server received no lifecycle request and no managed record exists
    When the external Lemonade server stops responding
    Then no loaded model is displayed and none are reported as discovered
    When the user selects engine status in the Serving view
    Then Lemonade returns to the managed installation status without external availability
    When the user quits the dashboard
    Then the dashboard exits successfully
