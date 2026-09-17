Feature: Interactive dashboard

  # These scenarios drive the real interactive TUI through a pseudo-terminal —
  # the crossterm raw-mode event loop that a piped command can't reach. Linux
  # only for now: portable-pty compiles on Windows via ConPTY, but that path is
  # not yet promoted to a blocking contract (tracked as a follow-up).

  @id:dash-opens-and-navigates @requires-os:linux
  Scenario: dash-01 - A user opens the dashboard and navigates to ROCm setup
    When the user opens the dashboard with demo data
    Then the dashboard home view is displayed
    When the user opens the ROCm view
    Then ROCm setup actions are displayed
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
    When the user scrolls to the end of dashboard help
    Then replay controls guidance is displayed
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
    When the user quits the dashboard
    Then the dashboard exits successfully


  @id:dash-gen-tps-held-after-scrape-failure @requires-os:linux
  Scenario: dash-08 - Gen throughput stays visible for the validity window after a scrape failure
    # EAI-7960 principal regression: after establishing a positive gen_tps
    # baseline through the scripted mock, a single /metrics transport failure
    # must NOT immediately clear the displayed "tok/s" value. The test-only
    # logical clock advances with daemon cycles rather than runner wall time, so
    # host scheduling cannot consume the validity window before this assertion.
    Given a managed model exposes scripted serving metrics
    And dashboard observation time is deterministic
    When the user opens the dashboard
    And the user opens the Observe view
    Then positive generation throughput is displayed for the managed model
    When the metrics endpoint fails transiently
    Then generation throughput remains visible within the validity window
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-gen-tps-expiry-boundary @requires-os:linux
  Scenario: dash-09 - Gen throughput expires after the validity window following sustained failure
    # EAI-7960 expiry-boundary scenario: immediately after the first failed
    # scrape, gen_tps remains visible as Held. Advancing the injected logical
    # clock beyond clamp(3 × instance_tick, 6 s, 30 s) then makes the next scrape
    # publish an expired value. No wall-clock sleep defines either boundary.
    Given a managed model exposes scripted serving metrics
    And dashboard observation time is deterministic
    When the user opens the dashboard
    And the user opens the Observe view
    Then positive generation throughput is displayed for the managed model
    When the metrics endpoint fails transiently
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
