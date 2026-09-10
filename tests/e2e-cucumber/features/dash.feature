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
    # must NOT immediately clear the displayed "tok/s" value.  The contract
    # requires the held value to remain visible for the validity window
    # clamp(3 x instance_tick, 6 s, 30 s).  Current code has no such window
    # (runner.rs clears gen_tps on the same tick as the failure), so the
    # "generation throughput remains visible" step is the RED assertion.
    Given a managed model exposes scripted serving metrics
    When the user opens the dashboard
    And the user opens the Observe view
    Then positive generation throughput is displayed for the managed model
    When the metrics endpoint fails transiently
    Then generation throughput remains visible within the validity window
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-gen-tps-expiry-boundary @requires-os:linux
  Scenario: dash-09 - Gen throughput expires after the validity window following sustained failure
    # EAI-7960 expiry-boundary scenario: two contract boundaries are pinned.
    #
    # BOUNDARY 1 (held assertion) — immediately after the first failed scrape,
    # gen_tps must still be visible (Held).  With current code this FAILS (RED)
    # because runner.rs clears gen_tps immediately.
    #
    # BOUNDARY 2 (expired assertion) — after the validity window elapses
    # (clamp(3 × instance_tick, 6 s, 30 s) = 6 s for the production 2 s tick),
    # gen_tps must be gone from the screen.  This step is unreachable today
    # because BOUNDARY 1 fails first; it becomes GREEN once the fix is applied.
    Given a managed model exposes scripted serving metrics
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

  @id:dash-sigterm-restores-terminal @requires-os:linux
  Scenario: dash-11 - A SIGTERM restores the terminal and exits 143
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
  Scenario: dash-12 - A SIGINT restores the terminal and exits 130
    # An externally delivered SIGINT (`kill -INT` from another process) takes the
    # same restore path and reports the conventional 128+2 exit code. This is NOT
    # the typed Ctrl-C gesture: raw mode clears ISIG, so that keystroke never
    # becomes a signal — dash-14 covers it as the key event it actually is.
    When the user opens the dashboard with demo data
    Then the dashboard home view is displayed
    When the dashboard receives a SIGINT
    Then the dashboard exits from the signal with code 130
    And the terminal is restored to the normal screen

  @id:dash-launcher-sigterm-restores-terminal-across-a-session @requires-os:linux
  Scenario: dash-13 - A SIGTERM to the launcher hub restores the terminal after a session
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
  Scenario: dash-14 - Typing Ctrl-C in the dashboard restores the terminal and exits 130
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
  Scenario: dash-15 - Typing Ctrl-C at the launcher front door restores the terminal and exits 130
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
