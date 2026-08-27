Feature: Interactive dashboard

  # These scenarios drive the real interactive TUI through a pseudo-terminal —
  # the crossterm raw-mode event loop that a piped command can't reach. Linux
  # only for now: portable-pty compiles on Windows via ConPTY, but that path is
  # not yet promoted to a blocking contract (tracked as a follow-up).

  @id:dash-opens-and-navigates @requires-os:linux
  Scenario: 1 - A user opens the dashboard and navigates to ROCm setup
    When the user opens the dashboard with demo data
    Then the dashboard home view is displayed
    When the user opens the ROCm view
    Then ROCm setup actions are displayed
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-chat-offline-reply @requires-os:linux
  Scenario: 2 - A user receives a response in interactive chat
    Given interactive chat uses an offline assistant
    When the user opens interactive chat
    And the user sends a message about GPU health
    Then the assistant's GPU status response is displayed
    When the user quits interactive chat
    Then interactive chat exits successfully

  @id:dash-loading-service-status @requires-os:linux
  Scenario: 3 - The dashboard reports a model that is still loading as loading
    Given a managed model is still loading
    When the user opens the dashboard
    And the user opens the Observe view
    Then the managed model is shown as loading rather than ready
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-managed-service-metrics @requires-os:linux
  Scenario: 4 - Observe displays metrics from a managed model
    Given a managed model exposes serving metrics
    When the user opens the dashboard
    And the user opens the Observe view
    Then live serving metrics are displayed for the managed model
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-help-guidance @requires-os:linux
  Scenario: 5 - A user can discover dashboard help and next-step guidance
    When the user opens the dashboard with demo data
    And the user opens dashboard help
    Then navigation and next-step guidance are displayed
    When the user closes dashboard help
    And the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-command-palette-navigation @requires-os:linux
  Scenario: 6 - A user navigates to Serving through the command palette
    When the user opens the dashboard with demo data
    And the user opens the command palette
    Then dashboard destinations are displayed
    When the user chooses Serving
    Then Serving actions are displayed
    When the user quits the dashboard
    Then the dashboard exits successfully

  @id:dash-managed-service-visible @requires-os:linux
  Scenario: 7 - A managed model is visible in the dashboard
    Given a running managed model is available locally
    When the user opens the dashboard
    And the user opens the Observe view
    Then the managed model is displayed
    When the user quits the dashboard
    Then the dashboard exits successfully


  @id:eai-7960-gen-tps-held-after-scrape-failure @requires-os:linux
  Scenario: 8 - Gen throughput stays visible for the validity window after a scrape failure
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

  @id:eai-7960-gen-tps-expiry-boundary @requires-os:linux
  Scenario: 9 - Gen throughput expires after the validity window following sustained failure
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

  @id:launcher-shows-live-serving-instance @requires-os:linux
  Scenario: 10 - The launcher front door shows a live serving model rather than idle
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
