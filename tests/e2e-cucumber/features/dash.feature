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
    When the user quits the dashboard
    Then the dashboard exits successfully

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
