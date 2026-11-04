Feature: Log inspection

  # `rocm logs` reads the CLI's own recorded command logs from the isolated data
  # dir and can filter them by a search term or a service. These behaviours were
  # verified correct during the walkthrough but had no scenario. They read planted
  # log files only — no GPU, no network — so they run on the mock lane every PR.
  # The search count is asserted by the number of MATCHING lines (deterministic),
  # not the recent-line total, which also counts other log sources.

  @id:logs-search-reports-match-count
  Scenario: logs-01 - Searching logs reports how many recent lines match
    Given recorded command logs containing several lines about a topic
    When the user searches the logs for that topic
    Then the CLI reports the matching recent lines

  @id:logs-search-absent-term-no-matches
  Scenario: logs-02 - Searching logs for an absent term reports no matches
    Given recorded command logs containing several lines about a topic
    When the user searches the logs for a term that appears nowhere
    Then the CLI reports no matching lines

  @id:logs-service-and-search-conflict
  Scenario: logs-03 - Asking for a service and a search term at once is refused
    Given recorded command logs containing several lines about a topic
    When the user asks for one service's logs and a search term together
    Then the CLI refuses and explains only one may be used

  # `--follow` keeps printing new output while a server starts (a model download
  # can take many minutes). It ends by itself once the server is no longer
  # starting or running, so the scenario plants a stopped record: the snapshot
  # is printed, the CLI says it is not following, and the command returns
  # instead of waiting. Streaming new output from a live service is covered by
  # `follow_service_logs_streams_new_output_until_the_service_stops` in
  # `apps/rocm/src/services_text.rs`.
  @id:logs-follow-stopped-service-returns
  Scenario: logs-04 - Following a stopped server's logs prints them and returns
    Given a stopped local server whose log shows a download at 43 percent
    When the user follows that server's logs
    Then the CLI prints the download line, says it is not following, and exits

  @id:logs-follow-requires-service
  Scenario: logs-05 - Following without naming a server is refused
    When the user asks to follow logs without naming a service
    Then the CLI refuses because --follow needs --service

  # The same follow behavior through the `rocm services logs` entry point, using
  # the `-f` short form (the `rocm logs` scenarios above use `--follow`).
  @id:services-logs-follow-stopped-service-returns
  Scenario: logs-06 - Following a stopped server's logs through the services command returns
    Given a stopped local server whose log shows a download at 43 percent
    When the user follows that server's logs with rocm services logs -f
    Then the CLI prints the download line, says it is not following, and exits
