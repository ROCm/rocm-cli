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

  # A log is read as text but a stray byte that is not valid text must not hide
  # the readable lines around it: the view decodes it lossily and the search
  # still finds every matching line. Before, one such byte made `rocm logs`
  # show a single "failed to read" placeholder instead of the whole file.
  @id:logs-invalid-byte-keeps-matching-lines
  Scenario: logs-04 - A log with a byte that is not valid text still yields its matching lines
    Given recorded command logs containing several lines about a topic and one byte that is not valid text
    When the user searches the logs for that topic
    Then the CLI reports the matching recent lines
