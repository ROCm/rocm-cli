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

  # The log views read at most the last 4 MiB of a log. A log whose newest line
  # is longer than that has content but no line the read can return whole, and
  # reporting it as "no output yet" / "empty" told the user the file held
  # nothing. Each pair below plants the two states that look alike (an empty file
  # and one huge line) and asserts the wording together with the file on disk.
  @id:logs-service-log-longer-than-the-read-bound-is-not-called-empty
  Scenario: logs-05 - A server log whose only line is longer than the read bound is not reported as having no output
    Given a local server record whose log holds one line longer than the log view reads
    When the user reads that server's log
    Then the CLI says the newest line is too long to show instead of saying there is no output

  @id:logs-service-log-empty-says-no-output-yet
  Scenario: logs-06 - An empty server log is reported as having no output yet
    Given a local server record whose log file is empty
    When the user reads that server's log
    Then the CLI says there is no output yet

  @id:logs-comfyui-log-longer-than-the-read-bound-is-not-called-empty
  Scenario: logs-07 - A ComfyUI log whose only line is longer than the read bound is not reported as empty
    Given a ComfyUI log holding one line longer than the log view reads
    When the user reads the ComfyUI logs
    Then the CLI says the ComfyUI log's newest line is too long to show instead of saying it is empty

  @id:logs-comfyui-log-empty-says-empty
  Scenario: logs-08 - An empty ComfyUI log is reported as empty
    Given a ComfyUI log file that is empty
    When the user reads the ComfyUI logs
    Then the CLI says the ComfyUI log is empty
