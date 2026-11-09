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
  @id:logs-services-follow-stopped-service-returns
  Scenario: logs-06 - Following a stopped server's logs through the services command returns
    Given a stopped local server whose log shows a download at 43 percent
    When the user follows that server's logs with rocm services logs -f
    Then the CLI prints the download line, says it is not following, and exits

  # A server that ended `failed` is not a success, whether `--follow` finds it
  # already failed (this scenario) or sees it fail while following (covered by
  # `follow_service_logs_fails_when_the_service_failed` in
  # `apps/rocm/src/services_text.rs`).
  @id:logs-follow-failed-service-exits-nonzero
  Scenario: logs-07 - Following a failed server's logs prints them and exits with an error
    Given a failed local server whose log shows a download at 43 percent
    When the user follows that server's logs
    Then the CLI prints the download line, says it is not following, and exits with an error

  # A log is read as text but a stray byte that is not valid text must not hide
  # the readable lines around it: the view decodes it lossily and the search
  # still finds every matching line. Before, one such byte made `rocm logs`
  # show a single "failed to read" placeholder instead of the whole file.
  @id:logs-invalid-byte-keeps-matching-lines
  Scenario: logs-08 - A log with a byte that is not valid text still yields its matching lines
    Given recorded command logs containing several lines about a topic and one byte that is not valid text
    When the user searches the logs for that topic
    Then the CLI reports the matching recent lines

  # The log views read at most the last 4 MiB of a log. A log whose newest line
  # is longer than that has content but no line the read can return whole, and
  # reporting it as "no output yet" / "empty" told the user the file held
  # nothing. Each pair below plants the two states that look alike (an empty file
  # and one huge line) and asserts the wording together with the file on disk.
  @id:logs-service-log-longer-than-the-read-bound-is-not-called-empty
  Scenario: logs-09 - A server log whose only line is longer than the read bound is not reported as having no output
    Given a local server record whose log holds one line longer than the log view reads
    When the user reads that server's log
    Then the CLI says the newest line is too long to show instead of saying there is no output

  @id:logs-service-log-empty-says-no-output-yet
  Scenario: logs-10 - An empty server log is reported as having no output yet
    Given a local server record whose log file is empty
    When the user reads that server's log
    Then the CLI says there is no output yet

  @id:logs-comfyui-log-longer-than-the-read-bound-is-not-called-empty
  Scenario: logs-11 - A ComfyUI log whose only line is longer than the read bound is not reported as empty
    Given a ComfyUI log holding one line longer than the log view reads
    When the user reads the ComfyUI logs
    Then the CLI says the ComfyUI log's newest line is too long to show instead of saying it is empty

  @id:logs-comfyui-log-empty-says-empty
  Scenario: logs-12 - An empty ComfyUI log is reported as empty
    Given a ComfyUI log file that is empty
    When the user reads the ComfyUI logs
    Then the CLI says the ComfyUI log is empty

  # The `service_logs` tool the assistant and the MCP clients call reads the same
  # log through the same bound, and used to answer "Read the last 80 line(s)" with
  # an empty tail for both an empty log and one huge line. `rocm mcp-call` runs
  # that tool in the CLI; the daemon's copy of it (`rocmd mcp-call`) is covered by
  # a `rocmd` unit test, because the harness drives only the `rocm` binary.
  @id:logs-service-logs-tool-longer-than-the-read-bound-is-not-called-empty
  Scenario: logs-13 - The service_logs tool does not report a log with one over-long line as empty
    Given a local server record whose log holds one line longer than the log view reads
    When a tool-using client reads that server's log through the service_logs tool
    Then the tool says the newest line is too long to show and flags it instead of returning an empty tail unexplained

  @id:logs-service-logs-tool-empty-says-no-output-yet
  Scenario: logs-14 - The service_logs tool reports an empty log as having no output yet
    Given a local server record whose log file is empty
    When a tool-using client reads that server's log through the service_logs tool
    Then the tool says the server has no log output yet and does not flag an over-long line

  # `rocm logs` lists the last lines of each command log it finds. A command log
  # whose only line is longer than the 4 MiB a log view reads has content but no
  # line the read can return whole; listing nothing made the view print "No logs
  # found yet." although the file is on disk. It is listed with a line that says
  # why nothing is shown.
  @id:logs-command-log-longer-than-the-read-bound-is-not-called-absent
  Scenario: logs-15 - A command log whose only line is longer than the read bound is not reported as no logs found
    Given a recorded command log whose only line is longer than the log view reads
    When the user lists the recorded logs
    Then the CLI lists that log as having output it could not show instead of saying no logs were found

  # The structured reply's `lines` is how many lines the tool returned, not how
  # many it was asked for: asking for 50 lines of a three-line log returns three,
  # and the number asked for is reported separately as `lines_requested`.
  @id:logs-service-logs-tool-reports-lines-returned
  Scenario: logs-16 - The service_logs tool reports the lines it returned, not the number it was asked for
    Given a local server record whose log holds three short lines
    When a tool-using client asks the service_logs tool for 50 lines of that server's log
    Then the tool reports the three lines it returned and the 50 lines it was asked for
