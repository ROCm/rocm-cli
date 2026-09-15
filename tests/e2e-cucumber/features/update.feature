Feature: Update report

  # `rocm update` (with no arguments) prints an update report: which managed
  # runtimes have updates, plus the status of each update feed (CLI, engines,
  # model recipes, runtimes). The walkthrough verified it correctly distinguishes
  # published feeds from not-configured ones, but nothing pinned it. Run with no
  # managed runtimes so the report needs no network — mock lane, every PR.

  @id:update-report-distinguishes-feed-status
  Scenario: update-01 - The update report distinguishes configured from not-configured feeds
    Given a machine with no managed runtimes
    When the user checks for updates
    Then the report shows there are no managed runtimes to update
    And it reports each update feed's status, marking unpublished feeds as not configured

  @id:update-json-reports-empty-runtimes
  Scenario: update-02 - The machine-readable update check reports an empty runtimes array
    Given a machine with no managed runtimes
    When the user checks for updates as machine-readable JSON
    Then the machine-readable check reports no runtimes to update

  @id:update-json-accepts-timeout-flag
  Scenario: update-03 - The machine-readable update check accepts a --timeout-secs flag
    Given a machine with no managed runtimes
    When the user checks for updates as machine-readable JSON with a 5 second timeout
    Then the machine-readable check reports no runtimes to update

  # `rocm update` is a query by default and only changes the machine when asked
  # to. Nothing here installs anything: this scenario covers what the command
  # accepts, which is pure argument handling and so needs no GPU, no runtime,
  # and no network.
  #
  # Expected to FAIL. Asking to see what an update would do, without asking for
  # it to be done, is refused as a misuse — even though checking is what this
  # command does when left alone. README.md's synopsis for `rocm update` brackets
  # `[--apply]` and `[--dry-run]` as independent options, so a user who wants a
  # preview before committing to anything is turned away from the one command
  # the documentation told them would give it.
  #
  # The row in expectations.toml records why this does not presume which side is
  # wrong: closing the gap by correcting the synopsis would satisfy it just as
  # well as accepting the flag.
  @id:update-preview-without-applying
  Scenario: update-04 - Previewing an update without asking to install it is accepted
    When the user asks to see what updating would do without asking for it to be done
    Then the request is accepted rather than refused as a misuse
    And the machine still manages no runtimes
