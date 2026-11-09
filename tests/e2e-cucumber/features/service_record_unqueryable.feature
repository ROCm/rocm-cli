Feature: A local server record held by a process rocm cannot query

  # On Windows a live process the account may not open cannot be queried, so
  # `rocm` cannot tell it from a dead server whose PID another program took.
  # It treats that as possibly running: a stop reports it unverified, and
  # `rocm services remove` refuses the record. `remove --force --yes` is the way
  # out for exactly that case, and these scenarios drive the whole path through
  # the real binary rather than the pieces.
  #
  # An unqueryable process cannot be staged from outside the binary, so the
  # `e2e-test-hooks` build reads ROCM_E2E_UNQUERYABLE_PIDS and answers
  # "cannot be queried" for the PIDs it lists; production builds never read it.
  # The records are planted as plain JSON, so no scenario needs a GPU.
  #
  # A new feature file, as `service_record_cleanup.feature` explains for its
  # own: indexes must be sequential per file, and several open branches claim the
  # next ones in the files that exist.

  @id:service-unqueryable-remove-force-clears-an-unverified-record
  Scenario: service-unqueryable-01 - A record whose process cannot be queried is stopped as unverified, refused by remove, and cleared by remove --force
    Given a local server record whose recorded process cannot be queried
    When the user stops that local server while its process cannot be queried
    Then the CLI reports the stop as not confirmed because the process could not be queried
    And every file belonging to that record is still there
    When the user removes that local server record while its process cannot be queried
    Then the CLI refuses the removal and names the forced removal
    And every file belonging to that record is still there
    When the user force-removes that local server record while its process cannot be queried
    Then the CLI removes the record and says its process could not be queried
    And every file belonging to that record is gone

  # `--force` is for a process `rocm` cannot query, never for one it can see
  # running. The recorded process here is a real one the scenario started, and a
  # stop was requested on the record, so the only thing standing in the way is
  # that the process is running.
  @id:service-unqueryable-force-never-passes-a-running-process
  @requires-os:linux
  Scenario: service-unqueryable-02 - --force is refused for a record whose process is confirmed running
    Given a local server record whose recorded process is running and whose stop was requested
    When the user force-removes that local server record
    Then the CLI refuses the forced removal because the record does not qualify
    And every file belonging to that record is still there
