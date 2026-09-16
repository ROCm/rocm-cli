Feature: Local server record cleanup

  # `rocm services remove` / `rocm services prune` delete the records a managed
  # serve leaves behind. Every scenario plants the records rather than failing a
  # real serve, so none needs a GPU and all run on the mock lane every PR.
  #
  # A record owns FOUR files, and only two of them sit in the services folder:
  # the details file and the log live there, the engine state file lives under
  # `<data>/engines/<engine>/state/`, and a public serve also leaves a 0600
  # endpoint key file. Deleting the two obvious ones — the documented manual
  # workaround — is what orphans the rest, so these scenarios assert on all of
  # them rather than on the command's own summary line.
  #
  # A new feature file rather than appended scenarios: `feature_naming.rs`
  # requires per-file indexes to be sequential in declaration order, so an
  # appended scenario must claim its file's *next* index, and several open
  # branches have already claimed the next `serve-`, `dash-` and `storage-`
  # ones. A new key collides with nothing.

  @id:service-cleanup-removes-every-file-a-record-owns
  Scenario: service-cleanup-01 - Removing a stopped record deletes every file it owns
    Given a local server record that is no longer running
    When the user removes that local server record
    Then the CLI names the log file before deleting it
    And every file belonging to that record is gone
    And the shared launch lock is still there
    And the record no longer appears in the full list

  @id:service-cleanup-refuses-a-running-server
  Scenario: service-cleanup-02 - Removing a running local server is refused
    Given a local server record that is still running
    When the user tries to remove that local server record
    Then the CLI refuses and tells the user to stop the server first
    And every file belonging to that record is still there

  @id:service-cleanup-prunes-previews-then-sweeps-leftovers
  Scenario: service-cleanup-03 - Pruning previews first, then sweeps leftover engine state
    Given a local server record that is no longer running
    And an engine state file whose local server record was deleted by hand
    When the user previews a prune of every record that is not running
    Then the preview lists the record and the leftover file
    And every file belonging to that record is still there
    When the user prunes every record that is not running
    Then every file belonging to that record is gone
    And the leftover engine state file is gone
