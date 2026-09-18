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

  # The age rule only helps if the way past it is discoverable. A user who ran
  # `prune`, saw nothing happen and read the summary has to be able to act on
  # what it says, so this drives the exact flag that summary prints rather than
  # the `--older-than-hours 0` long form every scenario above uses.
  @id:service-cleanup-any-age-takes-what-the-default-kept
  Scenario: service-cleanup-04 - The default keeps a fresh record and --any-age takes it
    Given a local server record that is no longer running
    When the user prunes with the default age rule
    Then the CLI says it kept the record for being recent and names --any-age
    And every file belonging to that record is still there
    When the user prunes every record whatever its age
    Then every file belonging to that record is gone

  # The case the default invocation exists for. `prune` reads the record file's
  # age, and loading the records rewrites that file whenever it first notices
  # the server has died — so on the host this command is most needed on, a
  # plain `rocm services prune --yes` removed nothing and called every
  # long-dead record too recent to touch.
  @id:service-cleanup-prunes-a-server-that-died-long-ago
  Scenario: service-cleanup-05 - A record whose server died long ago is pruned by default
    Given a local server record whose server died long ago and was never listed since
    When the user prunes with the default age rule
    Then every file belonging to that record is gone
    And the CLI does not claim it kept anything for being recent

  # A bulk delete that hits an unremovable file must not pretend it succeeded,
  # and must not throw away its own account of what it DID delete on the way
  # out. So the failure is reported and the command exits non-zero, but only
  # after the plan is printed and the removal is recorded in the audit log —
  # the record of a destructive action is worth most exactly when one went
  # wrong.
  @id:service-cleanup-names-a-file-it-could-not-remove
  Scenario: service-cleanup-06 - A file prune cannot remove is named and fails the command
    Given a local server record that is no longer running
    And that record's engine state file cannot be deleted
    When the user prunes every record that is not running
    Then the CLI names the file it could not remove and exits non-zero
    And the record's other files are gone
    And the prune is still recorded in the audit log

  # `rocm serve` writes the 0600 endpoint key file *before* it writes the
  # record, so a server that is still coming up has a key with no record beside
  # it — exactly what the leftover sweep above looks for. Sweeping on that shape
  # alone let a concurrent `--any-age` prune delete a live server's key. So a
  # leftover written in the last minute is kept whatever age was asked for,
  # while the records `--any-age` exists to remove are untouched by that floor:
  # a record file on disk is what makes its companions not-leftovers in the
  # first place. This scenario asserts both halves in one run.
  #
  # Planted as on-disk state rather than raced against a real launch: that file
  # is all the sweep decides on, and racing it would only make the scenario
  # flaky.
  @id:service-cleanup-keeps-a-key-a-launch-is-still-writing
  Scenario: service-cleanup-07 - A key file written moments ago with no record survives --any-age
    Given a local server record that is no longer running
    And an endpoint key file written moments ago whose record does not exist yet
    When the user prunes every record whatever its age
    Then every file belonging to that record is gone
    And the endpoint key file of the starting server is still there
