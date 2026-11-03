Feature: Naming a local server that does not exist

  # Every command that takes a service id reads that service's record through
  # one shared loader (`ManagedServiceRecord::read_manifest` in rocm-core). A
  # typo'd or long-removed id used to surface the record's file path plus the
  # raw OS error ("The system cannot find the file specified. (os error 2)").
  # Each scenario below drives one of those commands with an id that has no
  # record and checks, together, that it fails, that it says so in one plain
  # line pointing at `rocm services list --all`, that no OS error is chained
  # under it — and that the advice holds: the named listing command runs and
  # really does not list that id.
  #
  # No record is planted and nothing is served, so these need no GPU and no
  # network and run on the mock lane every PR.

  @id:service-unknown-stop
  Scenario: service-unknown-01 - Stopping an unknown local server says it does not exist
    When the user runs "services stop nope --yes" naming a local server that does not exist
    Then the CLI says there is no local server named "nope" and points at the full list
    And the full list really has no local server named "nope"

  @id:service-unknown-restart
  Scenario: service-unknown-02 - Restarting an unknown local server says it does not exist
    When the user runs "services restart nope --yes" naming a local server that does not exist
    Then the CLI says there is no local server named "nope" and points at the full list
    And the full list really has no local server named "nope"

  @id:service-unknown-logs
  Scenario: service-unknown-03 - Showing an unknown local server's logs says it does not exist
    When the user runs "services logs nope" naming a local server that does not exist
    Then the CLI says there is no local server named "nope" and points at the full list
    And the full list really has no local server named "nope"

  @id:service-unknown-remove
  Scenario: service-unknown-04 - Removing an unknown local server's record says it does not exist
    When the user runs "services remove nope --yes" naming a local server that does not exist
    Then the CLI says there is no local server named "nope" and points at the full list
    And the full list really has no local server named "nope"

  @id:service-unknown-logs-service
  Scenario: service-unknown-05 - `rocm logs --service` for an unknown local server says it does not exist
    When the user runs "logs --service nope" naming a local server that does not exist
    Then the CLI says there is no local server named "nope" and points at the full list
    And the full list really has no local server named "nope"
