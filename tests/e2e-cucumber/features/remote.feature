Feature: Working with GPU machines over a private network

  # These cover the parts of `rocm remote` that need neither a real tailnet nor a
  # second machine: discovery, refusals, and the local session list. Serving,
  # publishing and teardown need a reachable remote and are covered end to end by
  # tests/remote-ssh/run-e2e.sh, which drives this same binary against a
  # containerised stand-in.

  @id:remote-targets-lists-machines
  Scenario: 1 - The user sees which machines could host a model
    Given a private network with a GPU machine and a phone
    When the user asks which remote targets exist
    Then both machines are listed
    And the listing says it is not a readiness check

  @id:remote-targets-tag-filter
  Scenario: 2 - The list can be narrowed to machines marked as GPU nodes
    Given a private network with a GPU machine and a phone
    When the user asks for remote targets tagged as GPU machines
    Then only the GPU machine is listed

  @id:remote-targets-offline-shown
  Scenario: 3 - A machine that is offline is still listed, marked offline
    Given a private network whose GPU machine is offline
    When the user asks which remote targets exist
    Then the GPU machine is listed as offline

  @id:remote-targets-not-connected
  Scenario: 4 - Discovery explains itself when the private network is not connected
    Given the private network client is installed but not connected
    When the user asks which remote targets exist
    Then the user is told it is not connected and how to connect
    And the command still succeeds

  @id:remote-serve-unknown-machine
  Scenario: 5 - Serving refuses a machine that is not on the private network
    Given a private network with a GPU machine and a phone
    When the user asks to serve a model on a machine that is not there
    Then the user is told it is not on the network
    And they are pointed at the list of machines that are

  @id:remote-serve-offline-machine
  Scenario: 6 - Serving refuses an offline machine instead of waiting for it
    Given a private network whose GPU machine is offline
    When the user asks to serve a model on the GPU machine
    Then the user is told the machine is offline

  @id:remote-status-no-sessions
  Scenario: 7 - The user is told when they have no remote sessions
    When the user asks about their remote sessions
    Then the user is told there are none and how to start one

  @id:remote-stop-unknown-session
  Scenario: 8 - Stopping a session that does not exist is refused
    When the user asks to stop a remote session that does not exist
    Then the user is told no such session is recorded

  @id:services-list-json
  Scenario: 9 - Local servers can be listed in machine-readable form
    When the user lists local servers as JSON
    Then the output is valid JSON

  # The successful paths need a second machine. `rocm remote` drives a real SSH
  # connection, so no amount of local stubbing produces one — these stand a
  # container up instead. The GPU and the private network are still stand-ins;
  # what is real is the connection, the commands the CLI builds, and the records
  # it keeps.
  #
  # Opt-in via E2E_INCLUDE_DOCKER=1, set on the GitHub-hosted `E2E tests` lane.
  # A working daemon is not enough on its own: the self-hosted GPU runners have
  # one but cannot reach the package mirror the fixture image builds from, so
  # they skip these rather than failing on an image they could never build.

  @id:remote-serve-publishes-endpoint @requires-docker
  Scenario: 10 - Serving a model on another machine gives back a usable endpoint
    Given a reachable GPU machine on the private network
    When the user serves a model on that machine
    Then the user is given an endpoint and a credential
    And the user is told the endpoint is reachable by the whole network
    And the machine is publishing that endpoint

  @id:remote-status-reports-both-halves @requires-docker
  Scenario: 11 - Status reports the model and the endpoint separately
    Given a model serving on a reachable GPU machine
    When the user asks about their remote sessions
    Then the model and the endpoint are both reported healthy

  @id:remote-attach-restores-endpoint @requires-docker
  Scenario: 12 - An endpoint withdrawn behind the user's back is repairable
    Given a model serving on a reachable GPU machine
    When the endpoint is withdrawn on the machine itself
    And the user asks about their remote sessions
    Then the model is still healthy but the endpoint is reported gone
    When the user re-publishes the endpoint
    Then the endpoint is restored without restarting the model

  @id:remote-stop-clears-both @requires-docker
  Scenario: 13 - Stopping a session leaves nothing running and nothing exposed
    Given a model serving on a reachable GPU machine
    When the user stops the session
    Then the endpoint and the model are both reported stopped
    And the machine is publishing nothing
    And the session is no longer listed

  @id:remote-doctor-reports-remote-health @requires-docker
  Scenario: 14 - Checking a remote machine reports that machine's health
    Given a reachable GPU machine on the private network
    When the user checks that machine's health
    Then the report names that machine
