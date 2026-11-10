Feature: Working with GPU machines over a private network

  # These cover the parts of `rocm remote` that need neither a real tailnet nor a
  # second machine: discovery, refusals, and the local session list. Serving,
  # publishing and teardown need a reachable remote and are covered end to end by
  # tests/remote-ssh/run-e2e.sh, which drives this same binary against a
  # containerised stand-in.

  @id:remote-targets-lists-machines
  Scenario: remote-01 - The user sees which machines could host a model
    Given a private network with a GPU machine and a phone
    When the user asks which remote targets exist
    Then both machines are listed
    And the listing says it is not a readiness check

  @id:remote-targets-tag-filter
  Scenario: remote-02 - The list can be narrowed to machines marked as GPU nodes
    Given a private network with a GPU machine and a phone
    When the user asks for remote targets tagged as GPU machines
    Then only the GPU machine is listed

  @id:remote-targets-offline-shown
  Scenario: remote-03 - A machine that is offline is still listed, marked offline
    Given a private network whose GPU machine is offline
    When the user asks which remote targets exist
    Then the GPU machine is listed as offline

  @id:remote-targets-not-connected
  Scenario: remote-04 - Discovery explains itself when the private network is not connected
    Given the private network client is installed but not connected
    When the user asks which remote targets exist
    Then the user is told it is not connected and how to connect
    And the command still succeeds

  @id:remote-serve-unknown-machine
  Scenario: remote-05 - Serving refuses a machine that is not on the private network
    Given a private network with a GPU machine and a phone
    When the user asks to serve a model on a machine that is not there
    Then the user is told it is not on the network
    And they are pointed at the list of machines that are

  @id:remote-serve-offline-machine
  Scenario: remote-06 - Serving refuses an offline machine instead of waiting for it
    Given a private network whose GPU machine is offline
    When the user asks to serve a model on the GPU machine
    Then the user is told the machine is offline

  @id:remote-status-no-sessions
  Scenario: remote-07 - The user is told when they have no remote sessions
    When the user asks about their remote sessions
    Then the user is told there are none and how to start one

  @id:remote-stop-unknown-session
  Scenario: remote-08 - Stopping a session that does not exist is refused
    When the user asks to stop a remote session that does not exist
    Then the user is told no such session is recorded

  @id:remote-doctor-refuses-to-install @requires-docker
  Scenario: remote-09 - Checking a machine's health never installs anything on it
    Given a reachable GPU machine with no ROCm CLI on the private network
    When the user checks the health of a machine that has no ROCm CLI
    Then the user is told nothing was installed and how to install it
    And they are pointed at serving as the command that provisions

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
  Scenario: remote-10 - Serving a model on another machine gives back a usable endpoint
    Given a reachable GPU machine on the private network
    When the user serves a model on that machine
    Then the user is given an endpoint and a credential
    And the user is told the endpoint is reachable by the whole network
    And the machine is publishing that endpoint
    And the session is recorded where only the user can reach it

  @id:remote-status-reports-both-halves @requires-docker
  Scenario: remote-11 - Status reports the model and the endpoint separately
    Given a model serving on a reachable GPU machine
    When the user asks about their remote sessions
    Then the model and the endpoint are both reported healthy

  @id:remote-attach-restores-endpoint @requires-docker
  Scenario: remote-12 - An endpoint withdrawn behind the user's back is repairable
    Given a model serving on a reachable GPU machine
    When the endpoint is withdrawn on the machine itself
    And the user asks about their remote sessions
    Then the model is still healthy but the endpoint is reported gone
    When the user re-publishes the endpoint
    Then the endpoint is restored without restarting the model

  @id:remote-stop-clears-both @requires-docker
  Scenario: remote-13 - Stopping a session leaves nothing running and nothing exposed
    Given a model serving on a reachable GPU machine
    When the user stops the session
    Then the endpoint and the model are both reported stopped
    And the machine is publishing nothing
    And the session is no longer listed

  @id:remote-doctor-reports-remote-health @requires-docker
  Scenario: remote-14 - Checking a remote machine reports that machine's health
    Given a reachable GPU machine on the private network
    When the user checks that machine's health
    Then the report names that machine

  # `discover_started_service` (#604) refuses to publish an endpoint when the
  # remote reused a pre-existing service instead of starting a new one — a
  # running server cannot be given a key it did not start with. Reaching that
  # refusal needs the service to already be live on the machine with no local
  # session recorded for it (an operator's own `rocm serve`, or an earlier
  # unmanaged run); a `Given`/`When` pair that both call `rocm remote serve`
  # never gets there; the second call trips `serve_with_transport`'s local
  # `session::exists` guard first, on the session the first call recorded.
  # Planting the service straight on the container — the fake `rocm serve`'s
  # service_id is keyed on the port alone, so it lands under the same id the
  # one `remote serve` call then discovers — reproduces the real precondition
  # instead. This is the remote-side proof that the printed stop hint is not
  # just present but correct and directly runnable, the same proof remote-13
  # gives the session-stop path: the scenario runs the hint itself, over the
  # same container, and checks the service is actually gone.
  @id:remote-serve-refuses-reusing-already-serving-session @requires-docker
  Scenario: remote-15 - Serving again on an already-serving machine is refused with a working stop command
    Given the machine is already serving a model, started outside any session
    When the user serves a model on that machine
    Then the user is told it is already serving and given the exact command to stop it
    And running that command on the machine stops it

  # A session record names the command this tool runs on the machine. One that
  # has been edited to name something else must be refused before it reaches
  # the machine, not merely reported oddly after it ran there.
  @id:remote-status-refuses-altered-record @requires-docker
  Scenario: remote-16 - A session record altered to run something else is refused, not run
    Given a model serving on a reachable GPU machine
    When the session record is altered to run an extra command on that machine
    And the user asks about their remote sessions
    Then the user is told the session record was refused
    And the extra command never ran on the machine

  # The same refusal without a machine: a record already on disk that names a
  # remote command this tool never writes. Runs on every lane, because nothing
  # here reaches the network.
  @id:remote-status-counts-refused-record
  Scenario: remote-17 - A refused session record is counted, not reported as no sessions
    Given a session record on this machine that names a remote command this tool never writes
    When the user asks about their remote sessions
    Then the user is told the session record was refused

  @id:remote-stop-names-refused-record
  Scenario: remote-18 - Stopping a refused session names the record instead of claiming nothing is recorded
    Given a session record on this machine that names a remote command this tool never writes
    When the user asks to stop that session
    Then the user is told the record was refused and how to clear it
    And the session record is left in place

  @id:remote-status-refuses-misnamed-record
  Scenario: remote-19 - A session record filed under another session's name is refused
    Given a session record on this machine filed under a name other than its own session
    When the user asks about their remote sessions
    Then the user is told the session record is not named after its session

  # `serve` computes the same session id for the same machine and port, so a
  # refused record already filed under it blocks the serve. `stop` cannot reach
  # it, so the refusal must not send the user there.
  @id:remote-serve-refuses-over-refused-record
  Scenario: remote-20 - Serving over a refused session record names the record instead of offering stop
    Given a private network with a GPU machine and a phone
    And a session record for the GPU machine that names a remote command this tool never writes
    When the user asks to serve a model on the GPU machine
    Then the user is told the session record cannot be used and `stop` cannot reach it
    And the session record is left in place
