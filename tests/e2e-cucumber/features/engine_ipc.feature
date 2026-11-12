Feature: How the daemon reads an engine's answer

  # `rocmd` asks an engine to do things (here: stop a server) by running
  # `rocm __engine-stdio <engine>` and reading what the child prints. The
  # standalone `rocmd` binary has no such verb of its own and runs the `rocm`
  # beside it, which is what makes these scenarios stageable from outside: each
  # one copies the real `rocmd` into a scratch directory and puts a stand-in
  # `rocm` next to it, so the daemon's own tool reaches a child whose answer the
  # scenario controls. The stand-in records the argv and the request it was
  # given and prints what the scenario says.
  #
  # service-stop-02 covers the engine that answers cleanly; the two below are the
  # other ways an engine child ends. In each, the engine's message in the tool's
  # report is checked together with what the stop did to the server: its process,
  # its record and its endpoint key. The server is a `vllm` one, for which the
  # engine's `Stop` is advisory (the recorded process decides whether the stop
  # is confirmed), so the stop going through is part of what is shown, not a
  # side effect.
  #
  # Linux-only, for the same reason as service-stop-02: the recorded process
  # carries the start-time token `/proc` gives it, and the stand-in is a shell
  # script. No GPU and no real engine are involved, so these run on the mock lane.

  @id:engine-ipc-reads-an-answer-printed-before-a-failing-exit
  @requires-os:linux
  Scenario: engine-ipc-01 - An engine that answers the stop and then exits non-zero is read as having answered it
    Given a managed server whose recorded process is running
    And the engine answers the stop request and then exits with a failing status
    When the standalone daemon's stop_server tool stops that server
    Then the engine received one stop request for that server over stdio
    And the tool reports the engine's answer to the stop, not the status it exited with
    And the recorded process is no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone

  @id:engine-ipc-reports-an-unreadable-answer-with-the-engines-stderr
  @requires-os:linux
  Scenario: engine-ipc-02 - An engine that prints something unreadable and fails reports its stderr, and the stop still completes
    Given a managed server whose recorded process is running
    And the engine prints something that is not an answer, complains on stderr and exits with a failing status
    When the standalone daemon's stop_server tool stops that server
    Then the engine received one stop request for that server over stdio
    And the tool reports the stop request failed, with the engine's exit status and stderr
    And the tool still reports the server stopped
    And the recorded process is no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone
