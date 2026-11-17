Feature: How the daemon and an engine read each other over stdio

  # `rocmd` asks an engine to do things (here: stop a server) by running
  # `rocm __engine-stdio <engine>` and reading what the child prints. The
  # standalone `rocmd` binary has no such verb of its own and runs the `rocm`
  # beside it, which is what makes engine-ipc-01 to -07 stageable from outside:
  # each copies the real `rocmd` into a scratch directory and puts a stand-in
  # `rocm` next to it, so the daemon's own tool reaches a child whose answer the
  # scenario controls. The stand-in records the argv and the request it was
  # given and prints what the scenario says.
  #
  # service-stop-02 covers the engine that answers cleanly; engine-ipc-01 and -02
  # are the other ways an engine child ends; -03 and -04 are answers the daemon
  # refuses (written in an engine protocol it cannot use, or not alone on
  # stdout), and -05 to -07 answers it reads (in its own protocol, in a newer
  # one that only adds to it, and in none, as from an engine older than the
  # field). In each, the engine's message in the tool's
  # report is checked together with what the stop did to the server: its process,
  # its record and its endpoint key. The server is a `vllm` one, for which the
  # engine's `Stop` is advisory (the recorded process decides whether the stop
  # is confirmed), so the stop going through is part of what is shown, not a
  # side effect.
  #
  # engine-ipc-08 to -12 are the other direction: the real built-in `vllm`
  # engine, run as `rocm __engine-stdio vllm` the way the daemon runs it, reads a
  # request. -08 is a stop it cannot serve, for a server whose process the
  # engine itself recorded; -09 and -10 are requests it refuses for their
  # protocol before it decodes their method (one it does not have) or reads a
  # version that is not even text; -11 and -12 are a method it does not have in
  # its own protocol, and input that is not a request at all. Each is answered
  # on stdout, with a clean exit.
  #
  # engine-ipc-01 to -08 and -10 are Linux-only, for the same reason as
  # service-stop-02: the recorded process carries the start-time token `/proc`
  # gives it, and in -01 to -07 the stand-in is a shell script. -09, -11 and
  # -12 use neither, but have only been run on Linux, so they keep the tag until
  # they are run on Windows. No GPU and no real engine server are involved, so
  # these run on the mock lane.

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

  @id:engine-ipc-refuses-an-answer-in-a-protocol-it-cannot-use
  @requires-os:linux
  Scenario: engine-ipc-03 - An engine's answer in an engine protocol this release cannot use is refused, and the stop still completes
    Given a managed server whose recorded process is running
    And the engine answers the stop request in an engine protocol this release cannot use
    When the standalone daemon's stop_server tool stops that server
    Then the engine received one stop request for that server over stdio
    And the tool reports the engine's answer refused, naming the engine's protocol and this release's
    And the tool still reports the server stopped
    And the recorded process is no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone

  @id:engine-ipc-refuses-an-answer-with-other-output-on-stdout
  @requires-os:linux
  Scenario: engine-ipc-04 - An engine's answer with a log line before it on stdout is refused, and the stop still completes
    Given a managed server whose recorded process is running
    And the engine writes a log line on stdout before its answer to the stop request
    When the standalone daemon's stop_server tool stops that server
    Then the engine received one stop request for that server over stdio
    And the tool reports the engine's answer refused for the log line the engine wrote before it
    And the tool still reports the server stopped
    And the recorded process is no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone

  @id:engine-ipc-reads-an-answer-in-its-own-protocol
  @requires-os:linux
  Scenario: engine-ipc-05 - An engine that answers in this release's engine protocol is read as having answered
    Given a managed server whose recorded process is running
    And the engine answers the stop request in this release's engine protocol
    When the standalone daemon's stop_server tool stops that server
    Then the engine received one stop request for that server over stdio
    And the tool reports the engine's answer to the stop
    And the recorded process is no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone

  @id:engine-ipc-reads-an-answer-in-a-newer-compatible-protocol
  @requires-os:linux
  Scenario: engine-ipc-06 - An engine that answers in a newer engine protocol that only adds to this release's is read as having answered
    Given a managed server whose recorded process is running
    And the engine answers the stop request in a newer engine protocol that only adds to this release's
    When the standalone daemon's stop_server tool stops that server
    Then the engine received one stop request for that server over stdio
    And the tool reports the engine's answer to the stop
    And the recorded process is no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone

  @id:engine-ipc-reads-an-answer-that-names-no-protocol
  @requires-os:linux
  Scenario: engine-ipc-07 - An engine that names no engine protocol, as one from before the field, is read as having answered
    Given a managed server whose recorded process is running
    And the engine answers the stop request in no named engine protocol
    When the standalone daemon's stop_server tool stops that server
    Then the engine received one stop request for that server over stdio
    And the tool reports the engine's answer to the stop
    And the recorded process is no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone

  @id:engine-ipc-engine-refuses-a-request-in-a-protocol-it-cannot-serve
  @requires-os:linux
  Scenario: engine-ipc-08 - An engine asked to stop a server in an engine protocol it cannot serve refuses, and leaves the server running
    Given an engine server whose process the vllm engine recorded is running
    When the vllm engine is asked over stdio to stop that server, in an engine protocol it cannot serve
    Then the engine answers that it cannot serve that engine protocol, naming its own and the request's
    And the engine server's process is still running
    And the vllm engine's record of that server still names its process

  @id:engine-ipc-engine-refuses-an-unknown-method-in-a-protocol-it-cannot-serve
  @requires-os:linux
  Scenario: engine-ipc-09 - An engine sent a method it does not have, in an engine protocol it cannot serve, refuses it for the protocol
    When the vllm engine is sent over stdio a request for a method it does not have, in an engine protocol it cannot serve
    Then the engine answers that it cannot serve that engine protocol, naming its own and the request's

  @id:engine-ipc-engine-refuses-a-version-that-is-not-text
  @requires-os:linux
  Scenario: engine-ipc-10 - An engine asked to stop a server with a protocol version that is not text refuses, and leaves the server running
    Given an engine server whose process the vllm engine recorded is running
    When the vllm engine is sent over stdio a stop request for that server whose protocol version is not text
    Then the engine answers that it cannot read the request's protocol version
    And the engine server's process is still running
    And the vllm engine's record of that server still names its process

  @id:engine-ipc-engine-answers-an-unknown-method
  @requires-os:linux
  Scenario: engine-ipc-11 - An engine sent a method it does not have, in its own engine protocol, says it has no such method
    When the vllm engine is sent over stdio a request for a method it does not have, in its own engine protocol
    Then the engine answers that it has no such method

  @id:engine-ipc-engine-answers-input-that-is-not-a-request
  @requires-os:linux
  Scenario: engine-ipc-12 - An engine sent something that is not a request says so in an answer
    When the vllm engine is sent over stdio something that is not a request
    Then the engine answers that what it was sent is not a request
