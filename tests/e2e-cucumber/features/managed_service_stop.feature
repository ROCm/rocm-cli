Feature: Stopping a managed server through the daemon's tools

  # `rocmd sandbox-tool stop_server` is the stop an automation or an agent
  # reaches, through the daemon's restricted tool surface rather than through
  # `rocm services stop`. It reports a stop only once every recorded process is
  # confirmed gone, and the report is checked here together with the state it
  # claims: the process really exited, and the record and endpoint key say so.
  #
  # The recorded process is a real one the scenario starts, recorded with the
  # kernel's own start-time token for it, so the stop's identity check confirms
  # it and goes on to terminate it. No GPU and no engine are involved.
  #
  # Linux-only: the token is read from `/proc`, which is where `rocmd` reads it
  # too; without one the identity check is not exercised at all.
  #
  # The unconfirmed half of the contract — a recorded process that survives the
  # stop, or whose identity cannot be read — needs a process that outlives
  # SIGKILL or an unreadable `/proc` entry. Neither can be staged from outside
  # the binary, so that half is covered by `rocmd`'s unit tests instead.

  @id:service-stop-confirms-a-stop-and-cleans-up
  @requires-os:linux
  Scenario: service-stop-01 - Stopping a running managed server confirms the stop and cleans up after it
    Given a managed server whose recorded process is running
    When the daemon's stop_server tool stops that server
    Then the tool reports the server stopped, naming the process it stopped
    And the recorded process is no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone
