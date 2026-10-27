Feature: Stopping a managed service

  # `serve_summary.rs`'s `stop` row and the approval gate in `main.rs` both print
  # `rocm services stop <id> --yes` as the one canonical, copy-pasteable command
  # to stop a managed service. Unit tests on both sides prove the two agree on
  # that exact string; neither proves that running it changes the service's
  # recorded state. `service_record_cleanup.feature`'s scenarios run this same
  # command too, but only as best-effort teardown or as a refusal premise —
  # never asserting that a *running* record is left stopped. This file closes
  # that gap: the planted record's `supervisor_pid: 0` means there is no live
  # process for the stop to reap, so what this proves is that the on-disk
  # record is actually persisted `stopped`, not that a real server process
  # was signalled — that part is exercised only on a GPU lane, by a real
  # `rocm serve`.
  #
  # A new feature file rather than a `service-cleanup-NN` addition: that file's
  # scenarios are about `remove`/`prune` deleting the files a record owns; this
  # one is about `stop` changing a running service's own state, a different
  # behaviour area with nothing to delete.
  #
  # Plants the record rather than failing a real serve: no GPU, no network, so
  # this runs on the mock lane every PR.

  @id:service-stop-yes-stops-a-running-service
  Scenario: service-stop-01 - `rocm services stop <id> --yes` stops a running managed service
    Given a managed service that is running
    When the user stops it with --yes
    Then the CLI reports the service as stopped
    And the service record on disk is marked stopped

  # The daemon's half. `rocmd sandbox-tool stop_server` is the stop an
  # automation or an agent reaches, through the daemon's restricted tool surface
  # rather than through `rocm services stop`. It reports a stop only once every
  # recorded process is confirmed gone, and the report is checked here together
  # with the state it claims: the process really exited, and the record and
  # endpoint key say so.
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
  Scenario: service-stop-02 - Stopping a running managed server confirms the stop and cleans up after it
    Given a managed server whose recorded process is running
    When the daemon's stop_server tool stops that server
    Then the tool reports the server stopped, naming the process it stopped
    And the recorded process is no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone
