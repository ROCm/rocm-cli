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
  #
  # The `Local server stopped` header follows the stop's own verdict, so this
  # is the confirmed half; service-stop-03 and -04 below are the unconfirmed
  # one.

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
  # The daemon's unconfirmed half — a recorded process that survives the stop,
  # or whose identity cannot be read — needs a process that outlives SIGKILL or
  # an unreadable `/proc` entry. Neither can be staged from outside `rocmd`,
  # which has no test seam for it, so that half of `stop_server` is covered by
  # `rocmd`'s unit tests. `rocm services stop`'s unconfirmed half is staged
  # below, in service-stop-03 and -04.

  @id:service-stop-confirms-a-stop-and-cleans-up
  @requires-os:linux
  Scenario: service-stop-02 - Stopping a running managed server confirms the stop and cleans up after it
    Given a managed server whose recorded process is running
    When the daemon's stop_server tool stops that server
    Then the tool reports the server stopped, naming the process it stopped
    And the recorded process is no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone

  # `rocm services stop`'s unconfirmed half. Neither way a stop goes
  # unconfirmed can be produced from outside the binary — a recorded process
  # that outlives SIGKILL, or one whose start-time cannot be read back — so the
  # `e2e-test-hooks` build's `ROCM_E2E_STOP_OUTCOME` seam makes the verified
  # stop report that outcome for the recorded process without signalling it.
  # Everything else runs for real: the stop's record writes, the key handling,
  # the report and the exit status. The seam applies to the one stop the Given
  # sets it up for.
  #
  # Neither report names a command to run the stop again. A second stop cannot
  # be relied on to finish: when the recorded process is gone and only a
  # process under it survived, a second stop finds the recorded one gone and
  # confirms without reaching the survivor.
  #
  # What the seam cannot show is what `terminate_verified` itself does: it
  # replaces that call, so these scenarios prove the CLI's report, exit status
  # and kept record and key for each outcome, not whether a process was
  # signalled. That an `unverified` PID is never signalled rests on
  # `terminate_verified` returning before any signal when the identity is
  # indeterminate (`rocm-core`'s `proc_lifecycle.rs`); no unit test covers that
  # path yet, because an unreadable `/proc` entry cannot be staged there either.
  #
  # Linux-only for the same reason as service-stop-02: the recorded process
  # carries the start-time token `/proc` gives it.

  @id:service-stop-unconfirmed-fails-and-keeps-the-record
  @requires-os:linux
  Scenario: service-stop-03 - A stop that cannot confirm the exit fails, keeps the server's record, and names no retry
    Given a managed server whose recorded process is running
    And the next stop cannot confirm that process exited
    When the user stops it with --yes
    Then the CLI reports the stop as not confirmed for a process that did not exit
    And the server's record still names its process and the stop request
    And the report says the record and the endpoint key were kept, and the key file is still there

  @id:service-stop-unverified-fails-without-a-retry
  @requires-os:linux
  Scenario: service-stop-04 - A stop that cannot verify its process fails, keeps the server's record, and names a check, not a retry
    Given a managed server whose recorded process is running
    And the next stop cannot verify that process as the server's
    When the user stops it with --yes
    Then the CLI reports the stop as not confirmed for a process it could not verify
    And the server's record still names its process and the stop request
    And the report says the record and the endpoint key were kept, and the key file is still there

  # A server started without an API key has no key file, and an unconfirmed
  # stop of it keeps only the record; the report must not claim otherwise.
  @id:service-stop-unconfirmed-keyless-keeps-only-the-record
  @requires-os:linux
  Scenario: service-stop-05 - An unconfirmed stop of a server without an endpoint key reports keeping only its record
    Given a managed server without an endpoint key whose recorded process is running
    And the next stop cannot confirm that process exited
    When the user stops it with --yes
    Then the CLI reports the stop as not confirmed and names no stop command
    And the server's record still names its process and the stop request
    And the report says only the record was kept, and there is no key file
