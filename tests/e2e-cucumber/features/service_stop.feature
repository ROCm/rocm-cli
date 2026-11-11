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
  # it and goes on to terminate it. No GPU and no running engine are involved: the
  # daemon does ask the engine to stop (a real `vllm` engine, reached through the
  # `rocm` beside the standalone `rocmd`), and since the planted record names no
  # engine process the engine answers that it stopped nothing, which the scenario
  # checks along with the report.
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
    And the tool asked the engine to stop and the engine answered
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

  # `rocm services stop`'s confirmed half with a real process. service-stop-01
  # plants PID 0, so it never signals anything and cannot show what the stop
  # leaves in the record; service-stop-02 shows it for the daemon's tool. This
  # runs the CLI's own stop against a real recorded process and checks the
  # persisted state it claims: the process exited, and the record names no
  # process any more, so a second stop of that record would signal nothing (a
  # PID recorded without a start-time token cannot be refuted once the OS
  # recycles it). The unit test
  # `a_confirmed_stop_clears_the_recorded_processes_so_a_second_stop_signals_nothing`
  # covers the second stop itself.
  #
  # Linux-only for the same reason as service-stop-02: the recorded process
  # carries the start-time token `/proc` gives it.

  @id:service-stop-cli-confirms-a-stop-and-clears-the-record
  @requires-os:linux
  Scenario: service-stop-06 - `rocm services stop` of a running managed server confirms the stop and leaves no process in its record
    Given a managed server whose recorded process is running
    When the user stops it with --yes
    Then the CLI reports the service as stopped
    And the recorded process is no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone

  # The Windows half of the stop. A managed engine runs one level below the
  # recorded process (the daemon launches it through an `__engine-serve-http`
  # process), and `terminate_verified` cannot walk a process tree on Windows, so
  # the stop first runs `taskkill /PID <pid> /T /F` on the recorded process.
  # service-stop-01 plants PID 0 and never reaches that step, and the scenarios
  # above need Linux's start-time token, so none of them kills a real tree.
  #
  # This records a real parent that has started a real child, stops it, and
  # checks that BOTH are gone together with the report and the record. Without
  # the tree-kill step the stop ends only the recorded process, the child keeps
  # running, and the scenario fails on it. No GPU, no engine, no network. The
  # record carries no start-time token, like a Windows record written before the
  # token was recorded: the case where the PID cannot be refuted, which the tree
  # kill and the cleared record exist for. service-stop-08 below is the same stop
  # with a token that does not match.

  @id:service-stop-takes-down-the-whole-process-tree-on-windows
  @requires-os:windows
  Scenario: service-stop-07 - Stopping a managed server on Windows takes down its recorded process together with the process it started
    Given a managed server whose recorded process has started another process
    When the user stops it with --yes
    Then the CLI reports the service as stopped
    And the report counts one stopped process
    And the recorded process and the process it started are no longer running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone

  # A recorded PID is not proof that the recorded process still holds it: the
  # server can exit by itself and the OS hand its PID to an unrelated program.
  # The record pairs the PID with the process's start-time token (Linux's
  # `/proc` start time, Windows' creation time), and a stop whose token no longer
  # matches treats the recorded process as already gone: it reports the stop,
  # clears the record and leaves the process that now holds the PID running,
  # instead of signalling a stranger (on Windows, with `taskkill /T /F`, its
  # whole subtree). The stranger here is a real process, so "left alone" is
  # checked on the process itself. Runs on every OS: the planted token is one
  # no live process has.
  @id:service-stop-leaves-a-process-that-reused-the-recorded-pid-alone
  Scenario: service-stop-08 - Stopping a managed server whose recorded PID now belongs to another process leaves that process running
    Given a managed server whose recorded process id now belongs to a different process
    When the user stops it with --yes
    Then the CLI reports the service as stopped
    And the process now holding the recorded process id is still running
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone

  # The restart refusal, which shares this file's planted record. A stop drops a
  # server's endpoint key, and `rocm services restart` accepts a stopped id, so a
  # restart of a publicly bound server whose key is gone would bring it back up
  # without authentication; it is refused instead. The refusal names a `rocm
  # serve` command to launch the server again, and the command is checked
  # against the real parser with a model filled in for its `<model>`
  # placeholder (`--help` ends the parse after every other word has been
  # accepted, so nothing is started). The server is still running, and `rocm
  # serve` reuses a running server and discards the key it generated, so the
  # refusal names the stop that has to come first, and that is checked the same
  # way. That running both clears the condition is covered by the unit tests,
  # which follow them through the real stop, the spawn's reuse of a running
  # service and `serve`'s own policy and key file; here they are only shown to
  # be commands `rocm` accepts.
  #
  # The refusal comes before the stop, so the promise it makes is the one checked
  # last: the running server is left running, not marked stopped and with no
  # stop request recorded.
  #
  # Linux-only like the scenarios above: the recorded process carries the
  # start-time token `/proc` gives it.
  @id:service-stop-restart-refuses-a-public-server-without-its-key
  @requires-os:linux
  Scenario: service-stop-09 - Restarting a public server whose endpoint key is gone is refused and leaves it running
    Given a public managed server whose endpoint key is gone and whose recorded process is running
    When the user restarts it with --yes
    Then the CLI refuses the restart because it would reopen the public server without authentication
    And the refusal names a serve command that rocm accepts
    And the refusal names the stop that has to come first
    And the server is left running, with no stop recorded and no key file

  # The other way a server needs a key: it was launched with `--require-api-key`
  # on a loopback bind, because something outside this machine republishes its
  # port, and a stop dropped the key. `rocm services restart` refuses it for the
  # same reason as the public server above, but with its own wording, which says
  # nothing about a public host, and its own command: it carries the flag the
  # server asked for and no `--host`, because loopback is where it was. As above
  # the server is still running, so the refusal names the stop first; both
  # commands are checked against the real parser, and the promise that nothing
  # was stopped is checked on the process, the record and the missing key file.
  #
  # Linux-only like the scenario above, for the same reason.
  @id:service-stop-restart-refuses-a-loopback-server-that-required-a-key
  @requires-os:linux
  Scenario: service-stop-10 - Restarting a loopback server that required an API key, whose key is gone, is refused and leaves it running
    Given a loopback managed server that required an API key, whose key is gone and whose recorded process is running
    When the user restarts it with --yes
    Then the CLI refuses the restart because it would reopen the server that required an API key without authentication
    And the refusal names a serve command that asks for the key again and that rocm accepts
    And the refusal names the stop that has to come first
    And the server is left running, with no stop recorded and no key file
    And the server's record still says it required an API key

  # The common way to reach the restart refusal is a server that has already
  # been stopped: a stop drops the key, and `rocm services restart` accepts the
  # id of a stopped server. The refusal then names a stop before the launch, and
  # that stop has nothing to stop. It must still do no harm, so it is run here as
  # printed, on the stopped server: it exits 0, reports the server stopped, and
  # leaves a record that names no process and no pending stop, which is the state
  # the launch that follows it needs (nothing running for `rocm serve` to reuse,
  # and nothing for the daemon's recovery to bring back under the old id). That
  # the launch then goes through is covered by the unit test that follows the
  # advice through the real stop and the real spawn for a stopped record, since
  # a real `rocm serve` needs a GPU.
  @id:service-stop-the-stop-a-refusal-names-is-harmless-on-a-stopped-server
  Scenario: service-stop-11 - The stop a restart refusal names succeeds on a server that is already stopped and leaves it stopped
    Given a public managed server that is already stopped and whose endpoint key is gone
    When the user restarts it with --yes
    Then the CLI refuses the restart because it would reopen the public server without authentication
    And the refusal names the stop that has to come first
    When the user runs the stop the refusal named
    Then the CLI reports the service as stopped
    And the server's record reads stopped and names no process
    And the server's endpoint key file is gone
    And the server is not listed as running
