Feature: Stopping a managed service

  # `serve_summary.rs`'s `stop` row and the approval gate in `main.rs` both print
  # `rocm services stop <id> --yes` as the one canonical, copy-pasteable command
  # to stop a managed service. Unit tests on both sides prove the two agree on
  # that exact string; neither proves that running it actually stops anything.
  # `service_record_cleanup.feature`'s scenarios run this same command too, but
  # only as best-effort teardown or as a refusal premise — never asserting that
  # a *running* record is left stopped. This file closes that gap.
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
