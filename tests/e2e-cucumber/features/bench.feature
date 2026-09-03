Feature: Benchmarking a served endpoint

  # `rocm bench load` had no E2E coverage, which is how it reached users
  # measuring nothing: it POSTed to an unversioned chat path while its own model
  # probe used the versioned one, so whichever endpoint form the user supplied,
  # one of the two 404'd — and every request failure was swallowed, so the run
  # still exited 0 with a row of blanks.
  #
  # bench-01 to bench-04 run on every lane (MockServer-backed, no GPU needed)
  # and are what pin the request path, the failure reporting, and the CSV
  # columns. bench-05 is the hardware proof against a really served model.

  # The mock answers chat on BOTH the versioned and unversioned routes, so
  # "the benchmark succeeded" alone would pass even with the bug present. This
  # scenario therefore asserts which route the requests actually landed on.
  @id:bench-load-reports-throughput
  Scenario: bench-01 - Benchmarking a running server reports measured throughput
    Given a model is being served
    When the user benchmarks the served endpoint
    Then the benchmark reports measured throughput
    And the benchmark requests reached the versioned chat route

  @id:bench-load-accepts-plain-address
  Scenario: bench-02 - A plain host address is accepted and still reaches the server
    Given a model is being served
    When the user benchmarks the server using its plain host address
    Then the benchmark reports measured throughput
    And the benchmark requests reached the versioned chat route

  @id:bench-load-surfaces-failures
  Scenario: bench-03 - A benchmark whose every request is rejected fails loudly
    Given an endpoint that rejects every request
    When the user benchmarks the served endpoint
    Then the benchmark reports that the requests failed
    And the benchmark does not report a successful run

  # The value of the run is the CSV it writes, and two of its columns had no
  # coverage: `engine` (which the run now labels from the `/metrics` scrape) and
  # `tpot_ms` (the windowed per-output-token latency). A unit test on the helper
  # doesn't prove the columns reach the file, so this asserts the emitted row
  # against a metrics-backed mock whose histogram counters advance each scrape.
  @id:bench-load-records-engine-and-tpot
  Scenario: bench-04 - The recorded CSV row carries the engine and per-output-token latency
    Given a model is being served with a metrics endpoint
    When the user benchmarks the served endpoint recording results to a file
    Then the recorded benchmark row is labelled the vLLM engine with a per-output-token latency

  # Hardware proof: a real `rocm serve` on this host (vLLM on Instinct, lemonade
  # on Strix Halo) benchmarked through the endpoint the CLI itself reports.
  #
  # The runtime precondition is load-bearing on Instinct, where serving goes
  # through vLLM under a `gpu_required` device policy and is refused outright
  # without an active ROCm runtime. Lemonade hosts do not need it, so omitting it
  # fails on Instinct alone — mirror the sibling GPU serve scenarios and keep it.
  @id:bench-load-real-serve @requires-gpu
  Scenario: bench-05 - Benchmarking a really served model reports throughput
    Given a managed runtime is active
    And a model is being served on GPU
    When the user benchmarks the served endpoint
    Then the benchmark reports measured throughput
