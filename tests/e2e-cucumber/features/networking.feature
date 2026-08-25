Feature: Native HTTP networking

  # rocm-cli performs downloads and GETs over the native `ureq` stack (native
  # certificate store), replacing the generated PowerShell script that used to
  # run under `powershell.exe -ExecutionPolicy Bypass` on Windows. These
  # scenarios drive the real `rocm` binary end-to-end against the mock server so a
  # GET/round-trip reaches a local endpoint over the native stack. They run on the
  # mock (no GPU) on every platform the suite runs on, so a regression back to a
  # shell-out networking backend surfaces here rather than only in the field.

  # `rocm services list` extracts host:port from the service record and issues a
  # native HTTP GET to `/v1/models` as its readiness probe (served by the mock).
  # Listing the model and its endpoint therefore exercises that native GET.
  @id:networking-native-http-endpoint-reachable
  Scenario: 1 - The CLI reaches a served endpoint over the native HTTP stack
    Given a model is being served
    And the model is registered with the CLI
    When the user lists running services
    Then the served model is listed
    And the served model endpoint is listed

  # A full chat round-trip: the real `rocm chat` command drives the local provider
  # to GET `/v1/models` and POST `/v1/chat/completions` over the native stack, then
  # prints the reply — proving the native HTTP client works end-to-end via the CLI.
  @id:networking-native-http-chat-round-trip
  Scenario: 2 - A chat round-trip over a local endpoint uses the native HTTP stack
    Given a model is being served
    And the model is registered with the CLI
    When the user sends a one-shot chat prompt through the CLI
    Then the CLI prints the assistant's reply

  # The EAI-7409 public-bind contract. Binding the server to a non-loopback
  # interface exposes it to the network, so the CLI refuses unless the user
  # explicitly opts in with `--allow-public-bind`. This check is pre-flight — the
  # first thing `serve` does, before any engine or model work — so it needs no GPU
  # and runs on the mock lane every PR.
  @id:networking-public-bind-requires-opt-in
  Scenario: 3 - Binding to a public interface without opt-in is refused up front
    When the user serves a model bound to a public interface without allowing public binding
    Then serving is refused before any engine starts
    And the user is told to allow public binding first

  # The other half of the EAI-7409 contract, and the security property the ticket
  # calls out first: opting in to a public bind must not produce an open endpoint.
  # The CLI issues an API key, shows it exactly once, and hands the serving engine
  # a key file to enforce — so the endpoint must reject callers without the key and
  # with the wrong key, while accepting the key it just issued.
  #
  # GPU lane: the ENGINE enforces the key (the CLI only issues and hands it over),
  # so proving enforcement needs a real engine serving a real model. `serve` has no
  # plan-only mode that would let the mock lane observe it. The pure key-resolution
  # rules (loopback stays credential-free even when a key is passed) are already
  # unit-tested in the CLI, so this pins only what a unit test cannot: that a live
  # public endpoint actually authenticates.
  @id:networking-public-bind-endpoint-enforces-key @requires-gpu
  Scenario: 4 - A public endpoint issues a key once and then enforces it
    Given a managed runtime is active
    When the user serves a model on a public interface with public binding allowed
    Then the CLI shows the endpoint key once and how to send it
    And a request without the key is refused as unauthorized
    And a request with the wrong key is refused as unauthorized
    And a request carrying the issued key is accepted
