Feature: Native HTTP networking

  # EAI-7402: rocm-cli performs downloads and GETs over the native `ureq` stack
  # (native certificate store), replacing the generated PowerShell script that
  # used to run under `powershell.exe -ExecutionPolicy Bypass` on Windows. These
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
