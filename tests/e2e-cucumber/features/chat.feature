Feature: Chat and endpoint detection

  @id:chat-served-model-discoverable
  Scenario: chat-01 - A served model is discoverable through the services list
    Given a model is being served
    And the model is registered with the CLI
    When the user checks for running services
    Then the served model is listed

  # Driven through the interactive TUI (a pseudo-terminal): the privacy notice
  # lives on the chat consent gate, which only the real terminal path renders.
  # The model is registered so the CLI discovers it on its OS-assigned port and
  # offers it — the notice must precede any request. (Previously an
  # untestable-black-box gap.)
  @id:chat-privacy-notice-accurate @requires-os:linux
  Scenario: chat-02 - The privacy notice is shown before using a local endpoint
    Given a model is being served locally
    And the model is registered with the CLI
    When the user opens interactive chat
    Then the local endpoint is shown for confirmation
    And the privacy notice is shown before any message is sent
    When the user quits interactive chat
    Then interactive chat exits successfully

  @id:chat-managed-model-interactive @requires-os:linux
  Scenario: chat-03 - Interactive chat uses a running managed model
    Given a running managed model is available locally
    When the user opens interactive chat
    Then the local endpoint is shown for confirmation
    When the user accepts the local endpoint
    And the user sends a message to the managed model
    Then the managed model's response is displayed
    And the mock received the typed prompt
    When the user quits interactive chat
    Then interactive chat exits successfully

  @id:chat-endpoint-shown-in-services
  Scenario: chat-04 - A served model's endpoint is shown in the services list
    Given a model is being served
    And the model is registered with the CLI
    When the user lists running services
    Then the served model endpoint is listed

  # Runs on every lane: on a GPU host `a model is served in the background` does a
  # real `rocm serve`, on the no-GPU mock lane it's backed by MockServer. The
  # assertion (a tools-bearing request is accepted) is engine-agnostic, so no GPU
  # is required — dropping @requires-gpu gives this per-PR mock-lane coverage.
  @id:chat-tool-definitions-accepted
  Scenario: chat-05 - Chat requests that include tool definitions are accepted
    Given a managed runtime is active
    And a model is served in the background
    When a chat request with tool definitions is sent
    Then the chat response is successful

  # Runs on every lane (see chat-05): real serve on a GPU host, MockServer on
  # the no-GPU mock lane. Asserts only that a served model returns a non-empty
  # reply, which is engine-agnostic — real generation is covered by the
  # @requires-gpu serve-*-inference scenarios.
  @id:chat-end-to-end-local-model
  Scenario: chat-06 - End-to-end chat through a locally served model
    Given a managed runtime is active
    And a model is served in the background
    And the served model has been detected
    When the user sends a chat message
    Then the chat response is successful
    And the response contains a model-generated reply

  # Exercises the `rocm chat` CLI itself (one-shot `--prompt` via the local
  # provider), not just the model endpoint over HTTP — so the command surface
  # reports `rocm chat` as covered. Runs on mock (no GPU): the local provider
  # resolves the planted managed-service record and talks to the mock server.
  @id:chat-cli-oneshot-prompt
  Scenario: chat-07 - The chat CLI answers a one-shot prompt against a local server
    Given a model is being served
    And the model is registered with the CLI
    When the user sends a one-shot chat prompt through the CLI
    Then the CLI prints the assistant's reply

  # `rocm chat --help` documents `echo "…" | rocm chat` — the prompt is read
  # from stdin when `--prompt` is omitted. This drives that path (piped stdin,
  # no `--prompt`) and asserts the assistant reply is produced, proving stdin is
  # consumed and routed through the same send path as `--prompt`.
  @id:chat-cli-stdin-prompt
  Scenario: 8 - The chat CLI reads a one-shot prompt from stdin
    Given a model is being served
    And the model is registered with the CLI
    When the user pipes a one-shot chat prompt through the CLI
    Then the CLI prints the assistant's reply
