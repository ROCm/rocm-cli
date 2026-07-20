Feature: Chat and endpoint detection

  @id:chat-served-model-discoverable
  Scenario: 1 - A served model is discoverable through the services list
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
  Scenario: 2 - The privacy notice is shown before using a local endpoint
    Given a model is being served locally
    And the model is registered with the CLI
    When the user opens interactive chat
    Then the local endpoint is shown for confirmation
    And the privacy notice is shown before any message is sent

  @id:chat-managed-model-interactive @requires-os:linux
  Scenario: 3 - Interactive chat uses a running managed model
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
  Scenario: 4 - A served model's endpoint is shown in the services list
    Given a model is being served
    And the model is registered with the CLI
    When the user lists running services
    Then the served model endpoint is listed

  @id:chat-tool-definitions-accepted @requires-gpu
  Scenario: 5 - Chat requests that include tool definitions are accepted
    Given a managed runtime is active
    And a model is served in the background
    When a chat request with tool definitions is sent
    Then the chat response is successful

  @id:chat-end-to-end-local-model @requires-gpu
  Scenario: 6 - End-to-end chat through a locally served model
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
  Scenario: 7 - The chat CLI answers a one-shot prompt against a local server
    Given a model is being served
    And the model is registered with the CLI
    When the user sends a one-shot chat prompt through the CLI
    Then the CLI prints the assistant's reply
