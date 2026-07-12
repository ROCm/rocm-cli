Feature: Chat and endpoint detection

  @id:chat-served-model-discoverable
  Scenario: 1 - A served model is discoverable through the services list
    Given a model is being served
    And the model is registered with the CLI
    When the user checks for running services
    Then the served model is listed

  @id:chat-privacy-notice-accurate
  Scenario: 2 - The privacy notice is accurate for local endpoints
    Given a model is being served locally
    When the user is offered the detected endpoint
    Then the notice does not claim that requests leave the machine

  @id:chat-endpoint-shown-in-services
  Scenario: 3 - A served model's endpoint is shown in the services list
    Given a model is being served
    And the model is registered with the CLI
    When the user lists running services
    Then the served model endpoint is listed

  @id:chat-tool-definitions-accepted @requires-gpu @requires-engine:vllm
  Scenario: 4 - Chat requests that include tool definitions are accepted
    Given a managed runtime is active
    And a model is served in the background
    When a chat request with tool definitions is sent
    Then the chat response is successful

  @id:chat-end-to-end-local-model @requires-gpu @requires-engine:vllm
  Scenario: 5 - End-to-end chat through a locally served model
    Given a managed runtime is active
    And a model is served in the background
    And the served model has been detected
    When the user sends a chat message
    Then the chat response is successful
    And the response contains a model-generated reply
