Feature: Model serving

  @expected-failure @expected-failure-EAI-7219
  Scenario: 1 - Short model names are expanded to their full name
    When the user serves a model using its short name
    Then the output shows the full model name

  @expected-failure @expected-failure-EAI-7219
  Scenario: 2 - Short name expansion is consistent across engines
    When the user serves the same short name with different engines
    Then all engines expand to the same full model name

  Scenario: 3 - A running model server is discoverable by name
    Given a model is being served on the default port
    And the model is registered with the CLI
    When the user lists running services
    Then the service appears with the correct model name and connection details

  Scenario: 4 - Running services show the correct connection details
    Given a model is being served on a non-default port
    And the model is registered with the CLI
    When the user lists running services
    Then the connection details match the actual server port

  @gpu
  Scenario: 5 - A served model responds to inference requests
    Given a managed runtime is active
    And a model is being served on GPU
    When the user sends a chat completion request
    Then the response contains a model reply
    And the response identifies the correct model

  @gpu
  Scenario: 6 - Serving a model without specifying an engine produces a working endpoint
    Given a managed runtime is active
    When the user serves a model without specifying an engine
    Then an engine is selected automatically
    And the model is reachable
    And the model responds to inference requests
