Feature: API docs for running model servers

  Every running model server gets API docs with a working "Try it out",
  served on loopback by the background helper `rocm serve` starts. The CLI
  prints a server's docs link only once those docs answer.

  # @serial: each scenario starts a real background helper. Run alongside the
  # rest of the mock lane, these four pushed the dash throughput-window
  # scenarios (dash-08/09) past their timing window in full runs; run alone
  # they add a few seconds and no concurrent load.

  @id:service-docs-link-opens-chat-routes @serial
  Scenario: service-docs-01 - A running server's API docs are linked and list its chat routes
    Given a model is being served
    And the model is registered with the CLI
    And the background helper is running
    When the user checks for running services
    Then the running server has an API docs link
    And the link opens API docs listing the chat routes with the server's model

  @id:service-docs-try-it-reaches-server @serial
  Scenario: service-docs-02 - Trying a route from the API docs reaches the model server
    Given a model is being served
    And the model is registered with the CLI
    And the background helper is running
    When the user tries the chat completions route from the server's API docs
    Then the model server answers the request

  # `rocm services stop` signals the processes the record names, so the record
  # is pointed at a disposable process first (`sleep`, hence Linux only).
  @id:service-docs-gone-after-stop @requires-os:linux @serial
  Scenario: service-docs-03 - A stopped server's API docs say it is not running
    Given a running managed model is available locally
    And the managed model's record points at a disposable process
    And the background helper is running
    And the user has opened the server's API docs
    When the user stops the managed model
    Then the server's API docs say it is not running
    And the services list still shows the stopped server, without an API docs link

  # The CLI prints a link only once the docs answer for that server. The link
  # `rocm serve` prints is checked on the GPU serve canaries serve-05 (vLLM) and
  # serve-07 (Lemonade; an expected failure on Linux Lemonade lanes for an
  # unrelated reason, so effective on Windows Lemonade lanes only).
  @id:service-docs-no-link-without-helper @serial
  Scenario: service-docs-04 - Without the background helper no API docs link is printed
    Given a model is being served
    And the model is registered with the CLI
    And the background helper is not running
    When the user checks for running services
    Then the running server has no API docs link

  @id:service-docs-speech-to-text-upload @serial
  Scenario: service-docs-05 - A speech-to-text server's API docs offer an audio upload instead of chat
    Given a speech-to-text model is being served and registered with the CLI
    And the background helper is running
    When the user opens the server's API docs from the services list
    Then the API docs list the transcription route and no chat routes
    And a recording uploaded through the API docs is transcribed
