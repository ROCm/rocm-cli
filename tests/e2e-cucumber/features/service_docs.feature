Feature: API docs for running model servers

  `rocm services docs` serves browsable API docs, with a working "Try it out",
  for the running model servers, on this machine only, until the user stops
  it. `rocm serve` and `rocm services list` name that command for every
  running server.

  # Each scenario runs the docs command on a free port of its own, so parallel
  # scenarios never share docs. The command `rocm serve` prints is followed on
  # the GPU serve canaries serve-05 (vLLM), serve-07 (Lemonade) and the nightly
  # serve-26 (AIM), whose last step opens it against the real engine.

  @id:service-docs-link-opens-chat-routes
  Scenario: service-docs-01 - The docs command the services list names opens docs listing the server's chat routes
    Given a model is being served
    And the model is registered with the CLI
    When the user runs the API docs command the services list names for the running server
    Then the command prints a link to the server's API docs
    And the link opens API docs listing the chat routes with the server's model

  @id:service-docs-try-it-reaches-server
  Scenario: service-docs-02 - Trying a route from the API docs reaches the model server
    Given a model is being served
    And the model is registered with the CLI
    And the user is serving the API docs for the running server
    When the user tries the chat completions route from the server's API docs
    Then the model server answers the request

  # `rocm services stop` signals the processes the record names, so the record
  # is pointed at a disposable process first (`sleep`, hence Linux only).
  @id:service-docs-gone-after-stop @requires-os:linux
  Scenario: service-docs-03 - A stopped server's API docs say it is not running
    Given a running managed model is available locally
    And the managed model's record points at a disposable process
    And the user is serving the API docs for the running server
    When the user stops the managed model
    Then the server's API docs say it is not running
    And the services list still shows the stopped server, without an API docs command

  # A program holds the port the refused command is given: a command that
  # started serving before refusing would fail on that port instead, so the
  # refusal it prints shows it was refused before serving anything.
  @id:service-docs-refused-for-unknown-server
  Scenario: service-docs-04 - Asking for the docs of a server that does not exist is refused before anything is served
    Given a model is being served
    And the model is registered with the CLI
    And another program is listening on the port the API docs are asked to serve on
    When the user asks for the API docs of a server that does not exist
    Then the API docs command is refused, naming the services list
    And the refusal comes before the command tried to serve on its port

  @id:service-docs-refused-without-servers
  Scenario: service-docs-05 - With no model server running the docs command is refused before anything is served
    Given another program is listening on the port the API docs are asked to serve on
    When the user asks for the API docs with no model server running
    Then the API docs command is refused, saying no model server is running
    And the refusal comes before the command tried to serve on its port

  # Ctrl+C is sent as SIGINT, hence Linux only.
  @id:service-docs-ctrl-c-stops-docs-only @requires-os:linux
  Scenario: service-docs-06 - Ctrl+C stops the API docs and leaves the model server running
    Given a model is being served
    And the model is registered with the CLI
    And the user is serving the API docs for the running server
    When the user presses Ctrl+C in the API docs command
    Then the API docs command says the docs stopped
    And nothing serves API docs on its port any more
    And the model server still answers

  # Port 11430 is the same for every scenario, so this one runs alone.
  @id:service-docs-default-port-taken @serial
  Scenario: service-docs-07 - With the default docs port taken the command serves on a free port and says so
    Given a model is being served
    And the model is registered with the CLI
    And another program is listening on the default API docs port
    When the user runs the API docs command for the running server without choosing a port
    Then the command says the default port is taken and that it serves on a free port
    And the link it prints opens the server's API docs on that free port

  @id:service-docs-engine-docs-link
  Scenario: service-docs-08 - The engine docs the services list names are the engine's own docs page
    Given a model is being served
    And the model is registered with the CLI
    When the user opens the engine docs the services list names for the running server
    Then the model server's own API docs page opens

  # The refusal names the restart command `rocm services list --all` prints for
  # the server. Following the restart through needs a real engine to start,
  # which no lane without a GPU has.
  @id:service-docs-refused-for-stopped-server
  Scenario: service-docs-09 - Asking for the docs of a stopped server names how to start it again
    Given a model is being served
    And the model is registered with the CLI as a stopped server
    When the user asks for the API docs of the stopped server
    Then the API docs command is refused, naming the command that restarts the server
    And the services list the refusal names shows the server stopped, with that restart command

  # `rocm serve` of a model that is already running reuses the server and
  # prints its rows; that needs no GPU, so these gate every pull request. The
  # rows of a fresh launch are followed by the GPU serve canaries (see the
  # comment at the top). vLLM, hence Linux only.
  @id:service-docs-serve-reuse-plain @requires-os:linux
  Scenario: service-docs-10 - Serving a model that is already running names its API docs
    Given a live managed vLLM serve is already running
    When the user serves that same model again with plain output
    Then the output names the API docs command and the engine's own docs for the running server
    And the API docs command it names opens that server's API docs

  @id:service-docs-serve-reuse-attached @requires-os:linux
  Scenario: service-docs-11 - Serving a model that is already running, attached, names its API docs
    Given a live managed vLLM serve is already running
    When the user serves that same model again attached to its output
    Then the output names the API docs command and the engine's own docs for the running server
    And the API docs command it names opens that server's API docs
