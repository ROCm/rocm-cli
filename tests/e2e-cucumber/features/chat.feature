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

  # The reported defect was not a bad model, it was a blind one: the dashboard
  # assistant was never told which machine it was answering for, so it answered
  # platform questions from pretraining ("ROCm is not compatible with Windows").
  # Asserts what the CLI sends, not what the model replies — a 4B model's wording
  # is not deterministic, but what it is told is entirely ours.
  @id:chat-assistant-is-told-which-machine-it-is-on @requires-os:linux
  Scenario: chat-04 - The assistant is told what machine it is running on
    Given a running managed model is available locally
    When the user opens interactive chat
    Then the local endpoint is shown for confirmation
    When the user accepts the local endpoint
    And the user sends a message to the managed model
    Then the assistant is told which operating system this machine runs
    And the assistant is told which GPU this machine has
    When the user quits interactive chat
    Then interactive chat exits successfully

  @id:chat-endpoint-shown-in-services
  Scenario: chat-05 - A served model's endpoint is shown in the services list
    Given a model is being served
    And the model is registered with the CLI
    When the user lists running services
    Then the served model endpoint is listed

  # Runs on every lane: on a GPU host `a model is served in the background` does a
  # real `rocm serve`, on the no-GPU mock lane it's backed by MockServer. The
  # assertion (a tools-bearing request is accepted) is engine-agnostic, so no GPU
  # is required — dropping @requires-gpu gives this per-PR mock-lane coverage.
  @id:chat-tool-definitions-accepted
  Scenario: chat-06 - Chat requests that include tool definitions are accepted
    Given a managed runtime is active
    And a model is served in the background
    When a chat request with tool definitions is sent
    Then the chat response is successful

  # Runs on every lane (see chat-06): real serve on a GPU host, MockServer on
  # the no-GPU mock lane. Asserts only that a served model returns a non-empty
  # reply, which is engine-agnostic — real generation is covered by the
  # @requires-gpu serve-*-inference scenarios.
  @id:chat-end-to-end-local-model
  Scenario: chat-07 - End-to-end chat through a locally served model
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
  Scenario: chat-08 - The chat CLI answers a one-shot prompt against a local server
    Given a model is being served
    And the model is registered with the CLI
    When the user sends a one-shot chat prompt through the CLI
    Then the CLI prints the assistant's reply

  # `rocm chat --help` documents `echo "…" | rocm chat` — the prompt is read
  # from stdin when `--prompt` is omitted. This drives that path (piped stdin,
  # no `--prompt`) and asserts the assistant reply is produced, proving stdin is
  # consumed and routed through the same send path as `--prompt`. The piped text
  # is indented and trailing-spaced, so the second assertion also proves it
  # reaches the model unaltered apart from the newline the shell appends —
  # indentation is meaningful to a model and must not be trimmed away.
  @id:chat-cli-stdin-prompt
  Scenario: chat-09 - The chat CLI reads a one-shot prompt from stdin
    Given a model is being served
    And the model is registered with the CLI
    When the user pipes a one-shot chat prompt through the CLI
    Then the CLI prints the assistant's reply
    And the model receives the piped prompt with its whitespace intact

  # A request the CLI cannot plan on its own is handed to the configured local
  # planning assistant, whose answers are not the same twice. Asking twice would
  # let the plan the user reads and the plan that gets acted on come from two
  # different answers, so the assistant must be asked exactly once and the plan
  # shown must be its answer. The assistant here answers differently each time
  # it is asked, which is what makes a second ask visible. `--yes` never runs a
  # plan the assistant produced, so what this scenario can show about "acted on"
  # is that the request is refused with nothing executed; that the plan shown is
  # also the action taken is covered by the unit tests
  # `freeform_plan_dedup_prevents_provider_assisted_divergence` and
  # `freeform_execution_reuses_the_rendered_plan_without_rebuilding`.
  @id:chat-freeform-yes-plans-once
  Scenario: chat-10 - A vague request approved with --yes consults the planning assistant once
    Given a local planning assistant that suggests a different plan each time it is asked
    When the user approves a vague natural-language request with --yes
    Then the planning assistant was asked exactly once
    And the plan shown is the one the assistant gave
    And the request is refused rather than run without review

  # The same guarantee through the other path that plans a request: the chat
  # assistant asking for it to be planned. The chat only reports that the plan
  # was made, so the count of requests is what shows it was made once.
  @id:chat-assistant-plan-tool-plans-once
  Scenario: chat-11 - Asking the chat assistant to plan a vague request consults the planning assistant once
    Given a local assistant that hands the request to the planner, which suggests a different plan each time it is asked
    When the user asks the assistant in one-shot chat for help with a vague request
    Then the planning assistant was asked exactly once
    And the chat assistant was asked once before the plan and once after it
    And the chat reports that the request was planned
