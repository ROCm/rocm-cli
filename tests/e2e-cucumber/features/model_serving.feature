Feature: Model serving

  # Per-platform expectations (pass / xfail / skip) are resolved at runtime from
  # the host capability probe + expectations.toml, keyed by the @id tag — not by
  # @expected-failure tags. Engine requirements are declared via @requires-engine
  # so the harness can skip a scenario whose engine can't start on this host.

  @id:serve-short-name-expansion
  Scenario: 1 - Short model names are expanded to their full name
    When the user serves a model using its short name
    Then the output shows the full model name

  @id:serve-short-name-consistent-across-engines
  Scenario: 2 - Short name expansion is consistent across engines
    When the user serves the same short name with different engines
    Then all engines expand to the same full model name

  @id:serve-discoverable-by-name
  Scenario: 3 - A running model server is discoverable by name
    Given a model is being served on the default port
    And the model is registered with the CLI
    When the user lists running services
    Then the service appears with the correct model name and connection details

  @id:serve-connection-details
  Scenario: 4 - Running services show the correct connection details
    Given a model is being served on a non-default port
    And the model is registered with the CLI
    When the user lists running services
    Then the connection details match the actual server port

  # vLLM serve + inference (safetensors model). Engine coverage: vLLM. This is the
  # deliberate vLLM half of a per-engine pair with `serve-lemonade-inference`
  # below, so it stays pinned to vLLM (the slug names the engine).
  @id:serve-vllm-inference @requires-gpu @requires-engine:vllm
  Scenario: 5 - A served model responds to inference requests on vLLM
    Given a managed runtime is active
    And a model is being served on GPU
    When the user sends a chat completion request
    Then the response contains a model reply
    And the response identifies the correct model

  # Large-model coverage (dogfooding W9): serve a big vLLM model end-to-end at
  # least once. Qwen/Qwen3.6-27B (BF16, ~54 GiB) fits a single MI300X; pinned to
  # vLLM so in our fleet it runs only there (Strix Halo is lemonade / too little
  # VRAM). A cold load of ~54 GiB far exceeds the 600s default, so the scenario
  # declares a longer serve-readiness timeout. Confirmed working on app-dev MI300X.
  @id:serve-large-model-inference @requires-gpu @requires-engine:vllm @serve-timeout:2400 @nightly
  Scenario: 10 - A large vLLM model serves and responds to inference
    Given a managed runtime is active
    And a large model is being served on GPU
    When the user sends a chat completion request
    Then the response contains a model reply
    And the response identifies the correct model

  # Lemonade serve + inference (GGUF model). Engine coverage: Lemonade.
  @id:serve-lemonade-inference @requires-gpu @requires-engine:lemonade
  Scenario: 7 - A model served on lemonade responds to inference requests
    Given a managed runtime is active
    And a GGUF model is being served on lemonade
    When the user sends a chat completion request
    Then the response contains a model reply
    And the response identifies the correct model

  # Default-engine serve (no --engine): the effective engine is the platform
  # default from the capability probe. xfail only where that resolves to vLLM
  # (EAI-7333) — see expectations.toml.
  @id:serve-default-engine-working-endpoint @requires-gpu
  Scenario: 6 - Serving a model without specifying an engine produces a working endpoint
    Given a managed runtime is active
    When the user serves a model without specifying an engine
    Then an engine is selected automatically
    And the model is reachable

  # The inference half of scenario 6.
  @id:serve-default-engine-inference @requires-gpu
  Scenario: 6b - A default-engine served model responds to inference requests
    Given a managed runtime is active
    When the user serves a model without specifying an engine
    Then the model responds to inference requests

  # Default engine on Instinct: a vLLM-capable model served without --engine on
  # an Instinct data-center GPU (gfx*-dcgpu) defaults to vLLM. Checks only the
  # selection PLAN, not endpoint readiness. The assertion is vLLM-specific, so it
  # only applies where vLLM is the effective engine — `@requires-engine:vllm`
  # skips it on lemonade-default hosts (Strix Halo), where asserting a vLLM
  # default would be a guaranteed false failure.
  @id:serve-vllm-default-on-instinct @requires-gpu @requires-engine:vllm
  Scenario: 9 - vLLM is the default serving engine on Instinct
    Given a managed runtime is active
    When the user serves a vLLM-capable model without specifying an engine
    Then vLLM is selected as the default engine

  # Readiness contract: when the CLI reports a service ready, inference must work.
  # Engine-agnostic — the served model+engine follow the host (see
  # `a model is being served on GPU`), so this holds the contract on every GPU
  # platform. Where it resolves to vLLM, EAI-7333 makes it xfail (expectations.toml).
  @id:serve-readiness-contract @requires-gpu
  Scenario: 8 - A service reported ready can immediately serve inference
    Given a managed runtime is active
    And a model is being served on GPU
    When the CLI reports the service as ready
    Then an inference request succeeds immediately
