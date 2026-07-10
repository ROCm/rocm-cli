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

  # vLLM serve + inference (safetensors model). Engine coverage: vLLM.
  @gpu
  Scenario: 5 - A served model responds to inference requests
    Given a managed runtime is active
    And a model is being served on GPU
    When the user sends a chat completion request
    Then the response contains a model reply
    And the response identifies the correct model

  # Lemonade serve + inference (GGUF model). Engine coverage: Lemonade.
  # Currently expected-failure: on MI300X Lemonade falls back to its Vulkan
  # llama.cpp backend instead of system ROCm, and inference hangs (EAI-7052 —
  # Lemonade should use the installed ROCm libraries). Serving/discovery works;
  # only inference is blocked, so this stays a known bug until EAI-7052 lands.
  @gpu @expected-failure @expected-failure-EAI-7052
  Scenario: 7 - A model served on lemonade responds to inference requests
    Given a managed runtime is active
    And a GGUF model is being served on lemonade
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

  # Default engine on Instinct: a vLLM-capable model served without --engine on
  # an Instinct data-center GPU (gfx*-dcgpu, e.g. MI300X) defaults to vLLM. The
  # GPU-family default is pinned engine-side by rocm-core unit tests
  # (instinct_dcgpu_family_prefers_vllm); this exercises it through the real CLI.
  @gpu
  Scenario: 9 - vLLM is the default serving engine on Instinct
    Given a managed runtime is active
    When the user serves a vLLM-capable model without specifying an engine
    Then vLLM is selected as the default engine

  # Readiness contract (EAI-7333): when the CLI reports a service ready, inference
  # must actually work — no additional wait. Expect-pass on vLLM (whose ready
  # signal coincides with inference-readiness), so it guards against the
  # healthcheck's /v1/models-only readiness regressing to a false positive. The
  # engine-level cause is pinned by a rocm-core unit test
  # (models_endpoint_readiness_does_not_imply_inference_ready).
  @gpu
  Scenario: 8 - A service reported ready can immediately serve inference
    Given a managed runtime is active
    And a model is being served on GPU
    When the CLI reports the service as ready
    Then an inference request succeeds immediately
