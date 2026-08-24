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
  # below, so it stays pinned to vLLM (the slug names the engine). It is also the
  # vLLM per-PR canary: one real vLLM serve runs on every PR so a broken serve is
  # caught before merge, while the heavier `@merge-queue` serves (6, 6b, 8) run
  # only in the merge queue.
  @id:serve-vllm-inference @requires-gpu @requires-engine:vllm
  Scenario: 5 - A served model responds to inference requests on vLLM
    Given a managed runtime is active
    And a model is being served on GPU
    When the user sends a chat completion request
    Then the response contains a model reply
    And the response identifies the correct model

  # Large-model coverage (dogfooding W9): serve a representative large model for
  # each GPU platform end-to-end at least once. MI300X uses Qwen/Qwen3.6-27B through
  # vLLM; Strix Halo uses the hardware-verified
  # unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL checkpoint through Lemonade. These slow
  # loads stay off the ordinary per-PR path, and the longer readiness timeout also
  # gives the first inference request enough time to complete.
  @id:serve-large-model-inference @requires-gpu @serve-timeout:2400 @nightly
  Scenario: 10 - A large platform-specific model serves and responds to inference
    Given a managed runtime is active
    And a large model is being served on GPU
    When the user sends a chat completion request
    Then the response contains a model reply
    And the response identifies the correct model

  # Lemonade serve + inference (GGUF model). Engine coverage: Lemonade. The
  # lemonade per-PR canary: one real lemonade serve runs on every PR (the
  # counterpart to the vLLM canary above).
  @id:serve-lemonade-inference @requires-gpu @requires-engine:lemonade
  Scenario: 7 - A model served on lemonade responds to inference requests
    Given a managed runtime is active
    And a GGUF model is being served on lemonade
    When the user sends a chat completion request
    Then the response contains a model reply
    And the response identifies the correct model

  # HF-checkpoint direct-serve canary (EAI-8026). An `owner/repo:variant` ref
  # bypasses Lemonade's model router and runs a packaged llama-server directly
  # (`serve_hf_checkpoint`), which bails explicitly if no backend binary is found
  # under bin/llamacpp/<backend>/ — unlike the short-recipe-name path above, whose
  # managed-lemonade fallback would mask the identical failure behind the
  # unrelated EAI-7423 xfail. Reuses the same small Qwen3-0.6B-GGUF checkpoint as
  # `serve-lemonade-inference` (cache-shared, no extra download) so this stays a
  # fast per-PR canary rather than needing the @nightly large-checkpoint path.
  @id:serve-hf-checkpoint-inference @requires-gpu @requires-engine:lemonade
  Scenario: 14 - A canonical Hugging Face checkpoint serves and responds to inference
    Given a managed runtime is active
    And a canonical Hugging Face GGUF checkpoint is being served on lemonade
    When the user sends a chat completion request
    Then the response contains a model reply
    And the response identifies the correct model

  # Default-engine serve (no --engine): the effective engine is the platform
  # default from the capability probe, so this covers whichever engine the host
  # would actually pick.
  @id:serve-default-engine-working-endpoint @requires-gpu @merge-queue
  Scenario: 6 - Serving a model without specifying an engine produces a working endpoint
    Given a managed runtime is active
    When the user serves a model without specifying an engine
    Then an engine is selected automatically
    And the model is reachable

  # The inference half of scenario 6.
  @id:serve-default-engine-inference @requires-gpu @merge-queue
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
  # platform. Readiness is gated on a real inference probe, which is what makes
  # this contract hold rather than race the model load.
  @id:serve-readiness-contract @requires-gpu @merge-queue
  Scenario: 8 - A service reported ready can immediately serve inference
    Given a managed runtime is active
    And a model is being served on GPU
    When the CLI reports the service as ready
    Then an inference request succeeds immediately

  # GPU-required enforcement (EAI-7400). Under the GPU-required default, a host
  # with no usable AMD GPU must refuse to serve — before any engine is prepared or
  # launched — with an actionable message, never a CPU or device-0 fallback. Runs
  # on the no-GPU mock host, so it gates every PR (@requires-no-gpu, no GPU needed).
  @id:serve-no-gpu-fails-fast @requires-no-gpu
  Scenario: 11 - Serving is refused on a host with no AMD GPU
    When the user serves a model under the GPU-required default
    Then serving is refused before any engine starts
    And the user is told no AMD GPU was detected

  # The masked-device path: on a real GPU host where every device is hidden, the
  # GPU-required serve must treat it as "no GPU" and refuse, not fall back. Runs on
  # GPU hardware (Strix Halo / Instinct).
  @id:serve-masked-devices-fail @requires-gpu @requires-os:linux
  Scenario: 12 - Serving is refused when every GPU is masked from view
    When the user serves a model with every GPU masked from view
    Then serving is refused before any engine starts
    And the user is told no AMD GPU was detected

  # Honest device selection: a `--gpu` index that does not exist on the host is
  # rejected outright, never silently remapped to another device (no device-0
  # fallback). Runs on GPU hardware: on a no-GPU host the GPU-required pre-flight
  # refuses ("no usable AMD GPU") before the index is ever validated, so the
  # index-specific rejection can only be observed where a real device is present.
  @id:serve-absent-gpu-index-rejected @requires-gpu @requires-os:linux
  Scenario: 13 - Serving pinned to a GPU that does not exist is refused
    When the user serves a model pinned to a GPU index that does not exist
    Then serving is refused before any engine starts
    And the user is told that GPU index is unavailable

  # An address the user names is a requirement, not a preference. When something
  # else already holds it, the serve has to stop there — before the model is
  # loaded and before a service is recorded — instead of letting the engine
  # discover the collision on its own bind, which fails minutes later and leaves
  # a dead service behind. Runs on GPU hardware for the same reason as scenario
  # 13: on a no-GPU host the GPU-required pre-flight refuses before the address
  # is ever examined, so the address-specific refusal is unobservable there.
  # Not @merge-queue despite needing a GPU — that tag is for heavy real serves,
  # and this one never launches an engine: it is refused at the address check,
  # so it costs a process start and belongs on the per-PR path.
  @id:serve-named-address-already-taken @requires-gpu
  Scenario: 14 - Serving on an address something else holds is refused
    Given a managed runtime is active
    And another process holds the address the user is going to name
    When the user serves a model on that address
    Then serving is refused before any engine starts
    And the user is told the address is already in use
    And no service is recorded for the refused launch
