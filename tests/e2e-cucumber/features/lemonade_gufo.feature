Feature: Serving Lemonade's Gufo models

  # Gufo is an inference engine with model-specific kernels for AMD Strix Halo
  # (gfx1151). Lemonade runs it from a container image as its `gufo` recipe and
  # assigns a handful of models to it in its own registry. `rocm serve` of such a
  # model with the lemonade engine asks Lemonade whether Gufo can run on this
  # machine before the model or the Gufo image is downloaded, and refuses when it
  # cannot: on another GPU, or before the container setup is complete, where
  # Lemonade's own documentation says a container would silently run on the CPU.
  #
  # lemonade-gufo-01 to -03 and -05 need no GPU and download nothing, so they run
  # on the mock lane every PR. They plant a Lemonade install and stand in for
  # Lemonade's answer through the `rocm/e2e-test-hooks` build `cargo xtask e2e`
  # makes (`ROCM_E2E_LEMONADE_SYSTEM_INFO`, which also waives `rocm serve`'s
  # no-usable-GPU refusal, as the other scripted Lemonade hooks do), so each
  # state is reached on any host.
  #
  # lemonade-gufo-04 serves a real Gufo model, a download of about 20 GB plus
  # the Gufo container image, so it is nightly and runs only on a bare-metal
  # Strix Halo Linux host with Podman or Docker set up for Lemonade.

  @id:lemonade-gufo-refused-on-unsupported-gpu
  Scenario: lemonade-gufo-01 - Serving a Gufo model on a GPU Gufo does not support is refused before the model is downloaded
    Given Lemonade is installed and lists Gufo models
    And Lemonade reports that Gufo cannot run on this machine's GPU
    When the user serves a Gufo model with Lemonade
    Then serving the Gufo model is refused before anything is started
    And the user is told Gufo runs only on a Strix Halo GPU

  @id:lemonade-gufo-refused-until-container-setup
  Scenario: lemonade-gufo-02 - Serving a Gufo model before its container setup is complete is refused with the steps to finish it
    Given Lemonade is installed and lists Gufo models
    And Lemonade reports that Gufo's container setup is not complete
    When the user serves a Gufo model with Lemonade
    Then serving the Gufo model is refused before anything is started
    And the user is shown the setup steps Lemonade reports

  @id:lemonade-gufo-engines-report-availability
  Scenario: lemonade-gufo-03 - The engine list says when Gufo needs container setup, and which models it serves
    Given Lemonade is installed and lists Gufo models
    And Lemonade reports that Gufo's container setup is not complete
    And the user has tried to serve a Gufo model with Lemonade
    When the user lists the engines
    Then the engine list says Gufo needs container setup before it can run
    And the engine list names the Gufo models

  # @requires-bare-metal: Lemonade gives the Gufo container /dev/dri and
  # /dev/kfd (the `gufo:rocm` container entry in Lemonade's
  # src/cpp/include/lemon/backends/gufo/gufo.h), and a WSL2 guest reaches its
  # GPU through /dev/dxg instead (the tag table in the e2e README), so the Strix
  # Halo WSL nightly lane skips it.
  @id:lemonade-gufo-strix-halo-inference @requires-gpu @requires-engine:lemonade @requires-os:linux @requires-bare-metal @requires-gfx:gfx1151 @nightly @serve-timeout:3600
  Scenario: lemonade-gufo-04 - A Gufo model served on a Strix Halo GPU answers inference requests
    Given a Gufo model is being served on lemonade
    When the user sends a chat completion request
    Then the response contains a model reply
    And the serve log says the model is served by Gufo

  # The flags are refused before Lemonade is asked anything (`resolve_gufo_model`
  # in engines/lemonade/src/gufo.rs calls `refuse_llama_server_flags` before
  # `probe_status`), so even where Lemonade would say Gufo cannot run here, the
  # flags are what the user is told about.
  @id:lemonade-gufo-sampling-flags-refused
  Scenario: lemonade-gufo-05 - Serving a Gufo model with sampling defaults is refused rather than run without them
    Given Lemonade is installed and lists Gufo models
    And Lemonade reports that Gufo cannot run on this machine's GPU
    When the user serves a Gufo model with Lemonade and a sampling temperature
    Then serving the Gufo model is refused before anything is started
    And the user is told Gufo does not take the sampling flags
