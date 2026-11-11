Feature: Model catalog listing

  # ROCMAI-555: the AIM engine recipe (Qwen/Qwen3.5-9B) is curated out of the
  # default `rocm model` view (it is not `featured`), so only `--verbose`
  # actually lists it. `model` and its `models` alias share one parser branch,
  # so exercising `model --verbose` covers the epic's "rocm model or rocm
  # models" wording either way.
  @id:model-catalog-lists-aim-recipe
  Scenario: model-catalog-01 - Discovering available AIM images via rocm model
    When the user asks for the model catalog in verbose detail
    Then the catalog lists an AIM engine recipe

  # The epic names two reasons a listed AIM image can be unusable: an
  # unsupported GPU family, or no Docker. Only the Docker check runs at the
  # catalog layer today (see `engine_ruled_out_by_platform` in main.rs);
  # GPU-family fit is a property of the entry (`supported_gfx_families`, shown by
  # `--verbose`) but is only enforced by `rocm serve`, not by the listing.
  # This plants a `docker` that always fails so the
  # unavailable branch is exercised deterministically on every host, including
  # ones where Docker really is reachable (an earlier variant compared against
  # the host's real Docker state, which asserted a different outcome per host).
  # Linux only: native Windows rules AIM out for the OS before it asks Docker,
  # so it reports `unsupported_native_windows`, not the Docker reason asserted here.
  @id:model-catalog-flags-aim-unavailable-with-no-docker-daemon @requires-os:linux
  Scenario: model-catalog-02 - AIM image is flagged unavailable with no Docker daemon
    Given no Docker daemon is reachable
    When the user asks for the model catalog in verbose detail
    Then the catalog's AIM entry is flagged unavailable for lack of Docker

  # ROCMAI-556: an AIM entry pins an image that may since have been withdrawn, so
  # `rocm model --verbose` warns once the entry was last verified more than 180
  # days ago. The entry is shipped with a verification timestamp; this plants a
  # signed catalog whose timestamp is decades old, so the warning is exercised
  # without depending on the date the suite runs.
  @id:model-catalog-warns-on-stale-aim-entry
  Scenario: model-catalog-03 - A stale AIM catalog entry is flagged in rocm model --verbose
    Given a recipe catalog whose AIM entry was last verified long ago
    When the user asks for the model catalog in verbose detail
    Then the catalog warns that the AIM entry may no longer be current

  # The catalog is also read by tool-using clients — the dashboard's Chat tab
  # and any MCP client — and what they were handed was the verbose listing, not
  # the recommendation list a person sees. On a developer host that is 189 KB
  # against under 1 KB, which is more than a locally served model's whole
  # context, so asking the assistant what this machine can run ended with the
  # model's provider refusing the request instead of an answer.
  @id:model-catalog-tool-client-gets-the-recommendations
  Scenario: model-catalog-04 - A tool-using client asking about models gets the recommendations
    When a tool-using client asks for the model catalog
    Then the client is given the recommended models without the per-recipe diagnostics
    And what the client is given is small enough for a locally served model to read

  # The diagnostics are not gone, they are opt-in: a client that wants them
  # asks for them, the same way a person types `rocm model --verbose`.
  @id:model-catalog-tool-client-can-ask-for-diagnostics
  Scenario: model-catalog-05 - A tool-using client can still ask for the full diagnostics
    When a tool-using client asks for the model catalog in verbose detail
    Then the client is given the per-recipe diagnostics
