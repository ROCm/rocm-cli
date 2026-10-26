Feature: Model catalog listing

  # ROCMAI-555: the AIM engine recipe (aim-qwen3.5-9b) is curated out of the
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
  # GPU-family fit is a `rocm serve`/`rocm diagnose --model` readiness check,
  # not a catalog property. This cross-checks against the harness's own Docker
  # probe rather than hardcoding "unavailable", so it stays honest on a host
  # (e.g. the Strix Halo AIM runner) where Docker really is reachable.
  @id:model-catalog-flags-aim-unavailable-without-docker
  Scenario: model-catalog-02 - Host cannot run a listed AIM image
    When the user asks for the model catalog in verbose detail
    Then the catalog's AIM entry matches this host's real Docker availability

  # model-catalog-02 cross-checks the harness's own `docker info` probe, which
  # only proves the two probes agree, not that the gating logic itself is
  # correct — it stays green even if the "no Docker" branch were deleted
  # entirely on a Docker-reachable host. This plants a `docker` that always
  # fails so the unavailable branch is exercised regardless of host state.
  @id:model-catalog-flags-aim-unavailable-with-no-docker-daemon
  Scenario: model-catalog-03 - AIM image is flagged unavailable with no Docker daemon
    Given no Docker daemon is reachable
    When the user asks for the model catalog in verbose detail
    Then the catalog's AIM entry is flagged unavailable for lack of Docker
