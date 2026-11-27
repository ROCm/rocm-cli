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

  # `rocm model` calls itself "Recommended models" and chose them with a flag
  # that knew nothing about the machine: the four it carried need 17-62 GiB,
  # while the eight recipes at or below 8 GiB carried it on none of them. A
  # small or integrated GPU was recommended four models it cannot run and shown
  # none of the ones it can.
  #
  # These run on the ungated lane: the host's GPU capacity is planted, so they
  # describe the same machine whatever the runner has -- which is also the only
  # way the narrowed listing is reachable at all, since the no-GPU lane measures
  # no capacity and a GPU lane cannot be made small.
  @id:model-catalog-recommends-only-what-this-machine-can-run @requires-os:linux
  Scenario: model-catalog-06 - A small machine is recommended models it can run
    Given a machine whose GPU has room for only the smaller models
    When the user asks which models are recommended
    Then every model offered is one this machine can run
    And the user is told the offer was narrowed to this machine
    And the user is not told anything about shared memory

  @id:model-catalog-says-when-nothing-fits @requires-os:linux
  Scenario: model-catalog-07 - A machine too small for any curated model is told so
    Given a machine whose GPU is too small for any curated model
    When the user asks which models are recommended
    Then the user is told none of the curated models fit
    And no model is offered

  # The behaviour most likely to regress silently, and the reason the listing
  # does not simply filter on whatever figure it has: a machine nobody could
  # measure must not be told its options are narrow.
  @id:model-catalog-does-not-narrow-an-unmeasured-machine @requires-os:linux
  Scenario: model-catalog-08 - A machine whose memory could not be read is not narrowed
    Given a machine whose GPU memory cannot be read
    When the user asks which models are recommended
    Then the recommendations are not narrowed to this machine
    And the user is told the list is not specific to it

  # An assistant answering "what can this machine run" reads the same listing
  # through its tools. Same renderer, different caller -- so the host has to be
  # measured on both paths, and a caller that passes none hands the assistant
  # the unnarrowed list while the terminal shows the narrowed one.
  @id:model-catalog-assistant-is-told-what-the-user-is-told @requires-os:linux
  Scenario: model-catalog-09 - An assistant is told what the user is told
    Given a machine whose GPU has room for only the smaller models
    When the user asks which models are recommended
    And an assistant on the same machine asks the same question through its tools
    Then the assistant is given the same recommendations the user saw

  # The machine the whole listing exists for: an integrated GPU with no memory
  # of its own, which was offered the unfiltered catalog because the only figure
  # it has is a pool it shares with the rest of the system. It is narrowed on
  # that pool now -- and told what the pool is, because a ceiling the desktop
  # also draws on is not memory set aside for the model.
  @id:model-catalog-narrows-a-machine-whose-graphics-share-memory @requires-os:linux
  Scenario: model-catalog-10 - A machine whose graphics share system memory is recommended models that fit it
    Given a machine whose graphics share memory with the rest of the system, with room for only the smaller models
    When the user asks which models are recommended
    Then every model offered is one this machine can run
    And the offer includes a model too large for the carve-out the adapter reports
    And each offered row says the memory it was weighed against is shared
    And the user is told the offer was narrowed to this machine
    And the user is told that memory is shared with everything else running

  # Sharing memory is not the same as having a readable amount of it. An
  # integrated GPU whose pool could not be read is still a machine nobody
  # measured, and gets the unmeasured answer rather than a narrowing.
  @id:model-catalog-does-not-narrow-unreadable-shared-memory @requires-os:linux
  Scenario: model-catalog-11 - A machine whose shared memory cannot be read is not narrowed
    Given a machine whose graphics share memory with the rest of the system, and that amount cannot be read
    When the user asks which models are recommended
    Then the recommendations are not narrowed to this machine
    And the user is told the list is not specific to it

  # The same machine with no ROCm build for its graphics, which is how an
  # integrated GPU usually arrives: nothing reports a VRAM figure, because there
  # is no VRAM. The pool is still readable, so the answer is still narrowed --
  # and it has to match what `rocm diagnose --model` says about the same host,
  # which judges it on that pool.
  #
  # This machine has no carve-out to weigh against by mistake, so -10's
  # carve-out discrimination has no subject here. What it can still be told
  # apart from is a listing narrowed on too large a figure, or not narrowed at
  # all -- hence the absence step, which names the models the pool cannot hold.
  @id:model-catalog-narrows-shared-memory-with-no-vram-reading @requires-os:linux
  Scenario: model-catalog-12 - A machine whose graphics report no memory of their own is narrowed on the shared pool
    Given a machine whose graphics share memory with the rest of the system and report no memory of their own
    When the user asks which models are recommended
    Then every model offered is one this machine can run
    And no model too large for that pool is offered
    And each offered row says the memory it was weighed against is shared
    And the user is told that memory is shared with everything else running

  # The per-recipe detail reaches the same host, and has to describe the figure
  # the same way: a row that calls a shared pool "VRAM" asserts memory the part
  # does not have.
  @id:model-catalog-detail-names-shared-memory @requires-os:linux
  Scenario: model-catalog-13 - The per-recipe detail calls shared memory what it is
    Given a machine whose graphics share memory with the rest of the system, with room for only the smaller models
    When the user asks for the model catalog in verbose detail
    Then each recipe's memory verdict names the shared pool rather than VRAM
    And the remedy it offers is one this machine can act on
