Feature: Runtime configuration

  @id:runtime-install-sdk-active @requires-gpu @nightly
  Scenario: 1 - Installing the SDK makes it the active runtime
    Given a machine with no CLI-managed runtimes
    When the user installs the SDK
    Then a runtime is registered
    And the runtime is set as active
    And the runtime includes an inference engine

  # Dogfooding #17: re-provisioning was observed writing inside the previous
  # runtime, producing a recursively nested `runtimes/wheel/.../runtimes/wheel/`
  # path that bloats paths and breaks `services/*.log` globs. Assert the active
  # runtime's folder path has no such recursive segment. GPU-gated (needs a real
  # install so the folder path is populated).
  @id:runtime-path-not-nested @requires-gpu
  Scenario: 3 - The managed runtime path is not nested inside another runtime
    Given a managed runtime is active
    When the user inspects the system
    Then the managed runtime folder path is not recursively nested

  # The SDK and the engine share one Python environment, so a second
  # `install sdk` wrote the SDK's torch stack over the build the engine pins. The
  # engine still resolved, so the install reported success and every health surface
  # kept saying `ready` — the first signal was a serve failure naming neither. Needs
  # a real SDK install, a real engine install, and a second SDK install, so it runs
  # on the nightly GPU lane. `@requires-engine:vllm` because only vLLM shares the
  # runtime environment; Lemonade manages its own.
  @id:runtime-sdk-reinstall-keeps-engine-consistent @requires-gpu @requires-engine:vllm @nightly
  Scenario: 4 - Reinstalling the SDK leaves the installed engine's requirements satisfied
    Given a managed runtime with an inference engine already installed
    When the user installs the SDK again
    Then the install reports the engine's requirements as satisfied

  # The GPU E2E lanes no longer install the shared runtime once and keep it
  # forever: `xtask e2e-prewarm` asks `rocm update` whether the channel index has
  # published a newer version, and installs it side-by-side when it has (EAI-8057).
  # That makes CI depend on the freshness line this scenario pins. A unit test on a
  # hand-written fixture cannot catch the renderer drifting away from the parser —
  # only running the real command can, which is why this is a scenario and not just
  # an xtask test. Cheap enough for the per-PR lanes: one CLI call against the
  # already-installed shared runtime. `status=error` is an ACCEPTED outcome, so an
  # offline runner reports honestly instead of flaking.
  @id:runtime-update-reports-freshness @requires-gpu
  Scenario: 5 - The update check reports the active runtime's freshness
    Given a managed runtime is active
    When the user checks for runtime updates
    Then the report states the runtime's freshness against the channel index

  # The install used to record the path it was handed rather than the folder the
  # files land in, so reaching `data/runtimes` through a link made the runtime name
  # a folder that disappeared with the link — taking every console-script shebang
  # in the environment with it, while the files stayed where they were written
  # (rocm-cli#315). The E2E harness itself creates exactly that link when a scenario
  # opts into the shared pre-warmed runtime, so the shared tree on a runner was the
  # thing being poisoned. Previewing the install is enough to pin this and needs no
  # GPU and no download: the planned folder is resolved before the preview prints
  # it, so a regression shows up in the plan. `--family` is supplied because
  # without a GPU there is no target to detect.
  #
  # `@nightly` is not about this scenario's own cost — it runs in about eight
  # seconds, nearly all of it resolving the channel index. It is that the no-GPU
  # mock lane runs 64 scenarios at once, and that much concurrent network work is
  # enough to push `eai-7960-gen-tps-held-after-scrape-failure` and
  # `eai-7960-gen-tps-expiry-boundary` past the validity window they assert on
  # (measured: both fail 3/3 with this scenario on the mock lane, and pass with the
  # very same scenario once the suite is serialized). Those two are timing-fragile
  # under load, which is their own problem to fix; until then this runs on the
  # nightly lanes, where a GPU is present and scenarios are serialized.
  @id:runtime-install-records-the-real-folder @nightly
  Scenario: 8 - Previewing an install through a linked runtimes folder names the real folder
    Given a machine whose runtimes folder is a link to somewhere else
    When the user previews an SDK install
    Then the planned runtime folder is inside the folder the link points at
    And the planned runtime folder is not expressed through the link

  # Linux-only: the step adopts a standard `/opt/rocm` install with a Unix python
  # path. On Windows those paths don't exist (the CLI resolves `/usr/bin/python3`
  # to a bogus `C:/usr/bin/python3` and errors on the missing path before it can
  # emit the install-type guidance), so the scenario's premise doesn't hold there.
  @id:runtime-adopt-preexisting-rejected @requires-os:linux
  Scenario: 2 - Adopting a pre-existing ROCm install is rejected with guidance
    Given a machine with a standard ROCm install
    When the user tries to adopt the existing install
    Then the adoption is refused
    And the error explains which install types can be adopted

  # Regression guard for the bug where the release-channel multi-arch pip index
  # was queried with a per-family path segment (.../whl-multi-arch/{family}/),
  # which 403s because that index is flat and 404/403-ed straight into the
  # stale classic index every time, silently pinning every install to
  # whatever version predated the migration. `--family` bypasses GPU
  # auto-detection so this needs no GPU, and `--dry-run` resolves the real
  # index without installing anything.
  #
  # `@nightly` for the same reason as scenario 8 above: this dry-run still
  # resolves the real channel index over the network (dry-run only skips the
  # venv/download, not index resolution), and the no-GPU mock lane's 64-way
  # concurrency from that extra network work is what pushes
  # `eai-7960-gen-tps-held-after-scrape-failure` and
  # `eai-7960-gen-tps-expiry-boundary` past their validity window. Runs on the
  # nightly lanes instead, where scenarios are serialized.
  @id:runtime-install-sdk-release-index-shape @nightly
  Scenario: 4 - Resolving the SDK from the release channel never uses the broken multi-arch URL shape
    When the user dry-runs installing the SDK for a known family
    Then the resolved package index is not the broken per-family multi-arch path
