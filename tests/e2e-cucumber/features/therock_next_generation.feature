Feature: TheRock "next" ROCm 10 install layout

  # ROCm 10 ships from a different source layout than the canonical multi-arch
  # release/nightly streams: a separate pip index and a separate tarball catalog
  # whose listing mixes non-release `-tests-` siblings in with the real dist
  # archive. The next layout is an explicit extension, never a default: it is
  # reachable only by pinning a ROCm version whose major is 10 or newer AND
  # naming an exact raw GFX arch, and the canonical streams keep resolving
  # exactly as they do today when neither is asked for.
  #
  # Scenarios 01-05 point source overrides at loopback fixtures through the
  # explicit trust gate. Scenario 06 proves the same overrides are inert when
  # the trust gate is absent; scenario 07 supplies the live GPU evidence.

  # The regression this pins: a "next-first" dispatch that probes the ROCm 10
  # sources before the canonical ones. An unpinned release install must never
  # look at the next layout at all, so the next fixture is served and asserted
  # untouched rather than left unconfigured.
  @id:therock-next-01-unpinned-release-stays-canonical
  Scenario: therock-next-01 - An unpinned release install stays on the canonical multi-arch source
    Given a canonical release pip index fixture and a ROCm 10 pip index fixture
    When the user previews a wheel SDK install for arch gfx1200 with no version pin
    Then the preview resolves the canonical release pip index
    And the preview reports the canonical multi-arch source layout generation
    And the preview never mentions the ROCm 10 pip index

  # `--version 10.0.0` plus a raw arch is the whole opt-in. The device payload is
  # requested as an exact `device-gfx1200` extra on rocm, torch and torchvision;
  # torchaudio carries no device extra because it publishes none.
  @id:therock-next-02-wheel-pins-raw-device-extras
  Scenario: therock-next-02 - A pinned ROCm 10 wheel install resolves the next index with raw device extras
    Given a canonical release pip index fixture and a ROCm 10 pip index fixture
    When the user previews a wheel SDK install for arch gfx1200 pinned to ROCm 10.0.0
    Then the preview resolves the ROCm 10 pip index
    And the preview reports the next source layout generation
    And the preview requests the gfx1200 device extras

  # No silent fallback: a group label carries no exact arch, and the next layout
  # cannot guess one. The install must refuse and say which flag to pass, rather
  # than dropping back to the canonical stream or to a bucket payload.
  @id:therock-next-03-wheel-group-family-refused
  Scenario: therock-next-03 - A pinned ROCm 10 wheel install refuses a group family label
    Given a canonical release pip index fixture and a ROCm 10 pip index fixture
    When the user previews a wheel SDK install for family gfx120X-all pinned to ROCm 10.0.0
    Then the install fails
    And the failure asks for an exact GPU arch and names --family gfx1200

  # Linux-only: `--format tarball` is rejected outright on Windows (native
  # tarball installs are not supported there), so this scenario's premise does
  # not hold on that platform.
  #
  # The `-tests-` sibling carries a LATER mtime than the real dist archive on the
  # live AMD index, so highest-mtime selection alone picks the wrong file.
  @id:therock-next-04-tarball-skips-tests-sibling @requires-os:linux
  Scenario: therock-next-04 - A pinned ROCm 10 tarball install skips the tests sibling
    Given a canonical release tarball fixture and a ROCm 10 tarball fixture with a tests sibling
    When the user previews a tarball SDK install for arch gfx1200 pinned to ROCm 10.0.0
    Then the preview resolves the ROCm 10 tarball catalog
    And the preview reports the next source layout generation
    And the preview selects the real tarball artifact
    And the preview does not select the tests artifact

  # Same refusal on the tarball path, which is a separate dispatch: pinning a
  # version was rejected outright for tarball installs before ROCm 10, so the
  # newly reachable path needs its own proof that it still demands a raw arch.
  @id:therock-next-05-tarball-group-family-refused @requires-os:linux
  Scenario: therock-next-05 - A pinned ROCm 10 tarball install refuses a group family label
    Given a canonical release tarball fixture and a ROCm 10 tarball fixture with a tests sibling
    When the user previews a tarball SDK install for family gfx120X-all pinned to ROCm 10.0.0
    Then the install fails
    And the failure asks for an exact GPU arch and names --family gfx1200

  # Artifact base overrides are a trust boundary. Merely naming an override
  # must not redirect the CLI; the separate opt-in is required. This scenario
  # reaches the live default index (not a fixture, since proving the *default*
  # is what resolves requires the real default), so it runs on the scheduled
  # network lane rather than the hermetic ones above. The unit test
  # `env_override_is_ignored_end_to_end_without_the_opt_in` in
  # `apps/rocm/src/therock.rs` covers the same trust boundary on the blocking
  # lane, without depending on the live network.
  @id:therock-next-06-base-override-requires-opt-in @nightly
  Scenario: therock-next-06 - A ROCm 10 source override is ignored without explicit trust
    Given an untrusted ROCm 10 pip base override
    When the user previews a wheel SDK install for arch gfx1200 pinned to the latest published ROCm 10 version
    Then the preview resolves the default ROCm 10 pip index
    And the preview never mentions the untrusted ROCm 10 pip index

  # Not hermetic like the scenarios above: this one installs for real, against
  # the live stable.repo.amd.com, on a self-hosted GPU runner, with no
  # --family override at all. `resolve_family` already falls back to
  # `detect_host_gfx_target()` when nothing overrides it, so a pinned ROCm 10
  # install should resolve the runner's real GPU into the exact arch the next
  # layout needs without the user ever typing a raw GFX code. The fixture
  # scenarios above can't prove this: their fixtures serve a fixed gfx1200
  # regardless of what GPU the runner actually has.
  @id:therock-next-07-live-install-auto-detects-arch @requires-gpu @nightly
  Scenario: therock-next-07 - Installing the SDK from the live ROCm 10 preview source auto-detects the exact arch
    Given a machine with no CLI-managed runtimes
    When the user installs the SDK from the ROCm 10 preview source with no family override
    Then the install used the ROCm 10 preview source
    And the install requested the device extras for this host's detected GPU
    And a runtime is registered
    And the runtime is set as active
    And the runtime includes an inference engine
    And the ROCm 10 runtime passes SDK and Torch GPU probes
