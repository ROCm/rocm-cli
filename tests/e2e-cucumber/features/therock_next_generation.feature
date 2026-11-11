Feature: TheRock "next" ROCm 10 install layout

  # AMD now publishes every channel to a newer source layout than the canonical
  # multi-arch release/nightly streams: a separate pip index and a separate
  # tarball catalog per channel (`stable.repo.amd.com`, `nightly.repo.amd.com`),
  # whose tarball listing mixes non-release `-tests-` siblings in with the real
  # dist archive. The canonical streams no longer receive builds, so a request
  # with no pin reads the newer layout, and a pin reads whichever layout holds
  # that build: a stable ROCm 10+ release or a nightly built on or after
  # 2026-08-23 in the newer layout, an earlier ROCm 7 release or an earlier
  # nightly in the canonical stream. The newer layout picks its device payload
  # by exact arch, so a request that reads it also needs a raw GFX arch.
  #
  # Scenarios 01-05 and 16-22 point source overrides at loopback fixtures
  # through the explicit trust gate. Scenario 06 proves the same overrides are
  # inert when the trust gate is absent; scenario 07 supplies the live GPU
  # evidence.
  #
  # A preview of the newer layout has a real `uv pip compile` choose the torch
  # and torchvision versions, and a preview never fetches `uv` itself. The
  # scenarios below that print resolved package specs (01, 02, 10, 11, 13, 15,
  # 20 and 22) therefore start from an installed `uv`, which is the
  # precondition they always had when the preview fetched one for them.
  # runtime-27 and runtime-28 in `runtime_setup.feature` pin the preview's
  # behaviour with a `uv` present and with none.

  # The release stream the CLI used to install by default stopped at ROCm 7.14,
  # so an install with no pin that kept reading it would silently stay on old
  # builds while the current ones sat in the newer layout. The canonical fixture
  # is served and asserted untouched rather than left unconfigured, so reading
  # it would show up as a failure and not as a coincidence.
  @id:therock-next-01-unpinned-release-reads-the-newest-layout
  Scenario: therock-next-01 - An unpinned release install reads the newest layout's source
    Given a canonical release pip index fixture and a ROCm 10 pip index fixture
    And a uv is already installed
    When the user previews a wheel SDK install for arch gfx1200 with no version pin
    Then the preview resolves the ROCm 10 pip index
    And the preview reports the next source layout generation
    And the preview never mentions the canonical release pip index

  # `--version 10.0.0` plus a raw arch selects the newer layout explicitly, the
  # same layout an unpinned install reads. The device payload is requested as an
  # exact `device-gfx1200` extra on rocm, torch and torchvision; torchaudio
  # carries no device extra because it publishes none.
  @id:therock-next-02-wheel-pins-raw-device-extras
  Scenario: therock-next-02 - A pinned ROCm 10 wheel install resolves the next index with raw device extras
    Given a canonical release pip index fixture and a ROCm 10 pip index fixture
    And a uv is already installed
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
  # `apps/rocm/src/therock/source.rs` covers the same trust boundary on the blocking
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
  # regardless of what GPU the runner actually has. Deliberately not tagged
  # `@requires-engine:vllm`: that tag would skip this scenario's arch-detection
  # assertions wherever vLLM can't start (e.g. a lemonade-only Strix host),
  # narrowing coverage this scenario exists to provide. See therock-next-09 for
  # the vLLM-specific route, which pays for its own runtime instead.
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

  # The regression this pins is distinct from therock-next-02's: that scenario
  # proves a *fresh* install resolves an exact arch when the user supplies one.
  # This one proves *updating* an already-installed next-layout runtime works
  # when all the CLI has on hand is the manifest's group family (`gfx120X-all`)
  # — exactly what every installed ROCm 10 runtime's manifest carries, since
  # `--family` is only ever typed once. A fix that recovers the exact arch for
  # planning (deciding a newer version exists) but not for applying (actually
  # resolving and installing it) leaves this scenario red.
  @id:therock-next-08-update-apply-recovers-exact-arch-from-grouped-family
  Scenario: therock-next-08 - Updating a ROCm 10 wheel runtime resolves past its grouped family
    Given a canonical release pip index fixture and a ROCm 10 pip index fixture
    And a registered ROCm 10 wheel runtime with a grouped family
    When the user previews applying the pending update to that runtime
    # With the toolchain because that runtime's recorded specs have it: an
    # update must reinstall what was installed, not the current default.
    Then the preview requests the gfx1200 device extras with the toolchain

  # Proves the vLLM ROCm 10.x wheel discovery route rather than the fixed pin
  # table other SDK versions use: AMD publishes vllm, flash-attn, and
  # amd-aiter under rotating dev-tag filenames for ROCm 10.x, so the adapter
  # resolves each package's current wheel from AMD's live index with
  # `uv pip install --dry-run --reinstall` before installing pinned to what
  # that reported. No fixture can serve a rotating dev-tag filename and stay
  # meaningful, so this is the only place that mechanism runs against the
  # real index at all. Also the only place that proves ROCm CLI provisions the
  # cp314 interpreter this route's wheels require, rather than the cp312
  # every earlier SDK version uses. Provisions its own ROCm 10 runtime rather
  # than reusing therock-next-07's, so that scenario's arch-detection coverage
  # still runs on hosts that can't start vLLM.
  #
  # Not `@nightly`: `install_vllm_rocm10_discover` always runs its 403-tolerant
  # `--config-file` install and forced torch realignment regardless of which
  # row (10.0.x production vLLM, 10.1.x staging vLLM; both take their torch
  # stack from `whl-next`) the live preview index currently serves as newest,
  # so this is the live per-PR regression guard for both fixes (ROCMAI-439)
  # as well as the discovery mechanism itself — accepted as
  # a real fresh-SDK-install cost on every vLLM-capable self-hosted GPU lane.
  #
  # The serve+inference tail matters because `serve-vllm-inference`
  # (model_serving.feature) does not cover this route: its runtime comes from
  # a plain unpinned `rocm install sdk`, never the ROCm 10 preview source. A
  # discovered wheel that installs cleanly but can't actually load or answer a
  # request (bad torch realignment, cp314 ABI mismatch, a `uv.toml` fallback
  # pulling an incompatible transitive dep) would pass every assertion above
  # this line and still be unservable, so this scenario reuses the same
  # serve/chat steps every other engine-canary scenario does rather than
  # stopping at "install reported the right pins".
  @id:therock-next-09-live-install-reports-vllm-rocm10x-discovery-pins @requires-gpu @requires-engine:vllm
  Scenario: therock-next-09 - Installing vLLM against a live ROCm 10 preview runtime reports the discovery pins
    Given a machine with no CLI-managed runtimes
    When the user installs the SDK from the ROCm 10 preview source with no family override
    Then a runtime is registered
    And the runtime is set as active
    And the runtime includes an inference engine
    And the ROCm 10 runtime provisioned a cp314 Python interpreter
    When the user reinstalls vllm
    Then the install reports the vLLM ROCm 10.x discovery pins
    Given a model is being served on GPU
    When the user sends a chat completion request
    Then the response contains a model reply
    And the response identifies the correct model

  # The opt-in half of therock-next-02. Both polarities run here, on the mock
  # lane, because this is the only place the flag's effect on the real install
  # plan is observable without a GPU and a multi-GiB download.
  #
  # Deliberately not `@nightly`. Gating it would make the asymmetry that
  # therock-next-02 alone cannot cover: hardcoding `include_devel = true` inside
  # `install_wheel_runtime` fails therock-next-02, but hardcoding it to FALSE —
  # making `--devel` a silent no-op on every real install — would pass every
  # blocking check with this scenario off the lane. The unit tests cannot close
  # that gap: they pass the flag literally, so they pin the helpers, not what
  # `install_wheel_runtime` passes them.
  #
  # This scenario was briefly moved to the nightly lane because the extra
  # index-resolving work tipped `dash-gen-tps-held-after-scrape-failure` and
  # `dash-gen-tps-expiry-boundary` past the validity window they assert on.
  # Those two now hold their observation clock instead of racing the host
  # (rocm-cli#412), which is the layer that was actually broken, so the reason
  # to displace this coverage is gone.
  #
  # The `@id:` still reads `-09-`: this scenario was written as therock-next-09
  # and the display index moved when main landed one ahead of it. The id is the
  # stable identifier and is deliberately not renumbered with the index.
  @id:therock-next-09-wheel-devel-adds-the-toolchain
  Scenario: therock-next-10 - A pinned ROCm 10 wheel install adds the toolchain when asked
    Given a canonical release pip index fixture and a ROCm 10 pip index fixture
    And a uv is already installed
    When the user previews a wheel SDK install for arch gfx1200 pinned to ROCm 10.0.0 with the toolchain
    Then the preview resolves the ROCm 10 pip index
    And the preview requests the gfx1200 device extras with the toolchain

  # The regression this pins: torchaudio's published metadata names no ROCm
  # build, so a pin to any ROCm 10 version but the newest resolved torchaudio
  # from the newest build, and the install refused the mixed stack. The other
  # scenarios publish one build per package and could not see it. The pinned
  # build also publishes a torchaudio rebuild (`2.9.1.2`), the shape the live
  # index uses, and the plan must take that rather than the original release.
  # ROCm 10.0 has no hardware-verified torchaudio recorded, so that newest build
  # is a fallback, and the plan says so.
  @id:therock-next-11-older-pin-keeps-one-rocm-build
  Scenario: therock-next-11 - A ROCm 10 wheel install pinned below the newest build installs only that build
    Given a ROCm 10 pip index fixture that also publishes a newer ROCm 10 build
    And a uv is already installed
    When the user previews a wheel SDK install for arch gfx1200 pinned to ROCm 10.0.0
    Then the preview resolves the ROCm 10 pip index
    And every package in the preview is built for ROCm 10.0.0
    And the preview says its torchaudio is the newest build for ROCm 10.0.0 and not hardware-verified

  # The refusal half of therock-next-11: when the pinned build publishes no
  # torchaudio this host can install — here only a wheel for another platform —
  # the install stops and says so instead of taking another build's torchaudio.
  @id:therock-next-12-missing-torchaudio-build-refused
  Scenario: therock-next-12 - A ROCm 10 wheel install refuses when the pinned build publishes no torchaudio
    Given a ROCm 10 pip index fixture whose ROCm 10.0.0 torchaudio is only a wheel this host cannot install
    When the user tries to preview a wheel SDK install for arch gfx1200 pinned to ROCm 10.0.0
    Then the install fails
    And the failure says no torchaudio for ROCm 10.0.0 is published for this platform, without falling back to another

  # The other half of therock-next-11, on the same fixture: newest is not a safe
  # choice among builds sharing a `+rocm` label. The live index publishes
  # `2.11.0`, `2.11.0.2` and `2.11.0.3` all as `+rocm10.1.0`, and `2.11.0` is a
  # CPU-only rebuild. When a torchaudio release is recorded as verified on
  # hardware for the requested line (ROCm 10.1: `2.11.0.2`, the same pin the
  # vLLM ROCm 10.x install uses) and is published, the plan takes it over the
  # newer `2.11.0.3`.
  @id:therock-next-13-verified-torchaudio-over-newer-rebuild
  Scenario: therock-next-13 - A ROCm 10 wheel install takes the hardware-verified torchaudio over a newer rebuild
    Given a ROCm 10 pip index fixture where several torchaudio rebuilds share one ROCm label
    And a uv is already installed
    When the user previews a wheel SDK install for arch gfx1200 pinned to ROCm 10.1.0
    Then the preview installs the hardware-verified torchaudio for ROCm 10.1.0, not the newer rebuild

  # The real-install half of therock-next-11: past the preview, an unverified
  # torchaudio is also announced in the install log, since it is the one stack
  # choice ROCm CLI makes on its own judgement. No GPU or download is needed to
  # see it: with an active default runtime and no consent flag, the install is
  # refused right after the note, before `uv` runs or anything changes on disk.
  @id:therock-next-14-real-install-notes-unverified-torchaudio
  Scenario: therock-next-14 - A real ROCm 10 wheel install notes an unverified torchaudio in its install log
    Given a ROCm 10 pip index fixture that also publishes a newer ROCm 10 build
    And a registered ROCm 10 wheel runtime with a grouped family
    And that runtime is the active default
    When the user installs a wheel SDK for arch gfx1200 pinned to ROCm 10.0.0 without confirming
    Then the install is refused without replacing the active default runtime
    And the install log notes that its torchaudio is the newest build for ROCm 10.0.0 and not hardware-verified

  # The dry-run's `command: uv … && uv …` line is offered as the command the
  # install would run, so it has to be one a shell reads back as that argv. Its
  # quoter used to wrap only on whitespace and a short metacharacter list, so a
  # folder whose name carries a lone `'` rendered bare and left a line no shell
  # can parse. The assertion reads the line back through a POSIX word-splitter
  # and requires the install folder to come back as one word. Reuses the
  # loopback fixture from therock-next-01 (an unpinned install reads its ROCm 10
  # index, so it needs an installed `uv` like therock-next-01), so it resolves
  # without live network or GPU and needs no gate.
  @id:therock-next-15-dry-run-command-keeps-a-quoted-folder-whole
  Scenario: therock-next-15 - The dry-run command line keeps an install folder named with a quote as one argument
    Given a canonical release pip index fixture and a ROCm 10 pip index fixture
    And a uv is already installed
    When the user previews a wheel SDK install for arch gfx1200 into a folder whose name has a quote
    Then the dry-run command line names that folder as a single argument

  # The other half of therock-next-01. An earlier ROCm 7 release is published
  # only in the canonical stream, so pinning one must keep reading it, from that
  # stream and as that stream's layout, instead of failing against the newer
  # layout's index. The newer layout's fixture is served and asserted untouched.
  @id:therock-next-16-earlier-release-pin-reads-the-canonical-stream
  Scenario: therock-next-16 - A release install pinned to an earlier ROCm 7 release reads the canonical stream
    Given a canonical release pip index fixture and a ROCm 10 pip index fixture
    When the user previews a wheel SDK install for arch gfx1200 pinned to an earlier ROCm 7 release
    Then the preview resolves the canonical release pip index
    And the preview reports the canonical multi-arch source layout generation
    And the preview never mentions the ROCm 10 pip index

  # No silent fallback, now for the default: with no pin the newer layout is
  # the only place the newest build is, and a group label names no exact arch
  # for it. The install must refuse and say which flag to pass, not drop back to
  # the canonical stream and hand over an old build. The refusal also names
  # pinning an earlier release as the way to still install a group family, so
  # that command is run here: the message and the behaviour it promises are
  # asserted together.
  @id:therock-next-17-unpinned-group-family-refused-and-the-named-pin-works
  Scenario: therock-next-17 - An unpinned release install refuses a group family, and the earlier-release pin it names installs
    Given a canonical release pip index fixture and a ROCm 10 pip index fixture
    When the user tries to preview a wheel SDK install for family gfx120X-all with no version pin
    Then the install fails
    And the failure asks for an exact GPU arch and names --family gfx1200
    When the user previews a wheel SDK install for family gfx120X-all pinned to an earlier ROCm 7 release
    Then the preview resolves the canonical release pip index

  # Linux-only for the same reason as therock-next-04. The unpinned tarball
  # install reads the newer layout's catalog and still skips the `-tests-`
  # sibling that carries a later mtime than the real archive.
  @id:therock-next-18-unpinned-tarball-reads-the-newest-catalog @requires-os:linux
  Scenario: therock-next-18 - An unpinned release tarball install reads the newest layout's catalog
    Given a canonical release tarball fixture and a ROCm 10 tarball fixture with a tests sibling
    When the user previews a tarball SDK install for arch gfx1200 with no version pin
    Then the preview resolves the ROCm 10 tarball catalog
    And the preview reports the next source layout generation
    And the preview selects the real tarball artifact
    And the preview does not select the tests artifact

  # A tarball catalog can only be pinned in the newer layout, so a pin into the
  # earlier release stream is refused. The refusal points at the wheel install,
  # and the wheel install it points at is run, so the advice is shown to work.
  @id:therock-next-19-earlier-release-tarball-pin-points-to-wheel @requires-os:linux
  Scenario: therock-next-19 - A release tarball install pinned to an earlier ROCm 7 release points to a wheel install
    Given a canonical release pip index fixture and a ROCm 10 pip index fixture
    When the user tries to preview a tarball SDK install for arch gfx1200 pinned to an earlier ROCm 7 release
    Then the install fails
    And the failure points to a wheel install for that earlier release

  # The nightly channel moved the same way on 2026-08-23. With no pin the
  # install reads the newer layout's nightly index; that index publishes a
  # `rocm` build before its PyTorch wheels land, so the install takes the newest
  # build that has the whole stack and says it skipped the newer one, rather
  # than failing on a build it cannot install.
  @id:therock-next-20-unpinned-nightly-reads-the-newest-layout-and-skips-an-unbuilt-build
  Scenario: therock-next-20 - An unpinned nightly install reads the newest layout's index at its newest installable build
    Given an earlier nightly pip index fixture and a newest-layout nightly pip index fixture whose newest build has no PyTorch wheels yet
    And a uv is already installed
    When the user previews a nightly wheel SDK install for arch gfx1200 with no version pin
    Then the preview resolves the newest layout's nightly pip index at its newest installable build
    And the preview warns that the newest nightly build has no PyTorch wheels and installs the build before it

  # The earlier nightly stream's last build is 2026-08-22, so a pin to that
  # date can only be read from it, as that stream's layout.
  @id:therock-next-21-earlier-nightly-pin-reads-the-canonical-stream
  Scenario: therock-next-21 - A nightly install pinned to the earlier stream's last build date reads the canonical stream
    Given an earlier nightly pip index fixture and a newest-layout nightly pip index fixture whose newest build has no PyTorch wheels yet
    When the user previews a nightly wheel SDK install for arch gfx1200 pinned to the earlier stream's last build date
    Then the preview resolves the earlier stream's nightly pip index

  # The other side of therock-next-21's boundary: a pin to a date the newer
  # layout publishes reads that layout, though both fixtures are served.
  @id:therock-next-22-newer-nightly-pin-reads-the-newest-layout
  Scenario: therock-next-22 - A nightly install pinned to a build date from the newest layout reads that layout
    Given an earlier nightly pip index fixture and a newest-layout nightly pip index fixture whose newest build has no PyTorch wheels yet
    And a uv is already installed
    When the user previews a nightly wheel SDK install for arch gfx1200 pinned to a build date from the newest layout
    Then the preview resolves the newest layout's nightly pip index for that build date

  # The tarball half of therock-next-19, on the nightly channel, where the
  # boundary is a date rather than a major. A tarball pin reads only the newest
  # layout's catalog, so a pin to the earlier stream's last build date is
  # refused, naming the date the newer layout begins and pointing to the wheel
  # install, which is then run and resolves that build.
  @id:therock-next-23-earlier-nightly-tarball-pin-points-to-wheel @requires-os:linux
  Scenario: therock-next-23 - A nightly tarball install pinned to the earlier stream's last build date points to a wheel install
    Given an earlier nightly pip index fixture and a newest-layout nightly pip index fixture whose newest build has no PyTorch wheels yet
    When the user tries to preview a nightly tarball SDK install for arch gfx1200 pinned to the earlier stream's last build date
    Then the install fails
    And the failure names the first build date of the newer layout and points to a wheel install that resolves that earlier nightly
