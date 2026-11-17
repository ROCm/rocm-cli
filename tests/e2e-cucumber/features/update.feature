Feature: Update report

  # `rocm update` (with no arguments) prints an update report: which managed
  # runtimes have updates, plus the status of each update feed (CLI, engines,
  # model recipes, runtimes). The walkthrough verified it correctly distinguishes
  # published feeds from not-configured ones, but nothing pinned it. Run with no
  # managed runtimes so the report needs no network — mock lane, every PR.

  @id:update-report-distinguishes-feed-status
  Scenario: update-01 - The update report distinguishes configured from not-configured feeds
    Given a machine with no managed runtimes
    When the user checks for updates
    Then the report shows there are no managed runtimes to update
    And it reports each update feed's status, marking unpublished feeds as not configured

  @id:update-json-reports-empty-runtimes
  Scenario: update-02 - The machine-readable update check reports an empty runtimes array
    Given a machine with no managed runtimes
    When the user checks for updates as machine-readable JSON
    Then the machine-readable check reports no runtimes to update

  @id:update-json-accepts-timeout-flag
  Scenario: update-03 - The machine-readable update check accepts a --timeout-secs flag
    Given a machine with no managed runtimes
    When the user checks for updates as machine-readable JSON with a 5 second timeout
    Then the machine-readable check reports no runtimes to update

  # `--dry-run` used to `requires = "apply"` in clap, so `rocm update --dry-run`
  # alone failed with a bare usage error instead of previewing. It no longer
  # requires `--apply`, so this asserts the command reaches real business logic
  # (the "no managed runtimes" bail, which reads nothing like a clap usage error)
  # rather than being rejected before `rocm` even looks at the registry.
  @id:update-dry-run-reaches-preview-path-without-apply
  Scenario: update-04 - Previewing an update with --dry-run does not require --apply
    Given a machine with no managed runtimes
    When the user previews an update
    Then the CLI refuses because no managed runtimes are registered

  # `--runtime`/`--activate` are apply-only flags: without `--apply` or
  # `--dry-run` alongside them, the old code rejected them with a bare clap
  # usage error (both were declared `requires = "apply"`) instead of a message
  # naming the actual constraint. This pins the intentional refusal message.
  @id:update-runtime-or-activate-without-apply-or-dry-run-is-refused
  Scenario: update-05 - --runtime or --activate without --apply or --dry-run is refused
    Given a machine with no managed runtimes
    When the user requests updating a specific runtime without --apply or --dry-run
    Then the CLI refuses because --apply or --dry-run is required with --runtime or --activate

  # --dry-run and --json are mutually exclusive: --json emits a single line of
  # machine-readable JSON, and --dry-run would print human-readable preview text
  # on top of it, corrupting the JSON contract. Pins the clap conflict instead of
  # one flag silently winning.
  @id:update-dry-run-conflicts-with-json
  Scenario: update-06 - --dry-run and --json cannot be combined
    Given a machine with no managed runtimes
    When the user checks for updates as JSON with --dry-run
    Then the CLI refuses because --dry-run and --json cannot be combined

  # Which runtime `rocm update` offers to install is decided by comparing the
  # installed version against the catalog's. That comparison used to fall back
  # to a plain text compare whenever either side was not exactly
  # MAJOR.MINOR.PATCH, so a two-component catalogue version inverted the verdict
  # in BOTH directions: a runtime a release ahead of the catalog was offered the
  # older build as an "update" (a silent downgrade on `--apply`), and a runtime
  # a release behind was called ahead of the catalog and never offered the newer
  # one. Both runtimes are checked in a single report so one run pins both
  # directions. Nightly channel on purpose: the release channel drops versions
  # it cannot parse before they ever reach the comparison, which hid this.
  # Tarball format keeps it metadata-only over a loopback catalog — no Python,
  # no GPU, no network. Mock lane, every PR.
  @id:update-report-compares-versions-numerically-not-as-text
  Scenario: update-07 - The update report compares catalog versions numerically, not as text
    Given a nightly tarball catalog older than one registered runtime and newer than another
    When the user checks for updates against that catalog
    Then the report offers no update for the runtime that is ahead of the catalog
    And the report offers an update for the runtime that is behind the catalog

  # The other half of the same comparison: a catalog that publishes nothing
  # newer must not be read as an update. Two shapes it has to get right. The
  # catalog may spell the installed version differently — an index serves the
  # normalised `7.0.0rc1` while an older manifest recorded `7.0.0-rc1`, which
  # PEP 440 calls one version — and an offer there re-downloads a runtime the
  # machine already has. And a four-component release (`7.10.0.71000`, the
  # shape ROCm's own packages are named with) is newer than a catalog's
  # `7.9.0`, so `--apply` must not install 7.9.0 over it; as text it sorts
  # below, which is the comparison this used to fall back to. Same loopback
  # nightly tarball catalog as update-07: metadata-only, no GPU, no network.
  # Mock lane.
  @id:update-report-offers-nothing-when-the-catalog-is-not-newer
  Scenario: update-08 - The update report offers nothing when the catalog has no newer version
    Given a nightly tarball catalog that has no newer version for either registered runtime
    When the user checks for updates against that catalog
    Then the report calls the runtime the catalog spells differently up to date
    And the report offers no update for the four-component runtime that is ahead of the catalog

  # `rocm update` is a check: it reads the installed runtimes and the index and
  # changes nothing. To tell which wheels fit a wheel runtime it needs the
  # interpreter's tag, and on a host with no usable Python it used to go and get
  # one: it downloaded a managed `uv`, had `uv` install Python and saved that
  # Python's path in the configuration, all before printing the report. The same
  # resolution runs for `rocm update --json` (the dashboard's check) and for the
  # startup check that runs ahead of other commands, so a read-only look at the
  # runtimes could prepare a Python. Now the check works the tag out from the
  # interpreter the install would prepare, says it would prepare one, and
  # prepares nothing.
  #
  # The report's note says the check prepared nothing, so the Then checks that
  # note and the state together, as runtime-23 does for the install preview: the
  # recording `uv` is reached only by going on to prepare a Python, so its log not
  # existing means no preparation was attempted, and the managed tools folder and
  # the configuration must be absent too. The first Then shows the check still got
  # as far as a verdict, so the absences are not just a command that stopped
  # early. `--json` and the startup check are pinned by the unit tests in
  # `apps/rocm/src/therock/update.rs`.
  #
  # Linux only, for the same empty-`PATH` host as runtime-23. The canonical
  # release index is served from a loopback fixture, so no network is needed.
  @id:update-check-prepares-no-python @requires-os:linux
  Scenario: update-09 - Checking for updates on a host with no usable Python prepares no Python
    Given a canonical release pip index fixture
    And a host with no usable Python and a recording uv
    And a registered wheel runtime older than the index with no recorded Python
    When the user checks for updates against the pip index
    Then the report offers the newer runtime and says it would prepare a managed Python
    And the update check prepared no Python, as it says

  # The sibling for `--dry-run`, which promises to show what `--apply` would do
  # and change nothing. Its plan step resolved the same way and prepared the
  # Python before the preview was printed; the apply itself still prepares what it
  # needs, which the unit tests keep apart from the check.
  #
  # Linux only, for the same empty-`PATH` host as update-09.
  @id:update-dry-run-prepares-no-python @requires-os:linux
  Scenario: update-10 - Previewing an update on a host with no usable Python prepares no Python
    Given a canonical release pip index fixture
    And a host with no usable Python and a recording uv
    And a registered wheel runtime older than the index with no recorded Python
    When the user previews applying the update to that wheel runtime
    Then the preview says a real install would prepare a managed Python
    And the preview prepared no Python, as it says

  # An update installs from the layout its runtime was recorded in, and names no
  # pin. A runtime of the earlier release stream recorded with only a family
  # label (`gfx120X-all`, which that stream serves) must therefore still update:
  # the newest layout needs an exact arch, but the runtime is not being moved
  # there. Gating the update on a fresh install's routing would refuse it with
  # "requires an exact GPU arch", though its own resolution never needed one.
  # The report is the update's plan, so it names the earlier stream it reads.
  #
  # Linux only, for the same empty-`PATH` host as update-09.
  @id:update-apply-keeps-a-grouped-family-runtime-on-its-recorded-stream @requires-os:linux
  Scenario: update-11 - Previewing an update of an earlier-stream runtime recorded with a family label keeps it on that stream
    Given a canonical release pip index fixture
    And a host with no usable Python and a recording uv
    And a registered wheel runtime of the earlier release stream with a grouped family
    When the user previews applying the update to that wheel runtime
    Then the preview resolves the canonical release pip index
