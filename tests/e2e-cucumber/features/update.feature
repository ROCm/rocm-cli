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
