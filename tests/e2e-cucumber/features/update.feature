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
  # `--dry-run` alongside them, the old code silently fell through to the plain
  # report and ignored both. This pins the refusal instead of the silent no-op.
  @id:update-runtime-or-activate-without-apply-or-dry-run-is-refused
  Scenario: update-05 - --runtime or --activate without --apply or --dry-run is refused
    Given a machine with no managed runtimes
    When the user requests updating a specific runtime without --apply or --dry-run
    Then the CLI refuses because --apply or --dry-run is required with --runtime or --activate
