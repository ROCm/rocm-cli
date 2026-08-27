Feature: The ROCm Doctor skill's instructions match the CLI they drive

  # `skills/rocm-doctor/` is a byte-verbatim mirror of the skill published in
  # amd/skills. The skill owns no probe, no catalog and no fixes — all of that
  # ships inside the `rocm` binary — so what it DOES own is a contract: which
  # fix-ids exist, which the CLI applies itself, which machines each one is for,
  # and what a diagnosis carries.
  #
  # The centre of this feature is the catalog diff (scenarios 1-3): the table in
  # `reference.md` is parsed and compared to what `rocm fix` really reports, so a
  # renamed fix-id, a re-scoped OS, a flipped auto-flag, or a 16th failure mode
  # cannot merge here and silently break the published skill. Nothing else tests
  # that seam — `diagnose.feature` deliberately asserts only the SHAPE of a
  # diagnosis so it stays host-independent.
  #
  # The catalog is authoritative in `crates/rocm-core`. When one of these fails,
  # the CLI is right and `skills/rocm-doctor/reference.md` is what changes.
  #
  # Scope is deliberately narrow: EVERY scenario here compares the CLI against
  # the published document, and nothing else belongs. Claims proven elsewhere
  # are not restated. `diagnose.feature` already owns the CLI's own behaviour —
  # that a fix meant for another OS is declined without changing anything
  # (`@id:fix-inapplicable-here-is-declined-not-attempted`), that the catalog
  # listing is complete against an in-test list
  # (`@id:fix-catalog-is-complete`), and that a report says plainly whether a
  # cause was established (`@id:diagnose-json-states-when-nothing-matched`).
  # The exit-code and confirm-before-mutating contract is unit tested in
  # `crates/rocm-core/src/fix.rs`, where a declined consent or a failed write
  # can be driven directly.
  #
  # The distinction that earns this feature its place: those tests compare the
  # CLI against expectations written in the test suite, so they catch the CLI
  # drifting. Only these scenarios catch the DOCUMENT drifting, because only
  # here is `reference.md` itself the expected value.
  #
  # Every scenario is a query: no GPU, no serve, no download, no mutation — so
  # they all run on the blocking mock lane and need no capability tags.

  @id:skill-catalog-ids-match-cli
  Scenario: 1 - Every remediation the skill documents is one the CLI still offers
    Given the ROCm Doctor skill as it is published
    When an agent asks the CLI which remediations it knows
    Then the skill and the CLI describe the same set of remediations

  @id:skill-auto-fix-set-matches-cli
  Scenario: 2 - The skill agrees with the CLI about which remediations the CLI will run itself
    Given the ROCm Doctor skill as it is published
    When an agent asks the CLI which remediations it knows
    Then the skill and the CLI agree on which ones the CLI applies without help

  @id:skill-os-scope-matches-cli
  Scenario: 3 - The skill agrees with the CLI about which machines each remediation applies to
    Given the ROCm Doctor skill as it is published
    When an agent asks the CLI which remediations it knows
    Then the skill and the CLI agree on which machines each remediation is for

  @id:skill-diagnosis-matches-what-the-skill-documents
  Scenario: 4 - A diagnosis carries everything the skill tells an agent to read
    Given the ROCm Doctor skill as it is published
    And a user who reports a recognised ROCm failure
    When an agent asks the CLI to diagnose that report for tooling
    Then the diagnosis carries every field the skill names
    And its confidence thresholds are the ones the skill reasons about

  @id:skill-examine-verdicts-are-documented
  Scenario: 5 - Inspecting the machine returns a verdict the skill documents
    Given the ROCm Doctor skill as it is published
    When an agent inspects the machine for tooling
    Then the inspection succeeds whatever it finds
    And its verdict is one the skill documents

  # The skill's standing rule is "never invent a fix — if nothing matched, route
  # the user upstream". That rule is only followable if the address the CLI hands
  # over is the one the skill's own routing table gives, so this is the last
  # claim in the document that the binary can be held to.
  @id:skill-escalation-target-matches-cli
  Scenario: 6 - A report the catalog cannot explain is routed where the skill says
    Given the ROCm Doctor skill as it is published
    And a user who reports a failure the catalog does not cover
    When an agent asks the CLI to diagnose that report for tooling
    Then the CLI routes the report to a tracker the skill documents
