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
  # Scope is deliberately narrow. Claims already proven elsewhere are NOT
  # restated here: the exit-code and confirm-before-mutating contract is unit
  # tested in `crates/rocm-core/src/fix.rs` (where a declined consent or a failed
  # write can be driven directly), the `status` enum in `examine.rs`, and the
  # WSL2 out-of-scope rule in `diagnose.rs`. This feature covers what only a real
  # binary can show: that the documented catalog and the shipped one agree, and
  # that the JSON an agent reads is actually plumbed through.
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

  @id:skill-diagnosis-shape-is-readable
  Scenario: 4 - A diagnosis hands the agent everything the skill tells it to read
    Given a user who reports a recognised ROCm failure
    When an agent asks the CLI to diagnose that report for tooling
    Then the diagnosis carries the confidence thresholds the skill reasons about
    And every cause it offers carries a title, a confidence, its evidence, and a plan

  @id:skill-examine-verdict-is-known
  Scenario: 5 - Inspecting the machine returns a verdict the skill knows how to read
    When an agent inspects the machine for tooling
    Then the inspection succeeds whatever it finds
    And its verdict is one the skill accounts for

  @id:skill-wrong-os-fix-declined
  Scenario: 6 - A remediation meant for a different machine is declined without changing anything
    Given an agent that picked a remediation meant for a different kind of machine
    When the agent asks the CLI to apply that remediation
    Then the CLI declines because it does not apply here
    And no managed state is written
