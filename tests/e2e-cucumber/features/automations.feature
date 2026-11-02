Feature: Automation watchers

  # `rocm automations enable/disable <watcher> --mode <mode>` toggles a built-in
  # automation watcher in the CLI config and prints a confirmation; bare `rocm
  # automations` lists the most recent review requests (any status — only the
  # per-entry approve/reject controls are gated on "pending"). This feature
  # pins the enable/disable/mode-confirmation slice, the proposal-history
  # listing's unit formatting, the bare listing's tolerance of a torn
  # audit-log tail, and how the audit log and proposal history are kept
  # bounded; the watcher/daemon trigger behaviour itself is covered
  # elsewhere. Config-only (no GPU, no network), so it runs on the mock lane
  # every PR.

  @id:automations-enable-confirms-mode
  Scenario: automations-01 - Enabling a watcher confirms its mode
    Given a fresh CLI configuration
    When the user enables an automation watcher in observe mode
    Then the CLI confirms the watcher is enabled in observe mode
    When the user re-enables the same watcher in propose mode
    Then the CLI confirms the watcher is enabled in propose mode

  @id:automations-disable-confirmed
  Scenario: automations-02 - Disabling a watcher is confirmed
    Given an enabled automation watcher
    When the user disables the watcher
    Then the CLI confirms the watcher is disabled

  @id:automations-enable-unknown-refused
  Scenario: automations-03 - Enabling an unknown watcher is refused
    Given a fresh CLI configuration
    When the user tries to enable a watcher that does not exist
    Then the CLI refuses and names it as unknown

  # Autostart daemon singleton (EAI-7194): the first `automations enable` (or
  # `serve`) that finds no running daemon spawns one, but the child does not
  # publish its `running` runtime state until well after `spawn()` (clap parse,
  # runtime build, config load, banner flush). A second invocation that lands in
  # that spawn→publish window re-reads "not running" and — before the fix — spawned
  # a duplicate daemon that orphaned the first. The autostart claim (child PID +
  # spawn time, written under the lock before it drops) closes the window: a caller
  # that sees a live, recent claim defers instead of spawning. This plants an
  # in-flight claim (no published runtime state yet) and asserts the enable defers
  # rather than launching a second daemon. Config-only — no GPU or network — so it
  # runs on the mock lane every PR, which is why the race half of the PR can be
  # covered without hardware.
  @id:automations-autostart-defers-during-spawn-window @requires-os:linux
  Scenario: automations-04 - Enabling during an in-flight daemon spawn starts no second daemon
    Given a daemon spawn is already in flight
    When the user enables an automation watcher in observe mode
    Then the CLI does not start a second background daemon

  # format_bytes_for_user (the only formatter behind this line) mislabeled
  # binary (base-1024) sizes as KB/MB/GB instead of KiB/MiB/GiB (#190).
  @id:automations-proposal-size-uses-binary-units
  Scenario: automations-05 - A pending artifact-download proposal shows its approved cap in binary units
    Given a pending artifact-download proposal approved for a size just under 1 MiB
    When the user lists automations
    Then the CLI reports the approved size in binary units

  # Readers of the audit/events/proposals stores take no lock, so one can
  # catch a write mid-append; the cut-off last line that leaves is skipped
  # instead of failing the whole read.
  # `render_automations_text` already swallows any read error from these
  # stores via `.unwrap_or_default()`, so `rocm automations` never exits
  # non-zero either way — what this scenario actually pins is that the clean
  # record ahead of the torn one still shows up, rather than the torn line's
  # read error wiping out the whole audit section the way it would without
  # this skip.
  # `render_automations_text` is the production reader of the audit store
  # specifically (the one this scenario seeds). The two places a person
  # reads its raw text directly are the `rocm automations` CLI subcommand
  # itself, and the TUI's automations-manager screen, which shells out to
  # `rocm automations list` as a subprocess — so that's where the audit
  # store's tolerance is directly user-observable. It's also reachable
  # through other paths that don't show the raw text the same way —
  # `rocm chat`'s and the TUI's slash-command `rocm_command` tool in-process
  # (the model sees it, or the TUI shrinks it to field counts first), and
  # rocmd's own MCP `rocm_command` tool via a further `rocm automations list`
  # subprocess — this isn't meant as an exhaustive map of every caller, just
  # context for why the CLI subcommand is the right level to pin this at.
  @id:automations-tolerates-torn-audit-tail
  Scenario: automations-06 - Listing automations tolerates a torn audit-log tail
    Given a fresh CLI configuration
    And the audit event log ends in a torn, unterminated line
    When the user lists automations
    Then the CLI still lists the record before the torn tail

  # The automation event log and the audit log are size-capped (ROCMAI-729):
  # the write that finds the live log at its cap first renames it aside to
  # `events.jsonl.1`, keeping a fixed number of such generations. This seeds the
  # audit log at the cap, makes one change the CLI audits, and checks both the
  # files and that the automations listing still shows activity from before the
  # rollover, which it can only read from the rotated generation.
  @id:automations-audit-log-rotates-at-cap
  Scenario: automations-07 - A full audit log is set aside and its recent history stays visible
    Given an audit log that has reached its size cap
    When the user selects a permissions mode
    Then the full audit log is kept aside as its first rotated generation
    And the permissions change starts a fresh audit log
    When the user lists automations
    Then the recent background activity still includes entries from before the rotation

  # The proposal history is compacted, not rotated (ROCMAI-729): a pending
  # proposal must stay findable by id until it is reviewed, so reviewing one
  # rewrites the history keeping every pending proposal but only the most
  # recently reviewed ones. Approves through `rocm mcp-call proposal_action`, the
  # hidden entry point the dashboard and chat review controls use.
  @id:automations-review-compaction-keeps-pending
  Scenario: automations-08 - Approving a proposal trims old review history but keeps older pending proposals
    Given a pending proposal queued before a long history of reviewed proposals
    When the user approves the newest pending proposal
    Then the CLI confirms the approval
    And only the most recently reviewed proposals are kept
    When the user looks up the older pending proposal
    Then the CLI shows it still pending

  # automations-07 only exercises a single rollover: a live file at the cap
  # gets a `.1`. The shift-and-drop-oldest chain beyond that — `.1` to `.2`,
  # `.2` to `.3`, `.3` to `.4`, and the previous `.4` dropped — is otherwise
  # pinned only by unit tests in `automation.rs`. This seeds a full chain of
  # four already-rotated generations plus a live file at the cap, forces one
  # more rotation, and checks the whole chain shifted down by one and the
  # oldest generation's content is gone, not just that `.1` exists.
  @id:automations-rotation-chain-shifts-and-drops-oldest
  Scenario: automations-09 - A further rotation shifts the whole chain and drops the oldest generation
    Given an audit log already holding a full chain of four rotated generations
    When the user selects a permissions mode
    Then each rotated generation holds what the previous one held
    And the oldest previous generation is gone
