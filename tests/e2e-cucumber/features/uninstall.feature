Feature: Uninstall

  # `rocm uninstall` removes the config, data and cache folders ROCm CLI uses.
  # `install_lifecycle.feature` covers a full uninstall of a released install,
  # but those scenarios are @lifecycle (skipped by default, OS-mutating). These
  # cover what the plan does with the folders it is pointed at, so they run on
  # the mock lane every PR.
  #
  # Every scenario passes --keep-binaries (the binary under test must survive)
  # and points HOME, the uv cache and the Hugging Face cache at the scenario's
  # own folder, so an uninstall can never reach the runner's shared caches.
  #
  # The link scenarios are Linux-only, and the behaviour they pin is only
  # claimed for Linux: on Windows a directory symlink or junction needs
  # `remove_dir` rather than `remove_file`, which `rocm uninstall` does not do
  # yet and nothing here verifies.

  @id:uninstall-trailing-slash-link-unlinks-only @requires-os:linux
  Scenario: uninstall-01 - A linked cache folder written with a trailing slash is unlinked, not emptied
    Given the cache folder is a link to another folder holding the user's files
    When the user uninstalls only the cache, writing its folder with a trailing slash
    Then the uninstall succeeds
    And the uninstall reports the cache link as removed
    And the cache link is gone
    And the folder the link pointed to still holds the user's files

  @id:uninstall-dangling-link-is-removed @requires-os:linux
  Scenario: uninstall-02 - A cache folder that is a broken link is listed and really removed
    Given the cache folder is a link to a folder that no longer exists
    When the user uninstalls only the cache
    Then the uninstall succeeds
    And the uninstall reports the cache link as removed
    And the cache link is gone

  @id:uninstall-unreadable-root-stops-before-removing @requires-os:linux
  Scenario: uninstall-03 - A cache folder that cannot be checked stops the uninstall before anything is removed
    Given the config folder holds the user's files
    And the cache folder is reached through a link that points at itself
    When the user uninstalls the config and the cache
    Then the uninstall fails
    And the error names the cache folder it could not check
    And the config folder still holds the user's files

  # `rocm uninstall --help` promises that, without --yes and outside a terminal,
  # the command prints the plan and refuses rather than asking or assuming an
  # answer. The harness runs the binary with no terminal, so this is that case.
  @id:uninstall-refused-without-yes-outside-terminal @requires-os:linux
  Scenario: uninstall-04 - Uninstalling without confirming outside a terminal is refused and removes nothing
    Given the config folder holds the user's files
    When the user uninstalls the config without confirming
    Then the uninstall is refused and the user is told to confirm explicitly
    And the config folder still holds the user's files

  # An install isolated by ROCM_CLI_CONFIG_DIR and ROCM_CLI_DATA_DIR alone,
  # with no ROCM_CLI_CACHE_DIR, keeps its cache inside its data folder. It once
  # fell back to the default `~/.rocm/cache`, so `rocm storage` reported the
  # default install's cache and `rocm uninstall` deleted it.
  @id:uninstall-moved-data-folder-keeps-its-own-cache
  Scenario: uninstall-05 - With only the config and data folders set, the cache folder is the data folder's own
    Given a default install's cache folder in the user's home holds the user's files
    And the data folder's own cache folder holds a cache file
    When the user asks what ROCm CLI keeps on disk with only the config and data folders set
    Then the storage report names the cache folder inside the data folder
    When the user uninstalls with only the config and data folders set
    Then the uninstall succeeds
    And the uninstall does not name the default install's cache folder
    And the data folder's own cache folder is gone
    And the default install's cache folder still holds the user's files

  # `rocm uninstall --dry-run` shows what would be removed without changing
  # anything. It once started the file logger first, which created the data
  # folder's `logs` folder, and then listed that folder as the one thing to
  # remove, so a machine ROCm CLI had never touched looked installed. The
  # promise is two-sided: the plan says there is nothing to remove, and the
  # folders it would have removed are still absent afterwards.
  @id:uninstall-dry-run-on-an-untouched-machine-creates-nothing
  Scenario: uninstall-06 - A dry run on a machine ROCm CLI never touched reports nothing to remove and creates nothing
    Given ROCm CLI has never created its folders in the user's home
    When the user dry-runs an uninstall
    Then the uninstall succeeds
    And the uninstall reports that nothing would be removed
    And ROCm CLI has still not created its folders

  # A dry run run by a person at a terminal ends by saying to run it again
  # without `--dry-run`. It once said "Choose Review uninstall", a control the
  # command line does not have. The promise is the line and the state together:
  # the dry run removes nothing, and running the same command again, as the
  # line says and answering its prompt, removes what was listed.
  @id:uninstall-dry-run-at-a-terminal-says-to-run-it-again-to-remove @requires-os:linux
  Scenario: uninstall-07 - At a terminal, a dry run removes nothing and running it again without --dry-run removes what it listed
    Given the config folder holds the user's files
    When the user dry-runs an uninstall of only the config at a terminal
    Then the uninstall succeeds
    And the dry run says to run it again without --dry-run
    And the config folder still holds the user's files
    When the user runs it again without --dry-run at a terminal and answers yes
    Then the uninstall succeeds
    And the config folder is gone

  # A run whose stdin or stdout is not a terminal (a script, the dashboard's
  # `/uninstall`, the daemon's `rocm_command` tool) has no one to tell to run it
  # again: the real step belongs to whatever drives it. It lists what it would
  # remove, offers no step, and removes nothing.
  @id:uninstall-dry-run-outside-a-terminal-offers-no-next-step
  Scenario: uninstall-08 - Outside a terminal, a dry run lists what it would remove and offers no step to remove it
    Given the config folder holds the user's files
    When the user dry-runs an uninstall of only the config
    Then the uninstall succeeds
    And the dry run lists the config folder
    And the dry run offers no step to remove anything
    And the config folder still holds the user's files

  # With nothing to remove there is nothing to approve, so even a person at a
  # terminal is offered no step. A config folder that is kept and still holds
  # the user's files shows the "nothing to remove" is true.
  @id:uninstall-dry-run-with-nothing-to-remove-offers-no-next-step @requires-os:linux
  Scenario: uninstall-09 - At a terminal, a dry run with nothing to remove offers no step to remove it
    Given the config folder holds the user's files
    When the user dry-runs an uninstall that keeps everything at a terminal
    Then the uninstall succeeds
    And the uninstall reports that nothing would be removed
    And the dry run offers no step to remove anything
    And the config folder still holds the user's files

  # `--force` is a natural guess for skipping the confirmation, and the nearest
  # real flag by name is `--force-dev-binaries`, which removes the binaries of a
  # development build. `--force` is refused with the flag to use instead, a near
  # miss of it never suggests the destructive flag, and nothing is removed.
  @id:uninstall-mistyped-force-is-not-answered-with-a-destructive-flag
  Scenario: uninstall-10 - A mistyped --force is refused without pointing at the flag that removes development binaries
    Given the config folder holds the user's files
    When the user uninstalls the config with --force
    Then the uninstall is refused and told to use --yes instead
    And the config folder still holds the user's files
    When the user uninstalls the config with a near miss of --force
    Then the error does not point at the flag that removes development binaries
    And the config folder still holds the user's files

  @id:uninstall-help-shows-a-dry-run-before-the-real-run
  Scenario: uninstall-11 - The uninstall help shows a dry run before the real run
    When the user asks for the uninstall help
    Then the help shows a dry run before the real run

  # The assistant's uninstall preview (the `rocm_command` tool of the chat
  # assistant and the dashboard, called here without a model through the hidden
  # `rocm mcp-call`) renders the same plan as a dry run, but it has no flag to
  # offer, so it says that removal needs approval. It is the one run here that
  # cannot pass --keep-binaries: the preview takes no flags and removes nothing,
  # and it is made without --allow-mutation, so a call that would remove
  # something is refused.
  @id:uninstall-assistant-preview-says-removal-needs-approval
  Scenario: uninstall-12 - The assistant's uninstall preview lists what it would remove, says removal needs approval, and removes nothing
    Given the config folder holds the user's files
    When the assistant previews an uninstall
    Then the preview succeeds
    And the preview lists the config folder
    And the preview says removal needs approval
    And the config folder still holds the user's files

  # The `[y/N]` prompt is shared with `rocm fix` and the SDK-overwrite prompt;
  # `rocm uninstall` is the one that is cheap to isolate, so the shared helper's
  # behaviour is pinned here. A prompt is only reached on a real terminal (without
  # one the command is refused, see uninstall-04), so these run under a
  # pseudo-terminal. An answer that is neither yes nor no used to be declined with
  # no sign it was not understood, and ending the input (Ctrl-D) left the
  # cancellation on the prompt line because no newline had been echoed.
  @id:uninstall-unrecognized-answer-is-declined-and-said-so @requires-os:linux
  Scenario: uninstall-13 - An answer that is neither yes nor no is declined, and the user is told so
    Given the config folder holds the user's files
    When the user is asked to confirm an uninstall on a real terminal and types an answer that is neither yes nor no
    Then the user is told the answer was not understood and is treated as no
    And the uninstall is reported as cancelled
    And the config folder still holds the user's files

  @id:uninstall-end-of-input-cancels-on-a-fresh-line @requires-os:linux
  Scenario: uninstall-14 - Ending the input at the confirmation prompt cancels on a fresh line
    Given the config folder holds the user's files
    When the user is asked to confirm an uninstall on a real terminal and ends the input instead of answering
    Then the cancellation is reported on its own line, not on the prompt line
    And the config folder still holds the user's files

  # The other side of uninstall-13: a deliberate no is not an error to explain.
  @id:uninstall-plain-no-is-declined-without-complaint @requires-os:linux
  Scenario: uninstall-15 - Answering no is declined without any complaint
    Given the config folder holds the user's files
    When the user is asked to confirm an uninstall on a real terminal and types no
    Then the uninstall is reported as cancelled
    And the user is not told the answer was misunderstood
    And the config folder still holds the user's files

  # A refusal that knows the command which completes the uninstall ends with it
  # as a `Try:` line. The promise is the line and the state together: running
  # the line as printed removes what the refused run listed, and keeps what it
  # kept. Linux-only like uninstall-04, whose refusal it starts from.
  @id:uninstall-refusal-names-the-command-that-completes-it @requires-os:linux
  Scenario: uninstall-16 - Running the command a refused uninstall suggests removes what was listed and keeps what was kept
    Given the config folder holds the user's files
    And the data folder holds the user's files
    When the user uninstalls the config without confirming
    Then the uninstall is refused and the user is told to confirm explicitly
    And the refusal ends by suggesting the same uninstall, confirmed
    When the user runs the command the refusal suggests
    Then the uninstall succeeds
    And the config folder is gone
    And the data folder still holds the user's files

  # The other side of uninstall-10: `--force` is refused, and the line it ends
  # with is the command the user meant, `--yes` in place of `--force`.
  @id:uninstall-force-refusal-names-the-command-that-completes-it
  Scenario: uninstall-17 - Running the command a refused --force suggests removes what was listed and keeps what was kept
    Given the config folder holds the user's files
    And the data folder holds the user's files
    When the user uninstalls the config with --force
    Then the uninstall is refused and told to use --yes instead
    And the refusal ends by suggesting the same uninstall, confirmed
    When the user runs the command the refusal suggests
    Then the uninstall succeeds
    And the config folder is gone
    And the data folder still holds the user's files
