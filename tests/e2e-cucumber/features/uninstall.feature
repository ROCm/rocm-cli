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
