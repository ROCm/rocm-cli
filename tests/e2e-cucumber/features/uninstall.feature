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
  # Linux-only: they plant directory symlinks. Removing a directory symlink on
  # Windows takes a different call, which this change does not touch.

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
