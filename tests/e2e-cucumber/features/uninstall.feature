Feature: Uninstall

  # `rocm uninstall` removes the config, data and cache folders ROCm CLI uses.
  # `install_lifecycle.feature` covers a full uninstall of a released install,
  # but those scenarios are @lifecycle (skipped by default, OS-mutating). These
  # cover what the plan does with the folders it is pointed at, so they run on
  # the mock lane every PR.
  #
  # Every scenario passes --keep-binaries (the binary under test must survive)
  # and points HOME, the cache folder, the uv cache and the Hugging Face cache
  # at the scenario's own folder, so an uninstall can never reach the runner's
  # shared caches. The one setting that names a folder outside it, a data
  # folder of `/`, is only ever previewed with --dry-run.
  #
  # uninstall-01 and -02 are Linux-only, and the behaviour they pin is only
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

  # Linux-only: on Windows `/` names the current drive's root and renders as
  # `\`, so this line would read differently. The Windows drive-root refusal is
  # covered only by rocm-core's `a_windows_drive_root_is_the_filesystem_root`
  # unit test, not end to end.
  @id:uninstall-filesystem-root-refused @requires-os:linux
  Scenario: uninstall-03 - A data folder set to the top of the filesystem is refused
    Given the data folder is set to the top of the filesystem
    When the user previews an uninstall
    Then the uninstall is refused because the data folder "/" is "the top of the filesystem"
    And the refusal names ROCM_CLI_DATA_DIR as where the data folder came from
    And the refusal advises re-running with --keep-data

  @id:uninstall-home-refused-nothing-removed
  Scenario: uninstall-04 - A data folder set to the home folder is refused, nothing is removed, and the advised flag works
    Given the data folder is set to the user's home folder, which holds the user's files
    When the user uninstalls
    Then the uninstall is refused because the data folder "<home>" is "your home folder"
    And the refusal says nothing was removed
    And the refusal advises re-running with --keep-data
    And the user's files in the home folder are still there
    And the config folder is still there
    When the user uninstalls again with the flag the refusal advised
    Then the uninstall succeeds
    And the user's files in the home folder are still there
    And the config folder is gone

  @id:uninstall-shared-cache-inside-root-warned
  Scenario: uninstall-05 - Shared caches inside the cache folder are named as deleted before anything is removed
    Given the cache folder is set to the user's own cache folder, which holds the uv and model caches
    When the user previews an uninstall
    Then the review warns that the uv and model caches will be deleted with the cache folder
    And the uv cache is still there after the preview

  # The text of `.` is not the home folder, but the folder it names is. Linux-
  # only like uninstall-01/-02 in what it claims: the resolution goes through
  # the same symlink-following call, which nothing verifies on Windows.
  @id:uninstall-dot-from-home-refused @requires-os:linux
  Scenario: uninstall-06 - A data folder of `.` run from the home folder is refused and nothing is removed
    Given the data folder is set to `.`, and the user is in their home folder, which holds their files
    When the user uninstalls from their home folder
    Then the refusal says the data folder "." resolves to the home folder
    And the refusal says nothing was removed
    And the user's files in the home folder are still there
    And the config folder is still there
