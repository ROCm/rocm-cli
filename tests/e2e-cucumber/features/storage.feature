Feature: Storage cleanup

  # `rocm storage remove-downloads` deletes files ROCm CLI can fetch again. Its
  # `--help` promises that, without --yes and outside a terminal, it prints what
  # it would remove and refuses rather than asking or assuming an answer. The
  # harness runs the binary with no terminal, so these are that case.
  #
  # `remove-old-installs` reaches the same gate through the same function; it is
  # not given its own scenario because planting an old runtime install is a
  # much larger fixture than planting a download; the shared gate has its own
  # unit test (`removal_requires_yes_when_there_is_no_terminal_to_prompt`).
  #
  # Linux only, like `uninstall.feature`: the refusal itself is not
  # platform-specific, but nothing here has been run on Windows, so the claim is
  # not made there.

  @id:storage-remove-downloads-refused-without-yes-outside-terminal @requires-os:linux
  Scenario: storage-01 - Removing downloaded files without confirming outside a terminal is refused and removes nothing
    Given ROCm CLI has a downloaded file on disk
    When the user removes the downloaded files without confirming
    Then the removal is refused and the user is told to confirm explicitly
    And the downloaded file is still there
