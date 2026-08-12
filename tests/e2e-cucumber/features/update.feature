Feature: Checking for and previewing a ROCm update

  # `rocm update` is a query by default and only changes the machine when asked
  # to. Nothing here installs anything: the scenarios cover what the command
  # accepts, which is pure argument handling and so needs no GPU, no runtime,
  # and no network.

  # Expected to FAIL. Asking to see what an update would do, without asking for
  # it to be done, is refused as a misuse — even though checking is what this
  # command does when left alone. The two choices are documented as independent
  # of each other, so a user who wants a preview before committing to anything
  # is turned away from the one command that would give them it.
  @id:update-preview-without-applying
  Scenario: 1 - Previewing an update without asking to install it is accepted
    When the user asks to see what updating would do without asking for it to be done
    Then the request is accepted rather than refused as a misuse
