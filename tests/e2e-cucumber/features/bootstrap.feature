Feature: Non-interactive bootstrap setup

  # #75/#441: `rocm bootstrap setup` without a terminal must still name the
  # install-folder choice the interactive onboarding wizard offers, not just
  # point the caller back at it. A unit test pins the message string in
  # isolation; only running the built binary piped (the shape every CI runner
  # or script uses) proves a real non-interactive invocation actually prints it.
  @id:bootstrap-setup-no-tty-advertises-install-folder
  Scenario: bootstrap-01 - Running bootstrap setup without a terminal advertises the install-folder choice
    When the user runs bootstrap setup without a terminal
    Then the CLI tells the user how to choose an install folder
