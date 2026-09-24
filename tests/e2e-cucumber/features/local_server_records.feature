Feature: Local server records

  # A `rocm serve --managed` launch leaves a record on disk and keeps it after
  # the server stops. `model_serving.feature`'s serve-22 covers the surface that
  # names them (`rocm services list`); these cover the two other surfaces that
  # report them, and have no scenario anywhere else: `rocm storage report`, which
  # had never mentioned the folder at all, and the dashboard's services overlay,
  # which rendered only the live instances the daemon scrapes.
  #
  # Why a new feature file: these scenarios are about one behaviour area - the
  # records a managed `rocm serve` leaves on disk - rather than about the
  # commands that happen to display them, so they get their own `FEATURE_KEYS`
  # key (`server-records`) and stay together under one prefix. There is no
  # `storage.feature` to hold the report scenario, and filing the overlay
  # scenario under `dash-NN` would bury a records assertion among scenarios
  # about the dashboard itself. The deletion half of EAI-8075 adds its
  # scenarios here for the same reason.
  #
  # Appending to an existing file is still the right call when a scenario
  # belongs to that file's area: serve-22 tests `rocm services list`, a serve
  # surface, so this change appends it to `model_serving.feature` rather than
  # bringing it here.
  #
  # Both plant the record rather than failing a real serve: no GPU, so they run
  # on the mock lane every PR.

  @id:server-records-storage-report-names-the-folder
  Scenario: server-records-01 - The disk report names the folder local server records are kept in
    Given a local server attempt has failed
    When the user asks what ROCm CLI is keeping on disk
    Then the report names the folder holding local server records
    And the report says which of its folders can be downloaded again

  @id:server-records-dashboard-overlay-counts-them @requires-os:linux
  Scenario: server-records-02 - The dashboard's services overlay counts records that are no longer running
    Given a local server attempt has failed
    When the user opens the dashboard
    And the user opens the Observe view
    And the user opens the managed services overlay
    Then the overlay reports the record that is no longer running
    When the user closes the managed services overlay
    And the user quits the dashboard
    Then the dashboard exits successfully
