Feature: Local server records

  # A `rocm serve --managed` launch leaves a record on disk and keeps it after
  # the server stops. `model_serving.feature`'s serve-22 covers the surface that
  # names them (`rocm services list`); these cover the two other surfaces that
  # report them, and have no scenario anywhere else: `rocm storage report`, which
  # had never mentioned the folder at all, and the dashboard's services overlay,
  # which rendered only the live instances the daemon scrapes.
  #
  # Why a new feature file rather than `storage`/`dash` indexes: the naming guard
  # (`tests/feature_naming.rs`) requires per-file indexes to be sequential in
  # declaration order, so a scenario appended to an existing file must take that
  # file's *next* index - there is no way to reserve a higher one. `dash-12` is
  # already claimed by two open PRs and `serve-22` by two more, so appending here
  # would collide with whichever lands first, and the guard would only notice
  # after both had merged. A new file takes a new key and collides with nothing.
  # The deletion half of EAI-8075 adds its scenarios here.
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
