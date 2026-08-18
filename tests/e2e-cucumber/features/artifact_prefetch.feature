Feature: Artifact prefetch cleanup

  @id:artifact-prefetch-failed-marker-leaves-no-temp
  Scenario: A failed artifact cache write leaves no temporary marker
    Given a signed direct-download artifact fixture
    And its cache marker destination is occupied by a directory
    When the user approves the artifact prefetch
    Then the artifact download completes before marker publication fails
    And no temporary cache marker is left behind
    And the occupied cache marker destination remains unchanged
