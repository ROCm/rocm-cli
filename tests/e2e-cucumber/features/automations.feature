Feature: Background automation checks

  # A pure contract scenario: no fixtures, no GPU, no network. The listing is the
  # only place a user learns which background checks exist, so whatever it shows
  # has to be enough to act on. Deriving each check's identifier from the listing
  # the way a reader would IS the contract under test — the identifiers are
  # deliberately not written into the scenario or the steps, because hard-coding
  # them would pass on a listing that exposes nothing.
  @id:automations-listed-checks-can-be-enabled
  Scenario: 1 - Every background check that is listed can be turned on
    Given a machine with no background checks turned on
    When the user lists the background checks
    Then every listed check can be turned on by name
