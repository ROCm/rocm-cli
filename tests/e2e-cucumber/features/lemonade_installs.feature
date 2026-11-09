Feature: Lemonade installs on disk

  # Lemonade manages its own runtime, and where it installs depends on whether a
  # ROCm runtime is active: into the data folder when none is, into that
  # runtime's own folder when one is. So `rocm engines install lemonade` before
  # `rocm install sdk`, then `rocm serve` after it, leaves two full copies of
  # roughly 25 GiB each, and the second one used to arrive silently from
  # `rocm serve` (ROCMAI-825). These scenarios cover what the user is now told
  # about that, starting with what an install needs before it starts.
  #
  # A new feature file rather than appended scenarios: these are about one
  # behaviour area, Lemonade's footprint on disk, which spans `engines install`,
  # `storage report` and `runtimes uninstall`.
  #
  # None downloads anything or needs a GPU, so all run on the mock lane every PR.
  # lemonade-installs-01 and -02 need the `rocm/e2e-test-hooks` build
  # `cargo xtask e2e` makes, to report a nearly full drive without filling a
  # real one.

  @id:lemonade-installs-refused-without-space
  Scenario: lemonade-installs-01 - A Lemonade install the drive cannot hold is refused before downloading
    Given the drive for a new Lemonade install cannot hold even Lemonade itself
    When the user installs the Lemonade engine
    Then the Lemonade install is refused before anything is downloaded
    And the user is told what the Lemonade install needs and where it would go

  # `rocm serve` reaches the same install on its own, which is how the second
  # copy used to appear without a word. The free-space override also stands in
  # for the GPU the no-GPU lane does not have, as serve-18's failure hook does.
  @id:lemonade-installs-serve-refused-without-space
  Scenario: lemonade-installs-02 - Serving with Lemonade on a drive that cannot hold it is refused before downloading
    Given the drive for a new Lemonade install cannot hold even Lemonade itself
    When the user serves a model with Lemonade
    Then the Lemonade install is refused before anything is downloaded
    And the user is told what the Lemonade install needs and where it would go
