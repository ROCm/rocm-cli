Feature: Lemonade installs on disk

  # Lemonade manages its own runtime, and where it installs depends on whether a
  # ROCm runtime is active: into the data folder when none is, into that
  # runtime's own folder when one is. So `rocm engines install lemonade` before
  # `rocm install sdk`, then `rocm serve` after it, leaves two full copies of
  # roughly 25 GiB each, and the second one used to arrive silently from
  # `rocm serve` (ROCMAI-825). These scenarios cover what the user is now told
  # about that: what an install needs before it starts, both copies in the disk
  # report, and the install record when a runtime holding a copy is removed.
  #
  # A new feature file rather than appended scenarios: these are about one
  # behaviour area, Lemonade's footprint on disk, which spans `engines install`,
  # `storage report` and `runtimes uninstall`.
  #
  # None downloads anything or needs a GPU, so all run on the mock lane every PR.
  # lemonade-installs-01 and -02 need the `rocm/e2e-test-hooks` build
  # `cargo xtask e2e` makes, to report a nearly full drive without filling a
  # real one; lemonade-installs-07 needs it to report old host libraries on a
  # host whose libraries are new.

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

  @id:lemonade-installs-storage-lists-both-copies
  Scenario: lemonade-installs-03 - The disk report lists both Lemonade copies and flags the unused one
    Given Lemonade was installed in the data folder and again inside a ROCm install
    When the user asks what ROCm CLI is keeping on disk
    Then the report lists both Lemonade installs with their sizes
    And the report flags the data folder's Lemonade install as unused

  @id:lemonade-installs-runtime-uninstall-clears-record
  Scenario: lemonade-installs-04 - Removing the ROCm install that held Lemonade clears its install record
    Given Lemonade was installed in the data folder and again inside a ROCm install
    When the user uninstalls the ROCm install holding Lemonade
    Then the CLI says Lemonade's install was removed with that ROCm install
    And Lemonade is no longer recorded as installed

  @id:lemonade-installs-dangling-record-keeps-the-survivor
  Scenario: lemonade-installs-05 - A record naming a folder with no install does not make the surviving copy reclaimable
    Given the ROCm install that held Lemonade is gone and only the data folder's copy is left
    When the user asks what ROCm CLI is keeping on disk
    Then the report does not call the surviving Lemonade copy unused or reclaimable

  @id:lemonade-installs-read-only-runtime-keeps-its-copy
  Scenario: lemonade-installs-06 - Uninstalling a read-only ROCm install leaves its Lemonade copy, and the report says so
    Given a read-only ROCm install holds a Lemonade copy
    When the user asks what ROCm CLI is keeping on disk
    Then the report does not say that copy is removed with its ROCm install
    When the user uninstalls the ROCm install holding Lemonade
    Then that Lemonade copy is still on disk

  # The pinned Linux Lemonade server links against a minimum glibc and OpenSSL.
  # On an older host it used to exit with a loader error that reached the user
  # only as "Lemonade server did not become ready"; the install now refuses
  # first and names both versions. Linux only: Windows has no such floor.
  @id:lemonade-installs-refused-on-too-old-host-libraries @requires-os:linux
  Scenario: lemonade-installs-07 - A Lemonade install on a host with too old glibc and OpenSSL is refused before downloading
    Given this host's glibc and OpenSSL are older than the pinned Lemonade server needs
    When the user installs the Lemonade engine
    Then the Lemonade install is refused for the host's libraries before anything is downloaded
