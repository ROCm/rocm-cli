Feature: Release install lifecycle

  # The full release lifecycle — packaging, signature-verified install, tamper
  # rejection, reinstall, PATH handling, and uninstall — expressed as black-box
  # scenarios that drive `cargo xtask package`, the real root installer
  # (install.sh / install.ps1), and the installed binaries. These replace the
  # former scripts/acceptance-install-upgrade-tui-uninstall.{sh,ps1}.
  #
  # Every scenario is tagged @lifecycle and is SKIPPED by default so the fast
  # `cargo xtask e2e` suite stays fast. Run the current host's set explicitly:
  #   E2E_INCLUDE_LIFECYCLE=1 E2E_ONLY_LIFECYCLE=1 cargo xtask e2e
  # They are also expensive and OS-mutating, so heavy CI runs them per platform.
  #
  # Scenarios are independent: each packages its own bundle, generates its own
  # signing key, and installs into its own directory rooted in the scenario's
  # temp dir, so ordering never affects the outcome. Cargo's release build cache
  # is shared naturally.

  # ── Packaging + signature-verified install (Linux) ────────────────────

  @id:lifecycle-linux-install-signed-keypath @lifecycle @requires-os:linux
  Scenario: Linux - a bundle signed with a key file installs, verifies, and sets up the shell profile
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the signed bundle is installed with the public key file updating the shell profile
    Then the installer reports the signature verified
    And the installer reports the shell profile updated
    And the installed rocm and rocmd binaries are present
    And the install manifest is present
    And a minimal config is seeded with the lemonade default engine
    And the shell profile lists the install directory

  @id:lifecycle-linux-install-signed-pem @lifecycle @requires-os:linux
  Scenario: Linux - a bundle signed with an inline PEM installs and verifies
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key PEM
    And the signed bundle is installed with the public key PEM
    Then the installer reports the signature verified
    And the installed rocm binary is present
    And the install manifest is present

  # ── Trust rejection: untrusted key, bad checksum, bad/missing signature ─

  @id:lifecycle-linux-reject-untrusted-key @lifecycle @requires-os:linux
  Scenario: Linux - an install with no public key falls back to the pinned trust root and rejects an untrusted signer
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the signed bundle is installed with no public key supplied
    Then the install fails reporting signature verification failed
    And no binaries are activated in the target directory

  @id:lifecycle-linux-reject-bad-checksum @lifecycle @requires-os:linux
  Scenario: Linux - a tampered checksum is rejected before activation
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the bundle checksum sidecar is corrupted
    And the signed bundle is installed with the public key file
    Then the install fails reporting checksum verification failed
    And no binaries are activated in the target directory

  @id:lifecycle-linux-reject-bad-signature @lifecycle @requires-os:linux
  Scenario: Linux - a tampered signature is rejected before activation
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the bundle signature sidecar is corrupted
    And the signed bundle is installed with the public key file
    Then the install fails reporting signature verification failed
    And no binaries are activated in the target directory

  @id:lifecycle-linux-reject-missing-signature @lifecycle @requires-os:linux
  Scenario: Linux - a missing signature is rejected when a signature is required
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the bundle signature sidecar is removed
    And the signed bundle is installed with the public key file
    Then the install fails reporting the required signature is missing
    And no binaries are activated in the target directory

  # ── Reinstall: stale-manifest purge, config preservation, PATH idempotency ─

  @id:lifecycle-linux-reinstall-purges-stale @lifecycle @requires-os:linux
  Scenario: Linux - reinstalling purges a stale prior entry and preserves config
    Given a freshly built release tree
    And a generated signing keypair
    And a signed bundle installed with the shell profile updated
    When a stale engine entry is recorded in the prior install
    And the user changes the default engine to vllm in the installed config
    And the signed bundle is reinstalled with the shell profile updated
    Then the installer reports removing the previous install
    And the stale engine entry is gone
    And the install manifest is present
    And the preserved config still selects the vllm default engine
    And the shell profile has exactly one rocm-cli PATH marker

  # ── Installed-binary PTY, shell-profile / XDG, and uninstall ────────────

  @id:lifecycle-linux-installed-binary-pty @lifecycle @requires-os:linux
  Scenario: Linux - the installed binary opens and exits interactive chat through a pseudo-terminal
    Given a freshly built release tree
    And a generated signing keypair
    And a signed bundle installed with the public key file
    When the installed rocm opens interactive chat through a pseudo-terminal
    Then the installed interactive chat surface is displayed
    When the user quits the installed interactive chat
    Then the installed interactive chat exits successfully

  @id:lifecycle-linux-uninstall-full-purge @lifecycle @requires-os:linux
  Scenario: Linux - uninstall removes binaries, manifest, and XDG state
    Given a freshly built release tree
    And a generated signing keypair
    And a signed bundle installed with the shell profile updated
    And the installed binary has isolated XDG directories with state
    When the user uninstalls from the installed binary
    Then the installed rocm and rocmd binaries are gone
    And the install manifest is gone
    And the isolated XDG config, data, and cache state is gone

  # ── Windows user-PATH restoration, loopback HTTP install, isolated smoke ─

  @id:lifecycle-windows-install-signed @lifecycle @requires-os:windows
  Scenario: Windows - a signed zip installs and verifies with native crypto
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the signed bundle is installed with the public key file
    Then the installer reports the signature verified
    And the installed rocm and rocmd binaries are present
    And the install manifest is present

  @id:lifecycle-windows-key-rotation-fallback @lifecycle @requires-os:windows
  Scenario: Windows - a malformed current trust root falls through to the valid next key
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And an installer fixture has a malformed current key and the generated public key next
    And the signed bundle is installed through the pinned-key fixture
    Then the installer reports the signature verified
    And the installed rocm binary is present

  @id:lifecycle-windows-all-pinned-keys-malformed @lifecycle @requires-os:windows
  Scenario: Windows - all malformed pinned trust roots fail deterministically
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And an installer fixture has malformed current and next keys
    And the signed bundle is installed through the pinned-key fixture
    Then the install fails reporting signature verification failed
    And no binaries are activated in the target directory

  @id:lifecycle-windows-updates-user-path @lifecycle @requires-os:windows
  Scenario: Windows - a default install updates the user PATH and restores it afterwards
    Given a freshly built release tree
    And a generated signing keypair
    And the user PATH is captured for restoration
    When the release is packaged and signed with the private key file
    And the signed bundle is installed updating the user PATH
    Then the installer reports the user PATH updated
    And the installer reports the installer process PATH updated
    And the install directory is on the user PATH

  @id:lifecycle-windows-verifies-without-openssl @lifecycle @requires-os:windows
  Scenario: Windows - the installer verifies with native crypto when openssl is absent from PATH
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the signed bundle is installed with the public key file and openssl removed from PATH
    Then the installer reports the signature verified
    And the installed rocm binary is present
    And the install manifest is present

  @id:lifecycle-windows-rejects-bad-signature-without-openssl @lifecycle @requires-os:windows
  Scenario: Windows - a bad signature is rejected with native crypto when openssl is absent from PATH
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the bundle signature sidecar is corrupted
    And the signed bundle is installed with the public key file and openssl removed from PATH
    Then the install fails reporting signature verification failed
    And no binaries are activated in the target directory

  @id:lifecycle-windows-reinstall-purges-stale @lifecycle @requires-os:windows
  Scenario: Windows - reinstalling purges a stale prior entry
    Given a freshly built release tree
    And a generated signing keypair
    And a signed bundle installed with the public key file
    When a stale engine entry is recorded in the prior install
    And the signed bundle is reinstalled with the public key file
    Then the installer reports removing the previous install
    And the stale engine entry is gone
    And the install manifest is present

  @id:lifecycle-windows-reject-untrusted-key @lifecycle @requires-os:windows
  Scenario: Windows - an install with no public key rejects an untrusted signer
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the signed bundle is installed with no public key supplied
    Then the install fails reporting signature verification failed
    And no binaries are activated in the target directory

  @id:lifecycle-windows-reject-bad-checksum @lifecycle @requires-os:windows
  Scenario: Windows - a tampered checksum is rejected before activation
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the bundle checksum sidecar is corrupted
    And the signed bundle is installed with the public key file
    Then the install fails reporting checksum verification failed
    And no binaries are activated in the target directory

  @id:lifecycle-windows-reject-bad-signature @lifecycle @requires-os:windows
  Scenario: Windows - a tampered signature is rejected before activation
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the bundle signature sidecar is corrupted
    And the signed bundle is installed with the public key file
    Then the install fails reporting signature verification failed
    And no binaries are activated in the target directory

  @id:lifecycle-windows-reject-missing-signature @lifecycle @requires-os:windows
  Scenario: Windows - a missing signature is rejected when a signature is required
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the bundle signature sidecar is removed
    And the signed bundle is installed with the public key file
    Then the install fails reporting the required signature is missing
    And no binaries are activated in the target directory

  @id:lifecycle-windows-http-install @lifecycle @requires-os:windows
  Scenario: Windows - a signed bundle installs over a loopback HTTP download
    Given a freshly built release tree
    And a generated signing keypair
    When the release is packaged and signed with the private key file
    And the signed bundle is installed over a loopback HTTP server with the public key file
    Then the installer reports the signature verified
    And the installed rocm binary is present
    And the install manifest is present

  @id:lifecycle-windows-installed-binary-smoke @lifecycle @requires-os:windows
  Scenario: Windows - the installed binary honors isolated directories and keeps the running executable on uninstall
    Given a freshly built release tree
    And a generated signing keypair
    And a signed bundle installed with the public key file
    And the installed binary has isolated directories with state
    When the installed rocm version runs
    Then the installed command exits successfully
    When the installed rocm engines list runs
    Then the installed command exits successfully
    When the installed rocmd status runs
    Then the installed command exits successfully
    When the installed rocm examine runs with the isolated environment
    Then examine reads only the isolated config, data, and cache directories
    When the user uninstalls from the installed binary keeping config data and cache
    Then uninstall reports skipping the running executable on Windows
    And the install manifest is gone
