Feature: Runtime configuration

  @id:runtime-install-sdk-active @requires-gpu
  Scenario: 1 - Installing the SDK makes it the active runtime
    Given a machine with no CLI-managed runtimes
    When the user installs the SDK
    Then a runtime is registered
    And the runtime is set as active
    And the runtime includes an inference engine

  @id:runtime-adopt-preexisting-rejected
  Scenario: 2 - Adopting a pre-existing ROCm install is rejected with guidance
    Given a machine with a standard ROCm install
    When the user tries to adopt the existing install
    Then the adoption is refused
    And the error explains which install types can be adopted
