# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

Feature: Checking whether this machine is healthy

  # One read-only command reports what the machine has and what the CLI makes of
  # it. It supersedes the two commands that used to split that job in half, and
  # which asked the machine the same questions twice to do it.
  #
  # Every scenario here is black-box and GPU-independent (no serve, no download,
  # no mutation), so they all run on the mock lane / per-PR tier. They assert the
  # SHAPE of a report rather than any specific finding: what the catalog matches
  # is environment-dependent, and is covered by diagnose.feature.

  @id:doctor-reports-state-and-findings
  Scenario: 1 - One command reports both what was found and what looks wrong
    When the user asks the CLI to check this machine
    Then the CLI reports what hardware and ROCm setup it found
    And the CLI reports what it makes of them

  @id:doctor-changes-nothing
  Scenario: 2 - Checking the machine never changes it
    When the user asks the CLI to check this machine
    Then nothing on the machine is changed

  # The sharp version of scenario 2. The host report has one write on it: an
  # install that is present on disk but missing from the CLI's own records gets
  # silently re-recorded. That is a repair, and a command that promises to change
  # nothing must not do it. The second half is the control — it proves the setup
  # really would have been repaired, so the first half cannot pass vacuously.
  @id:doctor-does-not-repair-records
  Scenario: 3 - Checking does not repair what the superseded inspection repairs
    Given a machine with a ROCm install missing from the CLI's records
    When the user asks the CLI to check this machine
    Then the install is still missing from the records
    When a script runs the superseded host inspection
    Then the install has been added to the records

  # The two superseded commands are hidden, not removed. Scripts, the packaged
  # daemon, and the suite's own capability probe still call them, so "hidden"
  # must not have quietly become "broken".
  @id:doctor-supersedes-older-commands
  Scenario: 4 - Scripts written against the older inspection commands keep working
    Given a script written against an earlier release
    When it runs the superseded inspection commands
    Then each one still reports what it always did

  @id:doctor-is-the-only-advertised-check
  Scenario: 5 - Only one health check is advertised
    When the user asks the CLI what it can do
    Then a single health check is offered
    And the superseded inspection commands are not advertised

  # The whole point of being able to hand a captured report back in: a machine
  # that cannot be reached interactively can still be reasoned about, and the
  # expensive inspection is not repeated to do it.
  @id:doctor-checks-a-saved-report
  Scenario: 6 - A report captured earlier can be checked without inspecting the machine again
    Given a report captured from an earlier check
    When the user asks the CLI to check that saved report
    Then the CLI reports what it makes of the saved report
    And the CLI does not describe this machine

  @id:doctor-checks-a-saved-report-from-stdin
  Scenario: 7 - A saved report can be handed over on the standard input
    Given a report captured from an earlier check
    When the user pipes that saved report into the CLI
    Then the CLI reports what it makes of the saved report

  # The machine-readable report is a superset of the one the superseded command
  # emitted, so tooling written against that document keeps reading every key it
  # already relied on and simply gains the findings.
  @id:doctor-json-is-a-superset
  Scenario: 8 - The machine-readable report still carries everything it used to
    When the user asks for both machine-readable reports
    Then the newer report carries every fact the superseded one did
    And the newer report also carries the findings
