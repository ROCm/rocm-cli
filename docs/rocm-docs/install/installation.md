<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Installing ROCm CLI

ROCm CLI ships as a prebuilt bundle that contains the `rocm` command and the
`rocmd` background daemon. Platform support:

```{include} ../../../README.md
:start-after: "<!-- platform-support-table-start -->"
:end-before: "<!-- platform-support-table-end -->"
```

Live dashboard telemetry requires Linux or WSL2 (see
[Interactive interfaces](../getting-started.md#interactive-interfaces)). vLLM
serving is Linux or WSL2 only (see
[vLLM adapter](../engines/vllm.md)).

```{include} ../../../README.md
:start-after: "<!-- docs-site: platform-notes-end -->"
:end-before: "> [!IMPORTANT]"
```

## Install ROCm CLI

```{include} ../../../README.md
:start-after: "## Installation"
:end-before: "<!-- docs-site: verify-next-start -->"
```

Continue with [First run](../getting-started.md#first-run).

```{include} ../../../README.md
:start-after: "<!-- docs-site: verify-next-end -->"
:end-before: "See [CONTRIBUTING.md]"
```

## Next steps

- [Getting started](../getting-started.md): configure ROCm and serve your first
  model.
- [Contributing](../about/contributing.md): the full development setup, test
  commands, and commit-signing requirements.
- To remove ROCm CLI and what it manages, see
  [Logs and cleanup](../commands.md#logs-and-cleanup).
