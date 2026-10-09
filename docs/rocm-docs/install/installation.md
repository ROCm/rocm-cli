<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Installing ROCm CLI

ROCm CLI ships as a single prebuilt binary. Platform support:

```{include} ../../../README.md
:start-after: "<!-- platform-support-table-start -->"
:end-before: "<!-- platform-support-table-end -->"
```

Live dashboard telemetry requires Linux or WSL2 (see
[Interactive interfaces](../getting-started.md#interactive-interfaces)). vLLM
serving is Linux or WSL2 only (see
[vLLM adapter](../engines/vllm.md)).

```{include} ../../../README.md
:start-after: "only (see [docs/vllm.md](docs/vllm.md))."
:end-before: "> [!IMPORTANT]"
```

```{include} ../../../README.md
:start-after: "## Installation"
:end-before: "See [CONTRIBUTING.md]"
```

See [Contributing](../about/contributing.md) for the full development setup, test
commands, and commit-signing requirements.

## Uninstall ROCm CLI

To remove ROCm CLI and what it manages:

```bash
rocm uninstall
```

Pass `--dry-run` first to preview what it removes. For the full flag list, see
[Logs and cleanup](../commands.md#logs-and-cleanup).
