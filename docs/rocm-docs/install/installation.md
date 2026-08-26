# Installing ROCm CLI

ROCm CLI ships as a single prebuilt binary. Platform support:

| Platform | Prebuilt binary | Notes |
|---|---|---|
| Linux (x86_64) | Yes | Full support, including the live dashboard and both inference engines |
| Windows (x86_64) | Yes | CLI and Lemonade serving; no live dashboard or vLLM |
| WSL2 (x86_64) | Yes (Linux binary) | Full support, including the live dashboard; see `docs/wsl.md` for setup |
| macOS | No | No official installer, release, CI, or QA coverage |

Live dashboard telemetry requires Linux or WSL2 (see
[Interactive interfaces](../getting-started.md#interactive-interfaces)). vLLM
serving is Linux/WSL2 only (see `docs/vllm.md`).

```{include} ../../../README.md
:start-after: "## Installation"
:end-before: "See [CONTRIBUTING.md]"
```

See [Contributing](../about/contributing.md) for the full development setup, test
commands, and commit-signing requirements.
