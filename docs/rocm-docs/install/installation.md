<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Installing ROCm CLI

ROCm CLI ships as prebuilt binaries (`rocm` and `rocmd`). Platform support:

```{include} ../../../README.md
:start-after: "<!-- platform-support-table-start -->"
:end-before: "<!-- platform-support-table-end -->"
```

For WSL2 setup and diagnosis, see [ROCm CLI on WSL2](wsl.md).

Live dashboard GPU/CPU/memory telemetry and the managed servers
`rocm services list` reports work on Windows too; per-server throughput and
Docker-discovered instances are Linux/WSL2 only, since there's no
`rocmdashd` unix socket on Windows (see
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

## Behind a proxy

`rocm install sdk` and the other commands that download files follow the
standard proxy variables, in upper or lower case (the lower-case name wins when
both are set). That covers, among others, the managed `uv`, the SDK tarball
catalog and archives, ComfyUI and Lemonade installs, the AIM image metadata hint
and the `rocmd` artifact prefetch. Downloads of GGUF model metadata from the
Hugging Face hub, the provider requests of `rocm chat`, the `rocm bench load` model
probe and the `rocmd` Anthropic bridge still connect directly. (The dashboard
chat's own clients read the proxy variables themselves.)

| Variable | Use |
| --- | --- |
| `HTTPS_PROXY` | Proxy for `https://` URLs. |
| `HTTP_PROXY` | Proxy for `http://` URLs. |
| `ALL_PROXY` | Proxy for either scheme when the specific variable is unset. |
| `NO_PROXY` | Comma-separated hosts that skip the proxy: `example.com` and `.example.com` cover the host and its subdomains, `*` covers everything, and `host:port` limits an entry to one port (a URL without a port is on 80 or 443). A CIDR range is not interpreted. |

```bash
export HTTPS_PROXY=http://proxy.example.com:3128
export NO_PROXY=localhost,.example.com
rocm install sdk
```

Requests to `localhost` and other loopback addresses never use the proxy. A
proxy URL may carry credentials (`http://user:password@host:port`), which are
sent for `https://` downloads; a plain `http://` download through an
authenticated proxy is sent without them. Only an `http://` proxy address is
supported:

- an `https://` address is rejected: a warning is printed once and the request
  goes direct;
- a `socks5://` address makes the request fail.

The proxy is chosen from the URL a request starts at, so a redirect to another
host keeps it. The `pip` and `uv` steps of an install read the same variables
themselves. `rocmd` reads the environment it was started with.

## Uninstall ROCm CLI

To remove ROCm CLI and what it manages:

```bash
rocm uninstall
```

Pass `--dry-run` first to preview what it removes. For the full flag list, see
[Logs and cleanup](../commands.md#logs-and-cleanup).
