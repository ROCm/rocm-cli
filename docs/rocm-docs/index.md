---
myst:
  html_meta:
    "description": "ROCm CLI is a command-line tool for setting up and running local AI on AMD GPUs."
    "keywords": "ROCm CLI, ROCm, AMD, GPU, local AI"
---

<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# ROCm CLI documentation

ROCm CLI is a command-line tool for setting up and running local AI on AMD
GPUs, with a full-screen TUI dashboard for GPU telemetry, model serving, and
chat.

It ships as prebuilt binaries (`rocm` and `rocmd`) for Linux and Windows
(x86_64), needs no Python, Rust, or existing ROCm install, and includes
inference engine adapters for Lemonade and vLLM.

```{important}
**Tech Preview:** This software is provided as-is, without warranty or
guarantee of stability. APIs, commands, and behavior might change without
notice. Intended for experimentation and early feedback only.
```

The ROCm CLI public repository is located at
[ROCm/rocm-cli](https://github.com/ROCm/rocm-cli).

Inspect the environment, discover engines and models, find a running service,
and chat with a locally served model:

![ROCm CLI demo](https://raw.githubusercontent.com/ROCm/rocm-cli/media/cli.gif)

````{grid} 1 2 2 2
:gutter: 3

```{grid-item-card} Install ROCm CLI

Check platform support and install the prebuilt binaries.

* [Installing ROCm CLI](install/installation.md)
* [ROCm CLI on WSL2](install/wsl.md)
```

```{grid-item-card} Get started
:link: getting-started
:link-type: doc

Inspect your environment, configure ROCm, and serve a model.
```

```{grid-item-card} Look up a command
:link: commands
:link-type: doc

Each `rocm` command, its options, and what it does.
```

```{grid-item-card} Use the vLLM adapter
:link: engines/vllm
:link-type: doc

Serve with an existing vLLM installation on Linux and WSL.
```
````
