<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# ROCm CLI on WSL2

This page covers setting up ROCm CLI on WSL2 with ROCDXG (`librocdxg`) and a
TheRock-managed Python virtual environment, and diagnosing a WSL2 host. For
the other platforms, see [Installing ROCm CLI](installation.md).

<!-- The sections below are single-sourced from docs/wsl.md. Each heading is
     written here in sentence case, and each include is anchored on the
     matching heading in that file. docs/wsl.md also holds maintainer notes
     (runtime environment design, install UX recommendations, and test plans)
     that are deliberately not published, so keep the anchors in step with it. -->

## Prerequisites

```{include} ../../wsl.md
:start-after: "## Prerequisites"
:end-before: "## Install ROCDXG In WSL"
```

## Install ROCDXG in WSL

```{include} ../../wsl.md
:start-after: "## Install ROCDXG In WSL"
:end-before: "## TheRock Runtime Env In WSL"
```

## Diagnosing a WSL host

```{include} ../../wsl.md
:start-after: "## Diagnosing A WSL Host"
:end-before: "## What `rocm examine` Reports On WSL"
```

## What `rocm examine` reports on WSL

```{include} ../../wsl.md
:start-after: "## What `rocm examine` Reports On WSL"
:end-before: "## Install UX Recommendations"
```
