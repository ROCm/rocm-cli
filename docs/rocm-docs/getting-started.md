<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Getting started with ROCm CLI

```{include} ../../README.md
:start-after: "## First run"
:end-before: "## Interactive interfaces"
```

## Configure ROCm and serve a model

```{include} ../../README.md
:start-after: "## Configure ROCm and serve a model"
:end-before: "Running the command when a"
```

<!-- The prose below is a deliberate copy of the README sentence, with the
     cross-reference retargeted to this site. The link text is then reused as
     the `:start-after:` anchor for the next include, which also matches the
     original sentence in README.md; edit both together. -->
Running the command when a managed runtime is already the active default asks
first, because the new install takes over as the active default; see
[ROCm installation](commands.md#rocm-installation) for that gate and the flags
that approve it without a prompt.

```{include} ../../README.md
:start-after: "for that gate and the flags that approve it without a prompt."
:end-before: "You can also serve any compatible Hugging Face model directly"
```

You can also serve any compatible Hugging Face model directly — see
[Model serving](commands.md#model-serving) for the GGUF-vs-safetensors rule,
since which form works depends on the engine your GPU selects.

```{include} ../../README.md
:start-after: "form works depends on the engine your GPU selects."
:end-before: "## Commands"
```

## Interactive interfaces

```{include} ../../README.md
:start-after: "## Interactive interfaces"
:end-before: "## Configure ROCm and serve a model"
```
