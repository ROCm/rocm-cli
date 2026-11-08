<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Getting started with ROCm CLI

```{include} ../../README.md
:start-after: "## First run"
:end-before: "## Interactive interfaces"
```

The following recording shows these steps: inspecting the environment,
discovering engines and models, finding a running service, and chatting with a
locally served model.

![ROCm CLI demo](https://raw.githubusercontent.com/ROCm/rocm-cli/media/cli.gif)

## Configure ROCm and serve a model

```{include} ../../README.md
:start-after: "## Configure ROCm and serve a model"
:end-before: "Running `rocm install sdk` when a"
```

<!-- The prose below is a deliberate copy of the README sentence, with the
     cross-reference retargeted to this site. The end of the sentence is then
     reused as the `:start-after:` anchor for the next include, which also
     matches the original sentence in README.md; edit both together. -->
Running `rocm install sdk` when a managed runtime is already the active default
asks first, because the new install takes over as the active default; see
[ROCm installation](commands.md#rocm-installation) for that gate, the flags
that approve it without a prompt, and the ROCm 10 and newer requirements.

```{include} ../../README.md
:start-after: "newer requirements."
:end-before: "You can also serve any compatible Hugging Face model directly"
```

You can also serve any compatible Hugging Face model directly. See
[Model serving](commands.md#model-serving) for the GGUF versus safetensors rule,
because which form works depends on the engine your GPU selects.

```{include} ../../README.md
:start-after: "form works depends on the engine your GPU selects."
:end-before: "## Commands"
```

## Interactive interfaces

```{include} ../../README.md
:start-after: "## Interactive interfaces"
:end-before: "## Configure ROCm and serve a model"
```
