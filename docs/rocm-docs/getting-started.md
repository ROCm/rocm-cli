<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Getting started with ROCm CLI

<!-- Single-sourced from README.md. The `docs-site:` HTML comments in README.md
mark where each include starts and stops. They are invisible on GitHub. Keep
them in place when you edit the README. -->

## First run

```{include} ../../README.md
:start-after: "## First run"
:end-before: "## Interactive interfaces"
```

## Configure ROCm and serve a model

```{include} ../../README.md
:start-after: "## Configure ROCm and serve a model"
:end-before: "<!-- docs-site: install-gate-start -->"
```

<!-- Deliberate copy of the paragraph between the install-gate markers in
README.md, with the cross-reference retargeted to this site. Edit both
together. -->
Running the command when a managed runtime is already the active default asks
first, because the new install takes over as the active default. See
[ROCm installation](commands.md#rocm-installation) for that gate, the flags that
approve it without a prompt, and the ROCm 10 and newer requirements.

```{include} ../../README.md
:start-after: "<!-- docs-site: install-gate-end -->"
:end-before: "<!-- docs-site: serve-note-start -->"
```

<!-- Deliberate copy of the paragraph between the serve-note markers in
README.md, with the cross-reference retargeted to this site. Edit both
together. -->
You can also serve any compatible Hugging Face model directly. See
[Model serving](commands.md#model-serving) for the GGUF versus safetensors rule,
because which form works depends on the engine your GPU selects.

```{include} ../../README.md
:start-after: "<!-- docs-site: serve-note-end -->"
:end-before: "## Commands"
```

## Interactive interfaces

<!-- This page shows the interactive interfaces after the quick reference. The
README shows them before "Configure ROCm and serve a model". -->

```{include} ../../README.md
:start-after: "## Interactive interfaces"
:end-before: "## Configure ROCm and serve a model"
```
