<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# ROCm CLI compatibility matrix

This page shows which inference engines run on which platforms and ROCm
versions, which Python version each ROCm version uses, and which GPUs AIM
images support. It lists support status only. Exact package versions and the details behind each status are on the
linked pages.

<!--
Site-only page: no source file holds these tables. The platform table is
included from README.md (between the platform-support-table markers), not
copied. Keep this page status-only: exact package versions live in
docs/vllm.md and the engine code, so they are not repeated here.
-->

The engine tables use these statuses:

- **Supported**: `rocm engines install` sets up the engine for this
  combination, and it has run on GPU hardware.
- **Partial**: `rocm engines install` sets up the engine, but GPU serving is
  confirmed only in a limited setup. The note under the table says which.
- **Not verified**: `rocm engines install` sets up the engine, but it hasn't
  been confirmed on GPU hardware.
- **Not supported**: ROCm CLI doesn't offer the install, or refuses it.

## Platforms

```{include} ../../README.md
:start-after: "<!-- platform-support-table-start -->"
:end-before: "<!-- platform-support-table-end -->"
```

For the minimum Linux release and the library versions the Lemonade engine
needs, see [Installing ROCm CLI](install/installation.md). For WSL2 setup, see
[ROCm CLI on WSL2](install/wsl.md).

## Engines by platform

| Engine | Linux | Windows | WSL2 |
|---|---|---|---|
| Lemonade | Supported | Supported | Not verified |
| vLLM | Supported | Not supported | Partial |
| AIM | Supported (needs Docker) | Not supported | Not verified |

- **Lemonade on WSL2**: ROCm CLI installs Lemonade on WSL2, but GPU serving
  there hasn't been confirmed on GPU hardware.
- **vLLM on WSL2**: GPU serving has been confirmed only with a pre-release
  ROCm 10.2 SDK.
- **AIM on Windows**: ROCm CLI has no GPU passthrough for AIM containers on
  native Windows.
- **AIM on WSL2**: ROCm CLI passes the WSL2 GPU device to the container, but
  this hasn't been confirmed on GPU hardware.

## ROCm versions, Python, and vLLM

| ROCm version | Python | vLLM |
|---|---|---|
| 7.x | 3.12 | Not verified |
| 10.0 | 3.14 (default) | Not verified |
| 10.1 | 3.14 (default) | Supported |
| 10.2 | 3.14 (default) | Partial |
| Nightly | 3.14 (default) | Not supported |

- **ROCm 7.x**: ROCm CLI installs the same vLLM build, made for ROCm 7.2.3, on
  any 7.x runtime. It hasn't been confirmed on other 7.x versions. A ROCm 7
  install accepts Python 3.12 only.
- **ROCm 10.2**: vLLM installs and imports against a ROCm 10.2 SDK. It has
  served a model on a GPU under WSL2 only, not yet with a release SDK.
- **Nightly**: `rocm install sdk --channel nightly` installs the newest nightly
  build. The current nightly builds have no vLLM build, so the install stops
  instead of using a build made for another ROCm version.
- **Python on ROCm 10 and newer**: vLLM needs Python 3.14. If you don't need
  vLLM, you can pick Python 3.10 through 3.14 with
  `rocm install sdk --python-version`. For
  details, see "Which Python `install sdk` uses" under
  [ROCm installation](commands.md#rocm-installation).

Lemonade and AIM don't need a particular ROCm version on the host:

- **Lemonade**: Lemonade brings its own ROCm build for its llama.cpp backend.
  When the active ROCm SDK is a release version, not a nightly,
  `rocm engines install lemonade` tries to pair that backend with it, and keeps
  the build it had if the pairing can't be verified. Lemonade can also choose
  its Vulkan backend instead, and ROCm CLI then leaves the backend as it is.
- **AIM**: each model runs as a Docker image with its own ROCm stack. From the
  host, AIM needs Docker and a GPU that the image supports (see
  [GPUs](#gpus)).

## GPUs

To check whether a model runs on your computer before you download it, see
[Will a model run here?](commands.md#will-a-model-run-here). Some AMD GPUs have
no ROCm SDK build at all; see "GPUs with no ROCm build" under
[ROCm installation](commands.md#rocm-installation).

Each AIM image lists the GPU families it supports. `rocm model --verbose`
shows those families for each image and how many GPUs the image needs. The AIM
catalog uses these families:

| GPU | Family |
|---|---|
| AMD Instinct™ MI250X | `gfx90a` |
| AMD Instinct MI325X, MI300X | `gfx94X-dcgpu` |
| AMD Instinct MI355X, MI350X | `gfx950-dcgpu` |
| AMD Radeon™ PRO W7900 | `gfx110X-all` |
| AMD Radeon AI PRO R9700 | `gfx120X-all` |

Before it pulls an AIM image, `rocm serve` checks the following:

- **GPU family**: `rocm serve` refuses a GPU outside the image's families. For
  a Radeon image, a board the image doesn't list gets only a warning, because
  one family can cover boards the image wasn't built for.
- **GPU count**: `rocm serve` refuses an image that needs more than one GPU,
  because it runs each engine on a single GPU.
- **Hugging Face token**: `rocm serve` refuses a gated image until you pass
  `HF_TOKEN` with `--engine-env`.

## Next steps

- [Installing ROCm CLI](install/installation.md)
- [ROCm CLI on WSL2](install/wsl.md)
- [vLLM adapter](engines/vllm.md)
- [Command reference](commands.md)
