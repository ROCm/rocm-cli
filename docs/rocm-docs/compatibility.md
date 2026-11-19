<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# ROCm CLI compatibility

This page shows which inference engines run on which platforms and ROCm
versions, and which Python each ROCm version uses. It lists support status
only. Exact package versions and the details behind each status are on the
linked pages.

<!--
Site-only page: no source file holds these tables. The platform table is
included from README.md (between the platform-support-table markers), not
copied. Keep this page status-only: exact package versions live in
docs/vllm.md and the engine code, so they are not repeated here.
-->

The tables use these statuses:

- **Supported**: ROCm CLI installs the engine for this combination, and it has
  run on GPU hardware.
- **Partial**: ROCm CLI installs the engine, but GPU serving is confirmed only in
  a limited setup. The note under the table says which.
- **Not verified**: ROCm CLI has an install path, but it hasn't been confirmed on
  GPU hardware.
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
| Lemonade | Supported | Supported | Supported |
| vLLM | Supported | Not supported | Supported |
| AIM | Supported (needs Docker) | Not supported | Not verified |

- **Lemonade** chooses its own llama.cpp backend, ROCm or Vulkan. When it
  chooses Vulkan, ROCm CLI doesn't pair the backend with the ROCm SDK.
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
| 10.3 (nightly) | 3.14 (default) | Not supported |

- **ROCm 7.x**: ROCm CLI has one vLLM build for ROCm 7, built for ROCm 7.2.3,
  and installs it on any 7.x runtime. It hasn't been confirmed on other 7.x
  versions. A ROCm 7 install accepts Python 3.12 only.
- **ROCm 10.0**: ROCm CLI has an install path for this version, but it hasn't
  been confirmed on GPU hardware.
- **ROCm 10.2**: vLLM installs and imports against a ROCm 10.2 SDK. A GPU serve
  has run on WSL2 only, not yet against a release SDK.
- **ROCm 10.3 (nightly)**: no vLLM build exists for this version. The install
  stops instead of using a build made for another ROCm version.
- **Python on ROCm 10 and newer**: vLLM needs Python 3.14. Without vLLM, you can
  pick Python 3.10 to 3.14 with `rocm install sdk --python-version`. For
  details, see "Which Python `install sdk` uses" under
  [ROCm installation](commands.md#rocm-installation).

Lemonade and AIM don't depend on the ROCm version installed on the host:

- **Lemonade** brings its own ROCm build for its llama.cpp backend. When the
  active ROCm SDK is a release version, not a nightly,
  `rocm engines install lemonade` tries to pair that backend with it, and keeps
  the build it had if the pairing can't be verified.
- **AIM** runs each model as a Docker image with its own ROCm stack. What it
  needs from the host is Docker and a GPU that the image supports (see
  [GPUs](#gpus)).

## GPUs

To check whether a model runs on this computer before downloading it, see
[Will a model run here?](commands.md#will-a-model-run-here). Some AMD GPUs have
no ROCm SDK build at all; see "GPUs with no ROCm build" under
[ROCm installation](commands.md#rocm-installation).

AIM support is set per image. `rocm model --verbose` shows the GPU families
each AIM image supports and how many GPUs it needs. The families map to these
GPUs:

| GPU | Family |
|---|---|
| AMD Instinct MI250X | `gfx90a` |
| AMD Instinct MI300X, MI325X | `gfx94X-dcgpu` |
| AMD Instinct MI350X, MI355X | `gfx950-dcgpu` |
| AMD Radeon PRO W7900 | `gfx110X-all` |
| AMD Radeon AI PRO R9700 | `gfx120X-all` |

Before it pulls an AIM image, `rocm serve` checks the following:

- **GPU family**: a GPU outside the image's families is refused. For Radeon
  images, a board the image doesn't list gets a warning only, because one
  family covers boards the image wasn't built for.
- **GPU count**: an image that needs more than one GPU is refused, because
  `rocm serve` runs an engine on one GPU.
- **Hugging Face token**: a gated image is refused until you pass `HF_TOKEN`
  with `--engine-env`.

## Next steps

- [Installing ROCm CLI](install/installation.md)
- [ROCm CLI on WSL2](install/wsl.md)
- [vLLM adapter](engines/vllm.md)
- [Command reference](commands.md)
