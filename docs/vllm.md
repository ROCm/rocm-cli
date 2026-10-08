<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# vLLM adapter

`rocm-engine-vllm` is a first-party adapter around an existing vLLM
installation. It is intended for Linux and WSL2 ROCm GPU serving.

## Install vLLM

`rocm serve` does not install vLLM, and the adapter does not run CPU mode.
Install vLLM into a ROCm-capable Python environment first, then make the `vllm`
command visible to ROCm CLI. You can do that in one of these ways:

- **Automatically:** at the end of `rocm install sdk`, ROCm CLI installs vLLM
  when the GPU family prefers vLLM. If that install fails, `rocm install sdk`
  still exits with status 0 and prints a warning that points to
  `rocm engines install vllm --runtime-id <id>`.
- **Manually:** run `rocm engines install vllm [--yes]`. Use this after a failed
  automatic install, or when you force `--engine vllm` on a GPU that prefers
  Lemonade.
- **Your own vLLM:** build or install vLLM yourself and point ROCm CLI at it. See
  [Discovery paths](#discovery-paths-and-checks).

Native Windows vLLM serving is skipped in this adapter. Use WSL2 or Linux for
vLLM ROCm serving, or choose a different engine explicitly. No CPU fallback is
used.

For ROCm CLI-managed TheRock runtimes, build vLLM from source against the
existing TheRock PyTorch stack. A prebuilt vLLM ROCm wheel can replace the
TheRock torch packages or target a different ROCm soname set. That is not a valid
no-fallback setup for ROCm CLI GPU serving.

Building from source compiles HIP sources, so it needs the ROCm compiler
toolchain. The toolchain is opt-in: install the runtime with
`rocm install sdk --devel`, or the build fails on a missing compiler. A runtime
installed without `--devel` can still *serve* an already-built vLLM.

## Torch alignment on engine install

Installing an engine into a managed TheRock runtime can change the torch in that
runtime. Two installers write torch into the same environment: the SDK install
writes TheRock's build, and the engine install then writes the build from its own
index. `rocm engines install` settles which one stays and prints the result as a
`torch_alignment:` line.

- A torch that already executes a GPU kernel against the installed SDK is kept
  exactly as it is, whichever installer put it there.
- Otherwise, the runtime moves to the SDK's *build* of the torch *release* that
  the engine pins. The release comes from the engine, which was built against it.
  The build comes from the SDK, whose libraries it has to load.

A `device_check:` line reports what the result can do. A realignment also reports
what the runtime could do before it.

### Disable torch alignment

Set `ROCM_CLI_DISABLE_TORCH_ALIGNMENT` to keep whatever torch is installed and
skip the replacement:

```bash
ROCM_CLI_DISABLE_TORCH_ALIGNMENT=1 rocm engines install vllm --yes
```

Any value works, including an empty one. The variable being set is the signal.

The install then reports `torch_alignment: disabled`, naming both the build it
would have installed and the one it kept. The device check still runs. If the
opt-out leaves the runtime unable to serve, the install says so instead of
failing later during serving.

Use the variable when you are deliberately running a torch that the alignment
would replace, such as a locally built wheel, a version under test, or a stack
pinned for a reproduction. It is an escape hatch, not a supported configuration.
The resulting combination is not validated against the supported matrix, and a
runtime that cannot execute a kernel fails at serving time.

## ROCm 10.x wheel discovery

For most ROCm SDK versions, `rocm engines install vllm` pins a fixed vLLM wheel
and index URL. Any ROCm SDK 10.x version is different. AMD publishes vLLM,
flash-attn, and amd-aiter there under a rotating dev-tag filename, so there is no
fixed filename to pin in the adapter. An example filename is
`vllm-0.27.1.dev5+rocm10.0.0.gf46a9dfe2.d20260826-cp314-cp314-linux_x86_64.whl`.

### Python version

ROCm 10.x's vLLM, flash-attn, and amd-aiter wheels are published for **`cp314`
only**, unlike the `cp312` wheels of every earlier ROCm SDK version.

- ROCm CLI provisions a matching interpreter for a 10.x install automatically.
- A Python that is already on `PATH`, or already installed as ROCm CLI's managed
  Python, is rejected if it is not `cp314`. The message names the required tag.

### Version rows and indexes

This route is selected by `major.minor`, so `10.0.0` and `10.1.0` discover
through *different* rows:

- They pin different vLLM minors.
- They read vLLM from different index URLs, because AMD stages ROCm 10.1's vLLM
  builds on a separate host from 10.0's production index.
- Both rows take their torch stack from the same `whl-next` index. That is where
  `rocm install sdk` resolves either SDK's own `+rocmX.Y` torch from.

vLLM's discovery is independent of the SDK install. It resolves its own
torch, torchvision, and torchaudio pins from the row's own static version
prefixes, instead of reusing what the SDK install resolved. The two can differ.
The patch version and any dev or pre-release suffix are ignored within a row,
because AMD rotates them constantly.

### How the install resolves wheels

The install resolves each package's current wheel, including torch, from the
row's index with `uv pip install --dry-run --reinstall`. It parses the version
that the dry run reports, then reinstalls pinned to that exact version.

- **tensorizer:** It is not discovered or pinned this way. It has no ROCm-specific
  build, and vLLM's own wheel metadata already declares an exact tensorizer
  dependency. The install leaves it to vLLM's dependency resolution to avoid a
  conflicting pin.
- **ROCm line check:** Every resolved pin that carries a `+rocmX.Y` local version
  is checked against the SDK's own `major.minor` before installing. An index that
  serves more than one ROCm line at once therefore cannot silently install the
  wrong line's wheel onto this SDK.
- **No compatible build:** If AMD's index has no compatible build for a package,
  the resolver fails and the install fails. It does not fall back to an unpinned
  or CPU install.
- **PyPI dependencies:** The final install of vLLM, flash-attn, and amd-aiter also
  resolves vLLM's plain-PyPI transitive dependencies, such as
  `lm-format-enforcer`, which AMD's index doesn't host. A generated `uv.toml` sets
  `ignore-error-codes = [403]` for that index, so `uv` falls through to PyPI for
  those packages instead of treating the 403 from the index as fatal.

### Re-pinning after the full install

The full-dependency resolve can pull in an unconstrained `torch` from PyPI, which
undoes the exact ROCm pin that was installed a moment earlier. It can also pull in an
unconstrained `torchvision` and `torchaudio`. The install re-pins all three to the
exact builds discovered earlier.

The same re-pin covers AMD's per-GPU-architecture device-kernel plugins
(`amd-torch-device-gfx*` and `amd-torchvision-device-gfx*`). Each plugin is
versioned in lockstep with its own base package but resolved independently of it,
so the full-dependency install can leave one ahead of the base package it ships
kernels for. The install reads `uv pip freeze` and re-pins every such plugin it
finds to its own base package's resolved release.

The mismatch is invisible at install time. It only surfaces at serve time, as a
HIP "Cannot find Symbol" crash.

### Other ROCm versions

Every other ROCm SDK version, including the 7.14 default and 7.2.3, keeps using
the static pin table. An SDK version with no matching row falls back to the
table's default pin.

There is one exception. If the version's major release matches a discovery-table
entry, guessing the default pin would likely install an ABI-incompatible build.
The install then fails closed with a message that names the detected version and
points to `ROCM_CLI_VLLM_ROCM_INDEX_URL` as the way to install anyway.

## Discovery paths and checks

ROCm CLI looks for the `vllm` command in this order:

- `ROCM_CLI_VLLM_COMMAND=<path to vllm>`, where the value is the absolute path to
  the `vllm` executable
- `ROCM_CLI_VLLM_PYTHON=<path to python>`, where the value is the absolute path to
  a Python interpreter that has a sibling `vllm` command
- The active ROCm CLI-managed TheRock runtime, if vLLM has been installed into
  that Python environment
- `vllm` on `PATH`

To check what the adapter finds, run:

```bash
rocm-engine-vllm detect
rocm-engine-vllm capabilities
rocm-engine-vllm resolve-model Qwen/Qwen3.5-4B --device-policy gpu_required
```

## Serve a model

Serve a model through ROCm CLI:

```bash
rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required
```

By default, the server runs in the background under ROCm CLI's supervision. See
[Model serving](https://github.com/ROCm/rocm-cli/blob/main/README.md#model-serving)
for the full set of `rocm serve` options.

## GPU selection

Use `--gpu` to choose the AMD GPU that vLLM runs on:

```bash
# Default: first free GPU (auto)
rocm serve Qwen/Qwen3.5-4B --engine vllm

# Pin a specific GPU
rocm serve Qwen/Qwen3.5-4B --engine vllm --gpu 1
```

ROCm CLI pins the device with `HIP_VISIBLE_DEVICES`. Serving one model across
multiple GPUs is not supported.

## GPU memory

vLLM claims a fixed fraction of each GPU's **total** VRAM for weights plus KV
cache. The fraction is not of the free VRAM, and it is not scaled to the model.
On a large card, a small model therefore still reserves a large slice.

ROCm CLI sets no `--gpu-memory-utilization` of its own, so vLLM's own default
applies unless a value comes from somewhere else. A value can come from a model's
catalog recipe or from the flag below. The flag takes precedence over the recipe:

```bash
rocm serve <model> --engine vllm --gpu-memory-utilization 0.3
```

The value is a fraction in `(0, 1]` of total device VRAM.

- Lower it to leave room for a display, another workload, or a second server.
- Raise it to give a large model more KV cache.
- The flag applies to vLLM only. For other engines, it is ignored, with a note in
  the serve output.
- An out-of-range or unparsable value fails the command instead of falling back
  silently.

**Note:** ROCm CLI no longer pins this value to `0.80`, which earlier releases
used to leave display and WSL2 headroom. An unchanged command now reserves vLLM's
own, higher default. Pass `--gpu-memory-utilization 0.8` to restore the previous
reservation.

### Shared or busy GPUs

The reservation is a fraction of **total** VRAM, so it ignores memory that other
workloads already hold. On a shared multi-GPU node, the default can collide with
memory in use, and the engine fails with `HIP out of memory` even for a tiny
model. ROCm CLI helps in three ways:

- **Auto-selection avoids busy cards.** `--gpu auto` ranks GPUs by free VRAM and
  skips heavily used ones. When `amd-smi` is not installed, it falls back to the
  amdgpu DRM sysfs counters
  (`/sys/class/drm/card*/device/mem_info_vram_{total,used}`), so selection still
  works on stripped-down container images with a single GPU. On a multi-GPU host,
  that fallback withholds telemetry, because its `card<N>` numbering is not
  guaranteed to match HIP's device ordinal.
- **The serve summary warns on low free VRAM.** When the selected GPU is already
  heavily used, `rocm serve` prints a note. In the plain path, the note appears
  before launch. In the default interactive mode, it appears in the post-readiness
  summary. For vLLM, the note points to `--gpu-memory-utilization` as the fix.
- **OOM failures suggest a workaround.** When a startup failure log shows an
  out-of-memory error, the failure message suggests two things. One is retrying
  with a smaller reservation, such as `--gpu-memory-utilization 0.5`. The other is
  targeting a less busy GPU with `--gpu <index>`. It also points to
  `rocm diagnose --symptom '<the error>'` for the full conditional remediation,
  which separates a busy GPU from a model that does not fit.

  The printed command quotes your actual failing line when it can be rendered as
  one intact single-quoted argument. If the line contains an apostrophe or
  terminal control bytes, the command uses the canonical symptom shown below
  instead, so that log text can't break the quoting.

To work around an OOM on a shared card, run:

```bash
rocm serve <model> --engine vllm --gpu-memory-utilization 0.5
# optionally target a specific, less busy GPU by index
rocm serve <model> --engine vllm --gpu 1 --gpu-memory-utilization 0.5
# for the full breakdown of busy GPU versus model too large, pass the error to diagnose
rocm diagnose --symptom 'vllm: torch.OutOfMemoryError: HIP out of memory'
```

## Tool calling

The chat tab in the terminal user interface (TUI) attaches tool definitions to
every chat request. vLLM rejects those requests with HTTP 400 unless it is
launched with `--enable-auto-tool-choice` **and** a matching
`--tool-call-parser`. vLLM does not auto-detect the parser, and the parser is
model-specific, so ROCm CLI never guesses one:

- **Built-in catalog models** carry the correct parser in their recipe metadata,
  so tool calling works without extra flags. For example, the Qwen family uses
  `hermes` and Llama 3 uses `llama3_json`.
- **Other models** need an explicit parser. This includes arbitrary Hugging Face
  repositories and a catalog model that is forced onto vLLM without authored
  metadata:

  ```bash
  rocm serve <model> --engine vllm --tool-call-parser hermes
  ```

  `--tool-call-parser` implies `--enable-auto-tool-choice`, overrides any catalog
  default, and applies to vLLM only. Common values are `hermes`, `llama3_json`,
  and `mistral`. Without it, plain chat still works, but tool calls return
  HTTP 400.

## Contributor checks

This section is for people who build and test the adapter from a ROCm CLI source
checkout. It uses scripts that are not part of an installed ROCm CLI.

### GPU acceptance check

Run the acceptance script against a built adapter:

```bash
python3 scripts/vllm_therock_gpu_test.py \
  --engine target/debug/rocm-engine-vllm \
  --model facebook/opt-125m
```

To check that the script itself works, run
`python scripts/vllm_therock_gpu_test.py --self-test`.

The acceptance script runs on Linux or WSL2 only. It does the following:

- Requires vLLM to be discoverable through a ROCm CLI-managed TheRock runtime
  manifest.
- Launches with `gpu_required` and checks `/health` and `/v1/completions`.
- Verifies that the loaded ROCm libraries come from the managed TheRock SDK wheel
  directories.
- Rejects external vLLM command overrides and does not allow CPU fallback.

The script defaults to the active exact runtime key. If you pass `--runtime-id`,
use an exact runtime key or an unambiguous runtime ID.

### Source build notes

These notes record what was needed when building vLLM against TheRock 7.13. They
might not apply to later TheRock releases.

On WSL2, the tested source build needed two changes:

- vLLM's ROCm platform detection had to use TheRock PyTorch device data when
  `amdsmi` is unavailable.
- vLLM's ROCm GPTQ half-atomic compatibility path had to be enabled for TheRock
  7.13 headers.

On the MI300X (gfx942) TheRock 7.13 runtime, the vLLM source at the time required
the GPTQ compatibility guard in
`csrc/libtorch_stable/quantization/gptq/compat.cuh` to include HIP 7.13:

```diff
-    (defined(USE_ROCM) && (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) < 713)
+    (defined(USE_ROCM) && (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) <= 713)
```

Without that patch, `q_gemm.hip` fails to compile, because TheRock 7.13 headers
do not expose the `half` and `half2` `atomicAdd` overloads that vLLM's GPTQ kernel
uses. With the patch, the live acceptance harness passed on `facebook/opt-125m`
and verified that the HIP and BLAS libraries load from the managed TheRock SDK
wheel directories.

## Related resources

- [vLLM ROCm installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [AMD ROCm AI ecosystem: vLLM](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html)
