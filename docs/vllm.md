<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# vLLM Adapter

`rocm-engine-vllm` is a first-party adapter around an existing vLLM
installation. It is intended for Linux and WSL ROCm GPU serving.

The adapter does not install vLLM automatically and does not run CPU mode.
Install or build vLLM in a ROCm-capable Python environment first, then make the
`vllm` command visible to rocm-cli.

For rocm-cli managed TheRock runtimes, prefer building vLLM from source against
the existing TheRock PyTorch stack. A prebuilt vLLM ROCm wheel can replace the
TheRock torch packages or target a different ROCm soname set; that is not a
valid no-fallback setup for rocm-cli GPU serving.

## Torch alignment on engine install

Installing an engine into a managed TheRock runtime can change the torch in that
runtime. Two installers write torch into the same environment — the SDK install
writes TheRock's build, and the engine install then writes the build from its own
index — so `rocm engines install` settles which one stays and prints the result
as a `torch_alignment:` line.

A torch that already executes a GPU kernel against the installed SDK is kept
exactly as it is, whichever installer put it there. Otherwise the runtime is
moved to the SDK's *build* of the torch *release* the engine pins: the release
comes from the engine, which was built against it, and the build comes from the
SDK, whose libraries it has to load. A `device_check:` line reports what the
result can actually do, and a realignment also reports what the runtime could do
before it.

Set `ROCM_CLI_DISABLE_TORCH_ALIGNMENT` to keep whatever torch is installed and
skip the replacement:

```bash
ROCM_CLI_DISABLE_TORCH_ALIGNMENT=1 rocm engines install vllm --yes
```

Any value works, including an empty one — the variable being set is the signal.
The install then reports `torch_alignment: disabled`, naming both the build it
would have installed and the one it kept. The device check still runs, so an
opt-out that leaves the runtime unable to serve says so rather than failing later
during serving.

Use it when you are deliberately running a torch the alignment would replace — a
locally built wheel, a version under test, a stack pinned for a reproduction. It
is an escape hatch, not a supported configuration: the resulting combination is
not validated against the supported matrix, and a runtime that cannot execute a
kernel will fail at serving time.

Supported discovery paths:

- `ROCM_CLI_VLLM_COMMAND=/path/to/vllm`
- `ROCM_CLI_VLLM_PYTHON=/path/to/python` where a sibling `vllm` command exists
- the active rocm-cli managed TheRock runtime, if vLLM has been installed into
  that Python environment
- `vllm` on `PATH`

Useful checks:

```bash
rocm-engine-vllm detect
rocm-engine-vllm capabilities
rocm-engine-vllm resolve-model Qwen/Qwen3.5-4B --device-policy gpu_required
python scripts/vllm_therock_gpu_test.py --self-test
```

GPU acceptance check:

```bash
python3 scripts/vllm_therock_gpu_test.py \
  --engine target/debug/rocm-engine-vllm \
  --model facebook/opt-125m
```

The acceptance script is Linux/WSL only. It requires vLLM to be discoverable
through a rocm-cli managed TheRock runtime manifest, launches with
`gpu_required`, checks `/health` and `/v1/completions`, and verifies loaded
ROCm libraries come from the managed TheRock SDK wheel directories. It rejects
external vLLM command overrides and does not allow CPU fallback. It defaults
to the active exact runtime key; if `--runtime-id` is passed, use an exact
runtime key or an unambiguous runtime id.

On WSL, the tested source build needed vLLM ROCm platform detection to use
TheRock PyTorch device data when `amdsmi` is unavailable, and needed vLLM's
ROCm GPTQ half-atomic compatibility path enabled for TheRock 7.13 headers.

On the MI300X/gfx942 TheRock 7.13 runtime, current vLLM source required the
GPTQ compatibility guard in
`csrc/libtorch_stable/quantization/gptq/compat.cuh` to include HIP 7.13:

```diff
-    (defined(USE_ROCM) && (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) < 713)
+    (defined(USE_ROCM) && (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) <= 713)
```

Without that patch, `q_gemm.hip` fails to compile because TheRock 7.13 headers
do not expose the `half`/`half2` `atomicAdd` overloads used by vLLM's GPTQ
kernel. With the patch, the live acceptance harness passed on
`facebook/opt-125m` and verified HIP/BLAS libraries loaded from the managed
TheRock SDK wheel directories.

Serving through rocm-cli:

```bash
rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed
```

### GPU selection

Use `--gpu` to choose the AMD GPU vLLM runs on:

```bash
# Default: first free GPU (auto)
rocm serve Qwen/Qwen3.5-4B --engine vllm --managed

# Pin a specific GPU
rocm serve Qwen/Qwen3.5-4B --engine vllm --gpu 1 --managed
```

rocm-cli pins the device via `HIP_VISIBLE_DEVICES`. Serving one model across
multiple GPUs is not supported.

### GPU memory

vLLM claims a fixed fraction of each GPU's **total** VRAM — not of the free
VRAM, and not scaled to the model — for weights plus KV cache. On a large card
a small model therefore still reserves a large slice.

rocm-cli sets no `--gpu-memory-utilization` of its own, so vLLM's own default
applies unless a value comes from somewhere else — either a model's catalog
recipe or, taking precedence over it, the flag below:

```bash
rocm serve <model> --engine vllm --gpu-memory-utilization 0.3 --managed
```

The value is a fraction in `(0, 1]` of total device VRAM. Lower it to leave room
for a display, another workload, or a second server; raise it to give a large
model more KV cache. Applies to vLLM only — it is ignored, with a note in the
serve output, for other engines. An out-of-range or unparsable value fails the
command rather than falling back silently.

Earlier releases pinned this to `0.80` to leave display/WSL headroom. That pin is
gone, so an unchanged command now reserves vLLM's own (higher) default. Pass
`--gpu-memory-utilization 0.8` to restore the previous reservation.

#### Shared or busy GPUs

Because the reservation is a fraction of **total** VRAM, it ignores memory
already held by other workloads. On a shared multi-GPU node the default can
collide with in-use memory and the engine fails with `HIP out of memory` even for
a tiny model. rocm-cli helps in three ways:

- **Auto-selection avoids busy cards.** `--gpu auto` ranks GPUs by free VRAM and
  skips heavily-used ones. When `rocm-smi`/`amd-smi` are not installed it falls
  back to the amdgpu DRM sysfs counters
  (`/sys/class/drm/card*/device/mem_info_vram_{total,used}`), so selection still
  works on stripped-down container images.
- **The serve summary warns on low free VRAM.** When the selected GPU is already
  heavily used, `rocm serve` prints a note before launch and, for vLLM, points at
  `--gpu-memory-utilization` as the fix.
- **OOM failures hint the workaround.** When a startup failure log shows an
  out-of-memory error, the failure message suggests retrying with a smaller
  reservation, e.g. `--gpu-memory-utilization 0.1`, or targeting a less-busy GPU
  with `--gpu <index>`.

Explicitly, the workaround for an OOM on a shared card is:

```bash
rocm serve <model> --gpu-memory-utilization 0.1
# optionally target a less-busy GPU
rocm serve <model> --gpu 3 --gpu-memory-utilization 0.1
```

### Tool calling

The TUI chat tab attaches tool definitions to every chat request. vLLM rejects
those with HTTP 400 unless it is launched with `--enable-auto-tool-choice` **and**
a matching `--tool-call-parser`. vLLM does not auto-detect the parser and it is
model-specific, so rocm-cli never guesses one:

- **Built-in catalog models** carry the correct parser in their recipe metadata,
  so tool calling works out of the box (e.g. Qwen family → `hermes`,
  Llama&nbsp;3 → `llama3_json`).
- **Other models** (arbitrary Hugging Face repos, or a catalog model forced onto
  vLLM without authored metadata) need an explicit parser:

  ```bash
  rocm serve <model> --engine vllm --tool-call-parser hermes --managed
  ```

  `--tool-call-parser` implies `--enable-auto-tool-choice`, overrides any catalog
  default, and applies to vLLM only. Common values: `hermes`, `llama3_json`,
  `mistral`. Without it, plain chat still works but tool calls return HTTP 400.

Native Windows vLLM serving is skipped in this adapter. Use WSL/Linux for vLLM
ROCm serving, or choose a different engine explicitly. No CPU fallback is used.

References:

- vLLM ROCm installation: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- AMD ROCm vLLM guidance: https://rocmdocs.amd.com/en/latest/how-to/rocm-for-ai/inference/deploy-your-model.html
