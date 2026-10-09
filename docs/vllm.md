<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# vLLM adapter

`rocm-engine-vllm` is a first-party adapter around an existing vLLM
installation. It is intended for Linux and WSL ROCm GPU serving.

The adapter does not install vLLM automatically and does not run CPU mode.
Install or build vLLM in a ROCm-capable Python environment first, then make the
`vllm` command visible to ROCm CLI.

Native Windows vLLM serving is skipped in this adapter. Use WSL or Linux for vLLM
ROCm serving, or choose a different engine explicitly. No CPU fallback is used.

For ROCm CLI-managed TheRock runtimes, prefer building vLLM from source against
the existing TheRock PyTorch stack. A prebuilt vLLM ROCm wheel can replace the
TheRock torch packages or target a different ROCm soname set; that is not a
valid no-fallback setup for ROCm CLI GPU serving.

Building from source compiles HIP sources, so it needs the ROCm compiler
toolchain. That is opt-in: install the runtime with `rocm install sdk --devel`,
or the build fails on a missing compiler. A runtime installed without `--devel`
can still *serve* an already-built vLLM.

## Torch alignment on engine install

Installing an engine into a managed TheRock runtime can change the torch in that
runtime. Two installers write torch into the same environment: the SDK install
writes TheRock's build, and the engine install then writes the build from its own
index. So `rocm engines install` settles which one stays and prints the result
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

Any value works, including an empty one: the variable being set is the signal.
The install then reports `torch_alignment: disabled`, naming both the build it
would have installed and the one it kept. The device check still runs, so an
opt-out that leaves the runtime unable to serve says so instead of failing later
during serving.

Use it when you are deliberately running a torch that the alignment would
replace, such as a locally built wheel, a version under test, or a stack pinned
for a reproduction. It is an escape hatch, not a supported configuration: the
resulting combination is not validated against the supported matrix, and a
runtime that cannot execute a kernel will fail at serving time.

## ROCm 10.x wheel discovery

For most ROCm SDK versions, `rocm engines install vllm` pins a fixed vLLM wheel
and index URL. Any ROCm SDK 10.x version is different: AMD publishes vLLM,
flash-attn, and amd-aiter there under a rotating dev-tag filename (for example
`vllm-0.27.1.dev5+rocm10.0.0.gf46a9dfe2.d20260826-cp314-cp314-linux_x86_64.whl`),
so there is no fixed filename to pin in the adapter.

The wheel's own tag shows the other thing that changes on this route: ROCm
10.x's vLLM, flash-attn, and amd-aiter are published for **`cp314` only**, unlike
every earlier ROCm SDK version's `cp312` wheels. ROCm CLI provisions a matching
interpreter for a 10.x install automatically. A Python already on `PATH`, or
already installed as ROCm CLI's managed Python, is rejected if it is not
`cp314`, with a message naming the required tag.

This route is selected by `major.minor`, so `10.0.0` and `10.1.0` discover
through *different* rows: they pin different vLLM minors and read vLLM from
different index URLs, because AMD stages ROCm 10.1's vLLM builds on a separate
host from 10.0's production index. Both rows take their torch stack from the
same `whl-next` index, which is where `rocm install sdk` resolves either SDK's
own `+rocmX.Y` torch from, but vLLM's discovery is independent of that: it
resolves its own torch, torchvision, and torchaudio pins fresh from the row's
own static version prefixes, instead of reusing whatever the SDK install
resolved, which might be a different torch version than this row pins. Patch
and any dev or pre-release suffix are still ignored within a row, because AMD
rotates those constantly.

The install resolves each package's current wheel, including torch, from the
row's index with `uv pip install --dry-run --reinstall`, parses the version it
reports it would install, then reinstalls pinned to that exact version.
`tensorizer` is not discovered or pinned this way: it has no ROCm-specific
build, and vLLM's own wheel metadata already declares an exact tensorizer
dependency, so it is left to vLLM's own dependency resolution instead of
risking a conflicting pin of its own. Every resolved pin that carries a
`+rocmX.Y` local version is checked against the SDK's own major.minor before
installing, so an index that happens to serve more than one ROCm line at once
cannot silently install the wrong line's wheel onto this SDK. If AMD's index
has no compatible build for a package, the resolver fails and the install
fails instead of falling back to an unpinned or CPU install.

The final install of vllm, flash-attn, and amd-aiter also resolves vLLM's own
plain-PyPI transitive dependencies (for example, `lm-format-enforcer`), which
AMD's index doesn't host. A generated `uv.toml` sets `ignore-error-codes = [403]`
for that index so `uv` falls through to PyPI for those instead of treating the
index's 403 as fatal.

That same full-dependency resolve can also pull in an unconstrained `torch`
from PyPI, undoing the exact ROCm pin just installed, and transitively an
unconstrained `torchvision` and `torchaudio` with it. The install re-pins all
three back to the exact builds discovered above immediately afterwards.

That same realign also covers AMD's per-GPU-architecture device-kernel plugins
(`amd-torch-device-gfx*`, `amd-torchvision-device-gfx*`): each is versioned in
lockstep with its own base package but resolved independently of it, so the
full-dependency install can leave one ahead of the base package it ships
kernels for. The install reads `uv pip freeze`, and re-pins every such plugin
it finds to its own base package's resolved release. The mismatch is invisible
at install time and only surfaces at serve time, as a HIP "Cannot find Symbol"
crash.

Every other ROCm SDK version, including 7.2.3, keeps using the static pin
table. An SDK version with no matching row there falls back to the table's
default pin, *unless* its major release matches a discovery-table entry. In that
case, guessing the default pin would very likely install an ABI-incompatible
build, so the install fails closed instead. The error message names the
detected version and points at `ROCM_CLI_VLLM_ROCM_INDEX_URL` as the way to
install anyway.

## Discovery paths and checks

Supported discovery paths:

- `ROCM_CLI_VLLM_COMMAND=path_to_vllm`, where `path_to_vllm` is the absolute path to the `vllm` executable
- `ROCM_CLI_VLLM_PYTHON=path_to_python`, where `path_to_python` is the absolute path to a Python interpreter that has a sibling `vllm` command
- the active ROCm CLI-managed TheRock runtime, if vLLM has been installed into
  that Python environment
- `vllm` on `PATH`

Useful checks:

```bash
rocm-engine-vllm detect
rocm-engine-vllm capabilities
rocm-engine-vllm resolve-model Qwen/Qwen3.5-4B --device-policy gpu_required
python scripts/vllm_therock_gpu_test.py --self-test
```

## GPU acceptance check

Run the acceptance script against a built adapter:

```bash
python3 scripts/vllm_therock_gpu_test.py \
  --engine target/debug/rocm-engine-vllm \
  --model facebook/opt-125m
```

The acceptance script is Linux or WSL only. It does the following:

- Requires vLLM to be discoverable through a ROCm CLI-managed TheRock runtime
  manifest.
- Launches with `gpu_required`.
- Checks `/health` and `/v1/completions`.
- Verifies that loaded ROCm libraries come from the managed TheRock SDK wheel
  directories.

It rejects external vLLM command overrides and does not allow CPU fallback. It
defaults to the active exact runtime key. If you pass `--runtime-id`, use an
exact runtime key or an unambiguous runtime id.

### Source build notes

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
`facebook/opt-125m` and verified HIP and BLAS libraries loaded from the managed
TheRock SDK wheel directories.

## Serve a model

Serve a model through ROCm CLI:

```bash
rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed
```

## GPU selection

Use `--gpu` to choose the AMD GPU vLLM runs on:

```bash
# Default: first free GPU (auto)
rocm serve Qwen/Qwen3.5-4B --engine vllm --managed

# Pin a specific GPU
rocm serve Qwen/Qwen3.5-4B --engine vllm --gpu 1 --managed
```

ROCm CLI pins the device through `HIP_VISIBLE_DEVICES`. Serving one model across
multiple GPUs is not supported.

## GPU memory

vLLM claims a fixed fraction of each GPU's **total** VRAM for weights plus KV
cache. The fraction is not based on the free VRAM, and it is not scaled to the
model. On a large card, a small model therefore still reserves a large slice.

ROCm CLI sets no `--gpu-memory-utilization` of its own, so vLLM's own default
applies unless a value comes from somewhere else: either a model's catalog
recipe or, taking precedence over it, the flag below:

```bash
rocm serve <model> --engine vllm --gpu-memory-utilization 0.3 --managed
```

The value is a fraction in `(0, 1]` of total device VRAM. Lower it to leave room
for a display, another workload, or a second server. Raise it to give a large
model more KV cache. It applies to vLLM only and is ignored, with a note in the
serve output, for other engines. An out-of-range or unparsable value fails the
command instead of falling back silently.

Earlier releases pinned this to `0.80` to leave display or WSL headroom. That
pin is gone, so an unchanged command now reserves vLLM's own (higher) default.
Pass `--gpu-memory-utilization 0.8` to restore the previous reservation.

### Shared or busy GPUs

Because the reservation is a fraction of **total** VRAM, it ignores memory
already held by other workloads. On a shared multi-GPU node the default can
collide with in-use memory and the engine fails with `HIP out of memory` even for
a tiny model. ROCm CLI helps in three ways:

- **Auto-selection avoids busy cards.** `--gpu auto` ranks GPUs by free VRAM and
  skips heavily used ones. When `amd-smi` is not installed, it falls back to the
  amdgpu DRM sysfs counters
  (`/sys/class/drm/card*/device/mem_info_vram_{total,used}`), so selection still
  works on stripped-down container images with a single GPU. That fallback
  withholds telemetry on a multi-GPU host, because its `card<N>` numbering is
  not guaranteed to match HIP's device ordinal there.
- **The serve summary warns on low free VRAM.** When the selected GPU is already
  heavily used, `rocm serve` prints a note. The note appears before launch on
  the plain path, or in the post-readiness summary in the default interactive
  mode. For vLLM, the note also points at `--gpu-memory-utilization` as the fix.
- **OOM failures hint the workaround.** When a startup failure log shows an
  out-of-memory error, the failure message suggests retrying with a smaller
  reservation, such as `--gpu-memory-utilization 0.5`, or targeting a less busy
  GPU with `--gpu <index>`. It also points at
  `rocm diagnose --symptom '<the error>'` for the full conditional remediation
  (busy GPU versus a model that does not fit). The printed command quotes your
  actual failing line when it can be rendered as one intact single-quoted
  argument. A line carrying an apostrophe or terminal control bytes falls back to
  the canonical symptom below instead of handing you a command whose quoting the
  log text broke.

Explicitly, the workaround for an OOM on a shared card is:

```bash
rocm serve <model> --engine vllm --gpu-memory-utilization 0.5
# optionally target a specific, less busy GPU by index
rocm serve <model> --engine vllm --gpu 1 --gpu-memory-utilization 0.5
# for the full breakdown of a busy GPU versus a model that is too large, pass the error to diagnose
rocm diagnose --symptom 'vllm: torch.OutOfMemoryError: HIP out of memory'
```

## Tool calling

The chat tab in the terminal user interface (TUI) attaches tool definitions to
every chat request. vLLM rejects
those with HTTP 400 unless it is launched with `--enable-auto-tool-choice` **and**
a matching `--tool-call-parser`. vLLM does not auto-detect the parser, and the
parser is model-specific, so ROCm CLI never guesses one:

- **Built-in catalog models** carry the correct parser in their recipe metadata,
  so tool calling works out of the box. For example, the Qwen family uses
  `hermes` and Llama 3 uses `llama3_json`.
- **Other models** (arbitrary Hugging Face repos, or a catalog model forced onto
  vLLM without authored metadata) need an explicit parser:

  ```bash
  rocm serve <model> --engine vllm --tool-call-parser hermes --managed
  ```

  `--tool-call-parser` implies `--enable-auto-tool-choice`, overrides any catalog
  default, and applies to vLLM only. Common values: `hermes`, `llama3_json`,
  `mistral`. Without it, plain chat still works but tool calls return HTTP 400.

## Related resources

- [vLLM ROCm installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [AMD ROCm AI ecosystem: vLLM](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html)
