<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# SGLang Adapter

`rocm-engine-sglang` is a first-party adapter around an existing SGLang
installation. It does not install SGLang automatically and does not run CPU
mode.

Use Linux or WSL with a ROCm-capable SGLang Python environment, then expose one
of these launchers to rocm-cli:

- `ROCM_CLI_SGLANG_COMMAND=/path/to/sglang`
- `ROCM_CLI_SGLANG_PYTHON=/path/to/python`
- an active rocm-cli managed TheRock runtime that contains SGLang
- `sglang` on `PATH`

The adapter launches either:

```bash
sglang serve --model-path <model> --host <host> --port <port> --attention-backend triton
```

or, when only a Python environment is configured:

```bash
python -m sglang.launch_server --model-path <model> --host <host> --port <port> --attention-backend triton
```

The Triton attention backend is the rocm-cli default so basic SGLang serving
does not require a separately built AITER package. AITER remains an upstream
SGLang ROCm dependency for AITER-specific attention, MoE, and quantized paths.

Smoke commands:

```bash
rocm-engine-sglang detect
rocm-engine-sglang capabilities
rocm-engine-sglang resolve-model Qwen/Qwen3.5-4B --device-policy gpu_required
python scripts/sglang_therock_gpu_test.py --self-test
```

Managed serving:

```bash
rocm serve Qwen/Qwen3.5-4B --engine sglang --device gpu_required --managed
```

### GPU selection

Use `--gpu` to choose the AMD GPU SGLang runs on:

```bash
# Default: first free GPU (auto)
rocm serve Qwen/Qwen3.5-4B --engine sglang --managed

# Pin a specific GPU
rocm serve Qwen/Qwen3.5-4B --engine sglang --gpu 1 --managed
```

rocm-cli pins the device via `HIP_VISIBLE_DEVICES`. Serving one model across
multiple GPUs is not supported.

Native Windows SGLang serving is skipped in this adapter. Use WSL/Linux for
SGLang ROCm serving, or choose a different engine explicitly. No CPU fallback is
used.

## TheRock And RDNA4 Status

The adapter can resolve SGLang from a rocm-cli managed TheRock runtime and will
report that as `managed_env: true`. When a managed runtime is used, service
state records the TheRock SDK root/bin paths so acceptance tests can verify
that HIP libraries were loaded from the managed SDK wheel directories.

Live SGLang ROCm acceptance is currently gated on upstream SGLang kernel
support for the host GPU. On this WSL test host (`gfx1201`, Radeon RX 9070 XT),
both SGLang `v0.5.12` and current `origin/main` reject the ROCm kernel build in
`sgl-kernel/setup_rocm.py` with a supported-architecture check for `gfx942` and
`gfx950` only. rocm-cli must not force a different target or fall back to CPU
mode. Re-run live SGLang GPU acceptance only on a supported SGLang ROCm target,
or after upstream adds support for the host gfx target.

On the MI300X/gfx942 TheRock 7.13 runtime, SGLang `v0.5.12.post1` was installed
from source with `python/pyproject_other.toml`, `sgl-kernel/setup_rocm.py`, and
the active TheRock SDK compiler/library paths. The generic PyPI package is not a
safe automatic install path because it can resolve CUDA/NVIDIA packages.

The AITER-free MI300X smoke path also needed source guards so optional Quark
quantization imports do not require AITER for unquantized models, and needed
SGLang's HIP layernorm path to use its native fallback instead of the vLLM
RMSNorm op when AITER is absent. With those source adjustments plus the adapter
Triton attention default, the live harness passed on
`Qwen/Qwen2.5-1.5B-Instruct` and verified HIP/BLAS libraries loaded from the
managed TheRock SDK wheel directories.

On a supported ROCm target, run the live acceptance harness from the repo root:

```bash
python3 scripts/sglang_therock_gpu_test.py \
  --engine /home/user/.cache/rocm-cli-target/debug/rocm-engine-sglang \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

The harness uses the active rocm-cli managed TheRock runtime by default,
rejects external SGLang command/Python overrides, launches with
`gpu_required`, and verifies loaded HIP libraries came from the managed SDK
wheel directories.

References:

- SGLang launch server: https://sgl-project.github.io/basic_usage/send_request.html
- SGLang serve command: https://sgl-project-sglang-93.mintlify.app/backend/launch-server
