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
ROCm GPTQ half-atomic compatibility path enabled for TheRock 7.13 headers. The
adapter passes `--gpu-memory-utilization 0.80` by default so display/WSL VRAM
use does not prevent a small GPU model from starting.

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

Native Windows vLLM serving is skipped in this adapter. Use WSL/Linux for vLLM
ROCm serving, or choose a different engine explicitly. No CPU fallback is used.

References:

- vLLM ROCm installation: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- AMD ROCm vLLM guidance: https://rocmdocs.amd.com/en/latest/how-to/rocm-for-ai/inference/deploy-your-model.html
