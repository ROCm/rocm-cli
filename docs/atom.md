# ATOM Adapter

`rocm-engine-atom` is the first-party rocm-cli adapter for AMD's ATOM serving
runtime.

Current behavior:

- Linux/WSL only for ROCm GPU serving.
- Native Windows is explicitly gated and does not fall back to CPU.
- CPU policy is rejected. `gpu_preferred` is treated as `gpu_required`.
- The adapter can use:
  - `ROCM_CLI_ATOM_COMMAND` or `ATOM_COMMAND`
  - `ROCM_CLI_ATOM_PYTHON` or `ATOM_PYTHON`
  - an active rocm-cli managed TheRock runtime if its Python environment has
    the `atom` package installed
- TheRock runtime environment variables and library paths are applied before
  launching ATOM, so non-PyTorch HIP processes can load libraries from the
  managed SDK root.

Live acceptance status:

- Upstream ATOM documentation currently lists AMD Instinct MI355X (`gfx950`),
  MI300X (`gfx942`), and MI250X (`gfx90a`) as supported GPUs.
- This development machine is an RDNA4 Radeon GPU (`gfx1201`), so rocm-cli
  does not claim live ATOM GPU acceptance on this host.
- The adapter remains packaged and test-covered for protocol behavior,
  Windows no-fallback gating, CPU-policy rejection, and managed TheRock
  environment propagation. Live acceptance should be run only on a supported
  ATOM GPU/runtime/model combination.

Acceptance harness:

```bash
python -m py_compile scripts/atom_therock_gpu_test.py
python scripts/atom_therock_gpu_test.py --self-test
```

On supported Linux/WSL ATOM hardware with ATOM installed in the active
rocm-cli managed TheRock runtime:

```bash
python3 scripts/atom_therock_gpu_test.py \
  --engine /home/jam/.cache/rocm-cli-target/debug/rocm-engine-atom \
  --model Qwen/Qwen3-0.6B
```

The harness defaults to the active exact runtime key. An explicit
`--runtime-id` may be an exact runtime key or an unambiguous runtime id, but it
never picks the newest manifest on ambiguity. It rejects
`ROCM_CLI_ATOM_COMMAND`, `ATOM_COMMAND`, `ROCM_CLI_ATOM_PYTHON`, and
`ATOM_PYTHON` so live acceptance proves the managed TheRock runtime path. It
also checks `cpu_only` rejection, `gpu_required` launch state, OpenAI-compatible
serving, managed TheRock environment variables, and loaded ROCm HIP/math
libraries from the managed SDK wheel roots.

The upstream ATOM serving command documented by ROCm is:

```bash
python -m atom.entrypoints.openai_server --model Qwen/Qwen3-0.6B --kv_cache_dtype fp8
```

rocm-cli currently launches the same Python module form and passes `--model`,
`--host`, and `--port`.

Useful checks:

```bash
rocm-engine-atom detect
rocm-engine-atom capabilities
rocm-engine-atom resolve-model Qwen/Qwen3-0.6B
```

Sources:

- https://github.com/ROCm/ATOM
- https://rocm.github.io/ATOM/docs/
