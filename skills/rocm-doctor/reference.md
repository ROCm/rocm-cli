# ROCm Doctor — reference

The closed failure-mode catalog and the CLI it drives. The catalog is
authoritative in the `rocm` CLI (`crates/rocm-core`); this file mirrors it for
humans. To add or change a failure mode, change the CLI catalog — not this doc.

## CLI commands

### `rocm examine [--json]`

Inspect the host: GPU + gfx target, driver, ROCm install, render/video groups,
`/dev/kfd` + render devices, kernel modules, framework introspection, recent
amdgpu kernel-log evidence. `--json` emits the **Examination** document for
tooling.

- It's a general system inspector, so it **always exits 0**. The verdict is the
  `status` field: `ok` · `no-amd-gpu` · `wsl` · `unsupported-os` · `degraded`.

### `rocm diagnose [--symptom "<text>"] [--top N] [--json]`

Match the host + symptom against the closed catalog. **Always exits 0**; read
the result from `--json`:

- `matched[]` — ranked `{ id, title, score, evidence[], fix }`. Tiers:
  `>= 75` high confidence, `50–74` likely, `< 50` weak.
- `has_match` — whether any entry cleared `min_score_for_match`. **This is the
  gate, not whether `matched` is empty.** Several checkers open with a nonzero
  base score for a merely *potentially* relevant situation (running in a
  container, an APU alongside a discrete GPU), so a healthy host produces a
  non-empty `matched` full of sub-threshold entries. Gating on emptiness
  proposes a fix for a machine with nothing wrong and never routes upstream.
- `min_score_for_match` (50), `high_confidence_threshold` (75).
- `out_of_scope` — set when the host is off-catalog (e.g. WSL2); `matched` is empty.
- `route_when_no_match` — `{ target, url }` upstream tracker to use when `has_match` is false.

### `rocm fix [<id>] [--yes] [--dry-run] [--device-index N]`

Apply a fix by id (run with no id to list). Exit codes:

| code | meaning |
| --- | --- |
| 0 | applied / dry-run / print-only plan / list |
| 1 | internal error |
| 2 | usage error (incl. unknown fix-id) |
| 3 | not applicable on this host (OS mismatch, negative `--device-index`) — nothing changed |
| 4 | attempted but the command failed |
| 5 | user declined at the prompt |

Only four fixes are auto-applicable — `fix-2-unset-override`,
`fix-4-render-group`, `fix-6-path`, `fix-9-igpu-dgpu` — and the rest print their
plan for the user to run. Pass the **full** id (`rocm fix fix-2-unset-override`,
not `rocm fix fix-2`; a short id returns exit 2, unknown fix-id).

**Auto-fix** in the catalog below means only what the CLI reports in
`rocm fix`'s listing: the CLI has a runner for it and will carry it out itself,
rather than printing a plan for the user. It is not a promise that the runner
mutates anything.

The ones that do mutate print the exact command, honor `--dry-run`, refuse on a
non-interactive shell without `--yes`, and confirm first. **Two exceptions:**

- **`fix-2-unset-override` mutates on Windows only.** Its Linux runner reports
  where the override is set and which rc files carry it, then stops — it will
  not edit your dotfiles. So on Linux it never prompts, `--dry-run` has nothing
  to preview, and `--yes` is never read. Do not tell a Linux user a dry run
  previewed a change that was never going to happen.
- **`fix-9-igpu-dgpu` mutates only when you pass `--device-index N`.** Without
  it — on Linux and Windows alike — the runner just prints the
  `rocminfo`/`hipInfo` query that identifies which index is the discrete GPU
  and returns 0: no prompt, nothing for `--dry-run` to preview, nothing
  pinned. Run `rocm fix fix-9-igpu-dgpu --device-index N` (not the bare id)
  once you know N.

## Closed catalog (16 failure modes)

| id | OS | Failure mode | Typical signal | Auto-fix |
| --- | --- | --- | --- | --- |
| `fix-1-arch` | both | GPU gfx target not in the framework's build arch list | `hipErrorNoBinaryForGpu`, `HSA_STATUS_ERROR_INVALID_ISA`, "invalid device function" | no |
| `fix-2-unset-override` | both | `HSA_OVERRIDE_GFX_VERSION` set on a GPU that now has a native wheel | page faults / `OUT_OF_REGISTERS`, override set in env | yes |
| `fix-3-rocm-kernel` | linux | ROCm + distro/kernel form an unsupported triple | ROCm installed but `amdgpu` not loaded; DKMS build failure | no |
| `fix-4-render-group` | linux | User not in `render`/`video` group (or `/dev/kfd` owned by the other group) | cannot open `/dev/kfd`, permission denied | yes |
| `fix-5-amdgpu-load` | linux | `amdgpu` module not loaded (or blacklisted) | "ROCk module is NOT loaded", blacklist entry, Secure Boot | no |
| `fix-6-path` | both | ROCm/HIP binaries not on PATH after install | `rocminfo: command not found`, `hipInfo` missing from PATH | yes |
| `fix-7-stale-repos` | linux | Stale/conflicting APT/DNF repos from prior installer runs | apt 404 `repo.radeon.com`, unmet deps, ≥2 ROCm repo files | no |
| `fix-8-wheel-rocm` | both | Framework wheel built for a different ROCm major than the system | `libamdhip64.so.X` / `amdhip64_X.dll` load failure | no |
| `fix-9-igpu-dgpu` | both | iGPU enumerated alongside dGPU, destabilising the runtime | APU + discrete AMD present, `HIP_VISIBLE_DEVICES` unset, crash/segfault | yes |
| `fix-10-container` | linux | Container can't see `/dev/kfd` or `/dev/dri/renderD*` | running in docker/podman, kfd/render devices missing | no |
| `fix-11-iommu` | linux | Multi-GPU hang with IOMMU enabled | ≥2 AMD GPUs, `iommu=` not `pt`, hang/deadlock/timeout | no |
| `fix-12-installer` | linux | `amdgpu-install` left a broken DKMS / repo state | dpkg half-configured, DKMS failed, `--accept-eula` | no |
| `fix-13-hip-sdk-missing` | windows | HIP SDK not installed | no HIP SDK under Program Files, `hipInfo` not recognized | no |
| `fix-14-adrenalin-too-old` | windows | Adrenalin / kernel-mode driver too old for the HIP SDK | `hipInfo` can't enumerate, "driver too old", HSA "no agents found" | no |
| `fix-15-msvc-redist` | windows | MSVC runtime missing (HIP DLLs can't load) | `vcruntime140.dll` / `vcruntime140_1.dll` missing | no |
| `fix-17-torch-dlpack` | linux | `torch-c-dlpack-ext` loads its CUDA prebuilt on a ROCm torch, aborting vLLM's engine start at import time | vLLM engine start fails on import; error names `torch_c_dlpack_ext` or tvm_ffi's `_optional_torch_c_dlpack` | no |

The numbering has a gap: `fix-16` is a reserved handle, not a missing row. Ids
are stable handles rather than positions, so the catalog holds 16 modes
numbered 1-15 and 17.

Linux-only: fix-3, -4, -5, -7, -10, -11, -12, -17. Windows-only: fix-13, -14, -15.
Cross-platform: fix-1, -2, -6, -8, -9.

## Framework routing

`rocm diagnose` diagnoses frameworks built against the **system** ROCm/HIP:

- **PyTorch**, **llama.cpp** — in scope.

Apps that ship their own runtime are routed upstream **by the skill**, from the
table below. The CLI cannot do it: `route_when_no_match` keys off the
host-detected framework, which `rocm examine` only ever reports as `pytorch`,
`llama-cpp`, `unknown` or `skipped`, so it has no arm for these apps.

- **Lemonade** → https://github.com/lemonade-sdk/lemonade/issues
- **Ollama** → https://github.com/ollama/ollama/issues
- **LM Studio** → in-app support (no public repo)

`route_when_no_match` itself returns one of:

- **pytorch** → https://github.com/pytorch/pytorch/issues (tag with the rocm label)
- **llama-cpp** → https://github.com/ggml-org/llama.cpp/issues
- otherwise **rocm-core** → https://github.com/ROCm/ROCm/issues

## Out of scope

- **WSL2** — distinct platform (`/dev/dxg` + Windows host driver). Detected and
  routed out; point at AMD's ROCm-on-WSL guide:
  https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html
- NVIDIA / Intel / Apple Silicon GPUs; fresh installs on a clean machine.
