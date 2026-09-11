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
- `out_of_scope` — set when the host's platform family has **no catalog entries
  at all**, in which case `matched` is empty. Linux, Windows and WSL2 are all
  covered, so this now fires only for a platform outside those three. It means
  nothing was checked — not a clean bill of health.
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

## Closed catalog (23 failure modes)

The OS column is the platform family the CLI scopes an entry to, and WSL2 is a
family of its own — not a flavour of `linux`. An entry reaches a WSL host only
by naming `wsl`, so the bare-metal Linux entries (the `amdgpu` module,
`/dev/kfd`, the render group) stop applying there automatically rather than
reporting confident nonsense.

| id | OS | Failure mode | Typical signal | Auto-fix |
| --- | --- | --- | --- | --- |
| `fix-1-arch` | linux/windows/wsl | GPU gfx target not in the framework's build arch list | `hipErrorNoBinaryForGpu`, `HSA_STATUS_ERROR_INVALID_ISA`, "invalid device function" | no |
| `fix-2-unset-override` | linux/windows/wsl | `HSA_OVERRIDE_GFX_VERSION` set on a GPU that now has a native wheel | page faults / `OUT_OF_REGISTERS`, override set in env | yes |
| `fix-3-rocm-kernel` | linux | ROCm + distro/kernel form an unsupported triple | ROCm installed but `amdgpu` not loaded; DKMS build failure | no |
| `fix-4-render-group` | linux | User not in `render`/`video` group (or `/dev/kfd` owned by the other group) | cannot open `/dev/kfd`, permission denied | yes |
| `fix-5-amdgpu-load` | linux | `amdgpu` module not loaded (or blacklisted) | "ROCk module is NOT loaded", blacklist entry, Secure Boot | no |
| `fix-6-path` | linux/windows/wsl | ROCm/HIP binaries not on PATH after install | `rocminfo: command not found`, `hipInfo` missing from PATH | yes |
| `fix-7-stale-repos` | linux | Stale/conflicting APT/DNF repos from prior installer runs | apt 404 `repo.radeon.com`, unmet deps, ≥2 ROCm repo files | no |
| `fix-8-wheel-rocm` | linux/windows/wsl | Framework wheel built for a different ROCm major than the system | `libamdhip64.so.X` / `amdhip64_X.dll` load failure | no |
| `fix-9-igpu-dgpu` | linux/windows | iGPU enumerated alongside dGPU, destabilising the runtime | APU + discrete AMD present, `HIP_VISIBLE_DEVICES` unset, crash/segfault | yes |
| `fix-10-container` | linux | Container can't see `/dev/kfd` or `/dev/dri/renderD*` | running in docker/podman, kfd/render devices missing | no |
| `fix-11-iommu` | linux | Multi-GPU hang with IOMMU enabled | ≥2 AMD GPUs, `iommu=` not `pt`, hang/deadlock/timeout | no |
| `fix-12-installer` | linux | `amdgpu-install` left a broken DKMS / repo state | dpkg half-configured, DKMS failed, `--accept-eula` | no |
| `fix-13-hip-sdk-missing` | windows | HIP SDK not installed | no HIP SDK under Program Files, `hipInfo` not recognized | no |
| `fix-14-adrenalin-too-old` | windows | Adrenalin / kernel-mode driver too old for the HIP SDK | `hipInfo` can't enumerate, "driver too old", HSA "no agents found" | no |
| `fix-15-msvc-redist` | windows | MSVC runtime missing (HIP DLLs can't load) | `vcruntime140.dll` / `vcruntime140_1.dll` missing | no |
| `fix-17-torch-dlpack` | linux | `torch-c-dlpack-ext` loads its CUDA prebuilt on a ROCm torch, aborting vLLM's engine start at import time | vLLM engine start fails on import; error names `torch_c_dlpack_ext` or tvm_ffi's `_optional_torch_c_dlpack` | no |
| `fix-wsl-1-gpu-not-exposed` | wsl | `/dev/dxg` absent, so the distro cannot reach the GPU at all | no `/dev/dxg`; in a container, the device was never passed through | no |
| `fix-wsl-2-dxcore-missing` | wsl | `/usr/lib/wsl/lib` DXCore shims missing, so the runtime cannot reach the host driver | `/usr/lib/wsl/lib/libdxcore.so` missing (or the directory absent entirely) | no |
| `fix-wsl-3-rocdxg-missing` | wsl | ROCDXG, the ROCm-to-DXCore shim the WSL path runs on, is not installed | `librocdxg.so` not found under any ROCm install | no |
| `fix-wsl-4-rocdxg-not-linked` | wsl | `librocdxg` installed but not in the linker cache, so it is unloadable | `librocdxg` present on disk yet absent from `ldconfig -p` | no |
| `fix-wsl-5-distro-too-old` | wsl | Distro release below the floor the WSL path requires | distro release under the supported floor (e.g. Ubuntu 22.04, whose glibc 2.35 is below the 2.38 / `GLIBCXX_3.4.32` the engines need) | no |
| `fix-wsl-6-host-driver-too-old` | wsl | Windows host driver too old or absent, with the distro side already complete | the Windows host reports no AMD display adapter | no |
| `fix-wsl-7-wsl1` | wsl | Distro running under WSL 1, which exposes no GPU device at all | the running kernel is a WSL 1 kernel | no |

Two things the numbering does not tell you. `fix-16` is a reserved handle, not a
missing row — ids are stable handles rather than positions. And the `fix-wsl-N`
entries are a parallel series, not a continuation of the numeric one, because
they answer for a different platform family.

Linux-only: fix-3, -4, -5, -7, -10, -11, -12, -17. Windows-only: fix-13, -14, -15.
WSL-only: fix-wsl-1 through fix-wsl-7. Linux + Windows: fix-9.
Linux + Windows + WSL: fix-1, -2, -6, -8.

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

- NVIDIA / Intel / Apple Silicon GPUs; fresh installs on a clean machine.

WSL2 used to sit here and no longer does: it is a supported platform family with
its own seven catalog entries. It remains a *distinct* platform — the GPU
arrives over `/dev/dxg` from the Windows host driver, not through the in-tree
`amdgpu` module or `/dev/kfd` — which is why it gets its own entries rather than
inheriting the bare-metal Linux ones. AMD's ROCm-on-WSL guide is still the right
pointer for install questions the catalog does not cover:
https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/wsl/howto_wsl.html
