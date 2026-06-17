# Implementation Completion Audit

Audited on 2026-06-06 against:

- `plans/rocm-cli-implementation-plan.md`
- `plans/rocm-cli-pytorch-engine-spec.md`
- `plans/rocm-cli-remaining-implementation-plan.md`

This document is intentionally plan-bound. It does not add feature ideas. When a
row is not complete, the reason must be a concrete owner, hardware, upstream, or
infrastructure gate rather than an invented fallback.

## Local Functional Baseline

The branch currently has local implementations for the V1 product surfaces:

- Windows and Linux/WSL bootstrap installers, bundle verification, package
  scripts, minimal-config seeding, automatic PATH setup, and clean-machine
  operation without preinstalled ROCm, Python, Rust, or Cargo.
- First-time TUI setup with an arrow-key folder picker, approval cards, live
  setup logs, local pip cache under the selected ROCm install folder, and
  persisted settings under `~/.rocm`.
- Fast host/GPU checks for setup and Doctor: native Windows registry/system
  probes avoid PowerShell/CIM on the common path, native Linux uses sysfs/KFD/IP
  discovery first, and WSL collapses host display lookup to one bridge query
  only when Linux-side sysfs cannot answer.
- TheRock managed `pip` installs using the selected GPU family and a single
  pinned TheRock-index transaction for `rocm[libraries,devel]`, `torch`,
  `torchvision`, and `torchaudio`.
- Runtime list, activate, uninstall, import, adopt, active-runtime markers, and
  previous-runtime validation. User-facing selection is list-based, not
  rollback-first.
- Doctor, GPU detection, managed runtime inventory, services inventory, logs,
  update reports, and bounded startup update checks.
- PyTorch and `llama.cpp` GPU serving with managed TheRock library propagation.
  The `llama.cpp` adapter is backed by upstream `llama-server`.
- Local assistant chat against a managed local server, with ROCm tool calls,
  approval routing for mutating actions, and visible tool results/details in
  the TUI.
- ComfyUI install/status/logs/start/stop as a managed app surface using the
  active TheRock runtime, with live Windows and WSL GPU cat-generation
  acceptance through the ComfyUI HTTP API.
- Lemonade embeddable is available as a strict ROCm engine on Windows. WSL
  Lemonade GPU serving remains gated on Lemonade's own WSL `/dev/dxg` AMD GPU
  detection; rocm-cli does not use CPU or Vulkan fallback.
- vLLM, SGLang, and ATOM adapter packaging and explicit platform/support gates
  with no CPU fallback.
- Automations, reviewed proposals, watcher policy routing, sandbox runner, MCP
  read-only/mutating boundaries, lifecycle logs, and audit records.
- TUI command surfaces that open navigable screens or focused overlapping modal
  cards instead of transcript dumps.
- Native per-OS rocm-cli binary that runs the real CLI/TUI on Windows and
  WSL/Linux, with first-party engines and the `rocmd` helper surface linked into
  `rocm`. TheRock wheels, Python envs, ComfyUI, ROCm libraries, and model files
  remain user-approved runtime content installed on disk.
- CI and local verification hooks for all-target tests, clippy with warnings
  denied, release-readiness self-tests, installer lifecycle tests, and
  acceptance-harness self-tests.

## Remaining External Gates

| Area | Current State | Why It Is Still Gated |
|---|---|---|
| Production metadata signing | Test-key verification and tamper rejection are implemented. | The repository owner must publish production public keys, configure release secrets, and host signed metadata sidecars. |
| Hosted recipe index and source policy | Signed recipe-index schema, local verification, source-policy validation, direct HTTPS prefetch, and Hugging Face token handling exist. | Production hosted indexes and source-policy metadata must be supplied before rocm-cli can claim production publication. |
| Privileged Linux driver install acceptance | Distro plans, preflight checks, approval boundaries, execution state, and reconcile commands exist. | Live DKMS acceptance needs a supported Linux host with root/sudo control and compatible driver state. |
| ATOM live GPU acceptance | Adapter packaging, managed TheRock environment propagation, and offline exact-runtime selector tests exist. | Live acceptance needs upstream-supported ATOM GPU targets. Current local `gfx1201` host is not an upstream ATOM target. |
| SGLang live GPU acceptance | Adapter packaging, managed-runtime parity, explicit Windows gate, and offline selector tests exist. | Current local `gfx1201` host is blocked by upstream SGLang ROCm kernel support. |
| Broader GPU-family CI | Normal Linux/Windows CI, local no-fallback smoke, self-hosted adapter detect/capabilities smoke, and local RDNA4 Windows/WSL acceptance exist. | More live serving coverage needs additional CI hardware or lab machines. |
| Production driver-update feed | Local update-available event handling and reviewed driver-plan proposals exist. | A real AMD driver update source/feed must be defined before wiring production event detection. |
| Multi-target release publishing | Native per-OS Windows and Linux release/nightly bundles are built and published by CI. | Broader native Linux hardware coverage (server/bare-metal Instinct) still needs validation outside the current Windows/WSL hosts. |
| Future contained mutating actions | Current contained read-only checks and reviewed mutating proposals exist. | Additional mutating automation actions need explicit product requirements before implementation. |

## Verification Anchors

Use `docs/testing.md` for the full command list. The short evidence set is:

```powershell
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
python scripts/smoke_local.py --skip-build
python scripts/therock_sdk_install_test.py --root .rocm-work/tests/therock-sdk-install --dry-run --check-windows-tools
python scripts/pytorch_therock_gpu_test.py --self-test
python scripts/llama_cpp_therock_gpu_test.py --self-test
python scripts/comfyui_therock_gpu_test.py --self-test
python scripts/local_assistant_therock_gpu_test.py --self-test
```

Live GPU acceptance is opt-in because it installs packages and launches local
servers:

```powershell
python scripts/therock_sdk_install_test.py --root .rocm-work/tests/therock-sdk-install --fresh
python scripts/pytorch_therock_gpu_test.py
python scripts/llama_cpp_therock_gpu_test.py --launch-mode launch
python scripts/local_assistant_therock_gpu_test.py --model qwen --require-tool-call
python scripts/comfyui_therock_gpu_test.py
```

Live acceptance can target the native release binary built by
`cargo build --workspace --release` (`target\release\rocm.exe` on Windows,
`target/release/rocm` on Linux/WSL):

```powershell
python scripts\local_assistant_therock_gpu_test.py --rocm target\release\rocm.exe --engine lemonade --model qwen --require-tool-call
python scripts\local_assistant_therock_gpu_test.py --rocm target\release\rocm.exe --engine pytorch --model qwen --require-tool-call
python scripts\comfyui_therock_gpu_test.py --rocm target\release\rocm.exe
```

WSL verification should run from the WSL filesystem when possible for better IO:

```bash
export CARGO_TARGET_DIR=/home/user/.cache/rocm-cli-target
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
python3 scripts/smoke_local.py --skip-build
python3 scripts/therock_sdk_install_test.py --root .rocm-work/tests/therock-sdk-install --dry-run
python3 scripts/pytorch_therock_gpu_test.py --self-test
python3 scripts/llama_cpp_therock_gpu_test.py --self-test
python3 scripts/local_assistant_therock_gpu_test.py --self-test
```

For WSL/Linux, build and run the native Linux binary directly:

```bash
/home/user/rocm-cli-e2e/bin/rocm doctor
python3 scripts/local_assistant_therock_gpu_test.py --rocm /home/user/rocm-cli-e2e/bin/rocm --engine lemonade --model qwen --require-tool-call
python3 scripts/local_assistant_therock_gpu_test.py --rocm /home/user/rocm-cli-e2e/bin/rocm --engine pytorch --model qwen --require-tool-call
python3 scripts/comfyui_therock_gpu_test.py --rocm /home/user/rocm-cli-e2e/bin/rocm
```

## Regression Rules

- Do not add CPU fallback for GPU-required serving paths.
- Do not turn TUI command results back into transcript dumps.
- Keep TUI command surfaces arrow-key navigable, with clear footer hints for
  Enter, Esc, Up/Down, Tab, and PageUp/PageDown where those keys apply.
- Use overlapping modal-style cards for focused decisions, progress, logs,
  details, tool-call reviews, help, and short fix prompts. A visible card must
  own keyboard focus and preserve the covered screen and selection underneath.
- Do not expose raw log paths, runtime keys, wheel jargon, or backend labels in
  first-visible TUI screens unless the user opens an explicit detail/debug view.
- Do not claim production signing, production hosted indexes, privileged driver
  acceptance, ATOM/SGLang live acceptance, broader GPU CI, production driver
  feeds, or native-Linux server/bare-metal acceptance until the required owner,
  hardware, upstream, or infrastructure input exists and has passed acceptance.
