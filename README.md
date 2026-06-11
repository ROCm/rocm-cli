# rocm-cli

ROCm AI Command Center CLI for AMD systems.

## Quick Start

Install on Windows x86_64 from PowerShell:

```powershell
$script = "$env:TEMP\install-rocm-cli.ps1"; irm https://raw.githubusercontent.com/powderluv/rocm-cli/main/install.ps1 -OutFile $script; Set-ExecutionPolicy -Scope Process Bypass -Force; & $script
```

The bootstrap installer downloads the prebuilt rocm-cli bundle and updates PATH.
It does not require ROCm, Python, Rust, or Cargo to already be installed.

Install on Linux or WSL x86_64:

```bash
curl -fsSL https://raw.githubusercontent.com/powderluv/rocm-cli/main/install.sh | sh
```

Start rocm-cli:

```bash
rocm
```

On first run, rocm-cli opens setup before the main screen. Choose where ROCm
should be installed, approve the install, and wait for setup to finish.

After setup, check the machine:

```bash
rocm doctor
```

Serve a local model:

```bash
rocm serve qwen --engine lemonade --managed
```

## Developer Checks

Command-line ROCm install:

```bash
rocm install sdk --channel release --format pip
rocm runtimes list
```

Choose one installed ROCm folder:

```bash
rocm runtimes activate <runtime_key>
```

Replace `<runtime_key>` with a key printed by `rocm runtimes list`.

Install a serving engine:

```bash
rocm engines install pytorch
```

## Contributing

This repo uses [prek](https://github.com/j178/prek) (a fast, drop-in
replacement for `pre-commit`) to run the same checks locally that CI enforces:
`cargo fmt`/`clippy`/`test`, `ruff` (Python), `shellcheck` (shell),
PowerShell syntax, and assorted file hygiene.

Install prek and enable the hooks once after cloning:

```bash
uv tool install prek        # or: cargo install --locked prek
prek install                # fast checks on commit (fmt, ruff, shellcheck, ...)
prek install -t pre-push    # heavier checks on push (clippy, cargo test)
```

Run every hook against the whole tree on demand:

```bash
prek run --all-files
```

The hook configuration lives in `.pre-commit-config.yaml`; `clippy` and the test
suite run on `pre-push` to keep commits fast.

## More Docs

- Testing and verification: `docs/testing.md`
- Developer manual QA: `docs/manual-testing.md`
- Current implementation audit: `docs/implementation-completion-audit.md`
- Engine plugin policy: `docs/engine-plugins.md`
- ATOM adapter: `docs/atom.md`
- vLLM adapter: `docs/vllm.md`
- SGLang adapter: `docs/sglang.md`
- Implementation plan: `plans/rocm-cli-implementation-plan.md`
- PyTorch engine spec: `plans/rocm-cli-pytorch-engine-spec.md`

This is an early implementation, not a production release.
