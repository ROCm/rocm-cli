# Skill Card

## Description

Diagnoses ROCm / HIP SDK / PyTorch / llama.cpp failures on AMD GPUs (Linux,
Windows, and WSL2) against a closed list of known misconfigurations, and applies a
low-risk fix with consent or routes the user to the right upstream channel. It
also covers two vLLM startup failures: running out of GPU memory and an engine
import error (both on Linux and WSL2). Both fixes are print-only. A
thin driver over the `rocm` CLI (`rocm examine` / `rocm diagnose` / `rocm fix`) —
the probe, catalog, and fixes live in the CLI, versioned with the binary.

## Owner

rocm-cli team (AMD)

## License

MIT
