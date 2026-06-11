# Engine Plugins

rocm-cli discovers serving engine adapters as executable files.

Search order:

1. `<data_dir>/engines/plugins`
2. `<data_dir>/engines`
3. Packaged sibling binaries installed beside `rocm`

The first directory is the preferred location for external adapters. Use a
binary name in the form `rocm-engine-<engine>` on Linux/WSL and
`rocm-engine-<engine>.exe` on Windows. The `llama.cpp` engine uses
`rocm-engine-llama-cpp` on Linux/WSL and `rocm-engine-llama-cpp.exe` on
Windows.

Packaged first-party adapters are `pytorch`, `llama.cpp`, `lemonade`, `atom`,
`vllm`, and `sglang`. Linux/WSL-only ROCm GPU adapters fail explicitly on
native Windows instead of selecting a CPU fallback.

The `lemonade` adapter uses Lemonade embeddable and requires Lemonade's
`llamacpp:rocm` backend. Windows ROCm serving is validated. WSL is currently
blocked by Lemonade v10.6.0 reporting no AMD GPU through its own detector even
when TheRock/librocdxg works; rocm-cli does not use CPU or Vulkan fallback for
that path.

`rocm engines list` shows the exact plugin directories for the current host.
The same output is available in the TUI with `/engine`.

Installer policy:

- `install.sh` and `install.ps1` update only the rocm-cli binary install
  directory and its `.rocm-cli-manifest`.
- External plugins under the rocm-cli data directory are not touched by
  install or upgrade.
- `rocm uninstall` removes the data directory by default. Use
  `rocm uninstall --keep-data` when external plugins, managed runtimes,
  service records, or model cache entries should be preserved.

No fallback engine is selected automatically. If an engine adapter is missing
or cannot satisfy the requested device policy, the command must fail until the
requested engine is installed or the user explicitly selects a different one.
