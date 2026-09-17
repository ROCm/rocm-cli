<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Engine Plugins

rocm-cli discovers serving engine adapters as executable files.

Search order:

1. `<data_dir>/engines/plugins`
2. `<data_dir>/engines`
3. Packaged sibling binaries installed beside `rocm`

The first directory is the preferred location for external adapters. Use a
binary name in the form `rocm-engine-<engine>` on Linux/WSL and
`rocm-engine-<engine>.exe` on Windows.

Packaged first-party adapters are `lemonade` and `vllm`. Linux/WSL-only ROCm GPU
adapters (such as `vllm`) fail explicitly on native Windows instead of selecting
a CPU fallback.

The engine-selecting commands (`rocm serve --engine`, `rocm engines
install`/`shell`, `rocm config set-engine`/`set-default-engine`) currently accept
only the built-in `lemonade` and `vllm` engines. Discovery still lists external
plugins under the search directories above, but selecting one by name from the
CLI is not supported while the engine set is limited to the two built-ins.

The `lemonade` adapter uses Lemonade embeddable and prefers Lemonade's
`llamacpp:rocm` backend, falling back to `llamacpp:vulkan` when ROCm is
unsupported. rocm-cli does not use a CPU fallback for this path.

## Lemonade backend alignment on engine install

Lemonade's `resources/backend_versions.json` pins which ROCm SDK version its
`llamacpp:rocm` backend downloads. On Linux/WSL, `rocm engines install
lemonade` attempts to rewrite that pin to match the active ROCm SDK so the
backend it installs is paired with the SDK actually in use, rather than
whatever version Lemonade shipped pinned to. The attempt is skipped outright
on native Windows, and left alone when the active SDK reports a nightly
version rather than a plain `X.Y.Z` the alignment can match against. An
attempt that cannot be verified against the installed backend's shared
libraries reverts to the packaged pin instead of being kept.

Set `ROCM_CLI_DISABLE_LEMONADE_BACKEND_ALIGNMENT` to keep whatever
`backend_versions.json` already pins and skip the rewrite — any value works,
including an empty one, since the variable being set is the signal:

```bash
ROCM_CLI_DISABLE_LEMONADE_BACKEND_ALIGNMENT=1 rocm engines install lemonade --yes
```

Use it if you have hand-edited `backend_versions.json` to pin a specific
version, matching that file's own documented use as a first-party
customization point.

## Pinned runtime versions

The versions of the third-party runtimes rocm-cli downloads are pinned in
`runtime-deps.toml` at the repository root — one `[runtime.<name>]` table per
runtime. That file is the only place a runtime version is written down:
archive names, download URLs, and the dashboard's offline fallback are all
derived from it, so a bump is a one-line edit and the tree cannot end up
holding two different versions of the same runtime.

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
