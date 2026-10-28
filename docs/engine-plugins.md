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
`llamacpp:rocm` backend downloads. On Linux/WSL and Windows, `rocm engines
install lemonade` attempts to rewrite that pin to match the active ROCm SDK so
the backend it installs is paired with the SDK actually in use, rather than
whatever version Lemonade shipped pinned to. The attempt is skipped outright
on any other OS, where no mechanism exists to verify whether a backend
actually resolves its ROCm libraries, left alone when the active SDK reports
a nightly version rather than a plain `X.Y.Z` the alignment can match against
(or no active SDK version can be determined at all, or Lemonade's packaged
pin can't be read), skipped when the pin already matches the active version,
and skipped on a host where Lemonade would select `llamacpp:vulkan` instead of
`llamacpp:rocm` regardless (WSL2 being the documented case, and also the
fallback if that probe itself fails), since no rewrite can make an
unsupported ROCm build
resolve there.

The rewrite tries two builds in order. The first keeps Lemonade's own pinned
llama.cpp release and just repoints it at the active ROCm version — a
deliberate, reproducible pin. If that attempt fails to resolve against the
installed backend's shared libraries — for example because that release never
shipped an asset for the active ROCm version — the second queries GitHub's
`releases/latest` for `lemonade-sdk/llama.cpp` (an unauthenticated
`api.github.com` call, subject to GitHub's public rate limit) and installs
whatever build is newest at that moment. That second build is a moving
target, not a pin: which llama.cpp commit actually gets installed depends on
when the install ran, and the version installed is not recorded anywhere
`rocm examine`/`rocm version` report. An attempt that cannot be verified
against the installed backend's shared libraries reverts to whichever version
was pinned before the attempt — the packaged pin on a fresh install or after
`--reinstall`, but whatever alignment last wrote otherwise — rather than
being kept.

A pin an earlier alignment kept can go stale: the llama.cpp or TheRock build it
names may later be withdrawn upstream. When the pin already equals the active
ROCm version the install does not attempt another alignment, so if installing
that pinned backend fails, `rocm engines install lemonade` restores the two pins
alignment writes (`therock.version` and `llamacpp.rocm-stable`) in
`resources/backend_versions.json` from the embeddable archive it was extracted
from (digest-verified; the file is rewritten atomically and every other entry
in it is kept) and installs once more against Lemonade's own pins, printing a
warning that says so. This changes what a plain install does: it can rewrite
those two pins and replace the selected backend, where it used to report the
failure and leave both alone.

The restore never downloads. It reads the cached copy of the embeddable archive
that the same install verified a moment earlier (under `downloads/` next to the
runtime tree). If that copy is gone or no longer matches its digest, the restore
is skipped, a warning says why, and the original install error is reported. The
same applies when the pin file exists but cannot be read (it is left as it is,
never treated as absent), when the pins already hold the packaged values, or when the
restore fails for any other reason: there is no second attempt and the original
install error is reported unchanged.

The retry is a forced reinstall of the selected backend into the shared tree.
Like alignment's own revert, it sets the existing ROCm backend directories aside
first and puts them back if the retry fails. The ROCm backend entries are listed
before the first attempt, and after a failed retry anything that was not in that
list (a partial build the failed attempts left) is removed once the directories
are back. Any install failure starts the restore, including a transient network
or disk error, so when the second attempt fails too, its error is reported (the
first error is in the warning) and the previous `backend_versions.json` is put
back.

Known limitation, shared with alignment's own revert: between setting the
directories aside and putting them back, the working backend exists only under a
`.pre-alignment` name. If the process is killed in that window the directory
stays there and nothing recovers it automatically, because the runtime tree is
shared between processes and a leftover aside cannot be told from another
process's install in progress. Run `rocm engines install lemonade --reinstall`
to rebuild the tree.

The previous `backend_versions.json` is held in memory only while the retry runs.
If the process is killed in that window the packaged pins stay on disk and the
previous bytes are lost; the next install runs alignment against the packaged
pins again. The recovery applies only to the case above (the kept pin equals the
active ROCm version): with no active SDK, an unreadable pin file, or a failed
backend probe, the install uses what the file pins and does not restore.

Setting the variable below skips this recovery along with the alignment.

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
  directory (its binaries, `.rocm-cli-manifest`, and the `.rocm-cli-path-edit`
  record of the `PATH` change they made) and that `PATH` change itself: a
  marked block in the shell profile, or the user `PATH` on Windows.
- External plugins under the rocm-cli data directory are not touched by
  install or upgrade.
- `rocm uninstall` removes the data directory by default. Use
  `rocm uninstall --keep-data --keep-config` when external plugins, managed
  runtimes, service records, or model cache entries should be preserved. On the
  default layout the configuration and data directories are the same
  directory, so both flags are needed.

No fallback engine is selected automatically. If an engine adapter is missing
or cannot satisfy the requested device policy, the command must fail until the
requested engine is installed or the user explicitly selects a different one.
