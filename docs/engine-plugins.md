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

Packaged first-party adapters are `lemonade`, `vllm`, and `aim`. Linux/WSL-only
ROCm GPU adapters (such as `vllm` and `aim`) fail explicitly on native Windows
instead of selecting a CPU fallback.

The engine-selecting commands (`rocm serve --engine`, `rocm engines
install`/`shell`, `rocm config set-engine`/`set-default-engine`) currently accept
only the built-in `lemonade`, `vllm`, and `aim` engines. Discovery still lists
external plugins under the search directories above, but selecting one by name
from the CLI is not supported while the engine set is limited to the built-ins.

The `lemonade` adapter uses Lemonade embeddable and prefers Lemonade's
`llamacpp:rocm` backend, falling back to `llamacpp:vulkan` when ROCm is
unsupported. rocm-cli does not use a CPU fallback for this path.

## The stdio protocol contract

An adapter run as `rocm-engine-<engine> stdio` (and `rocm __engine-stdio
<engine>`, which serves the built-in ones) reads one JSON request envelope on
stdin, to its end, and writes one JSON response envelope on stdout. The client
that sends it is [`stdio_client.rs`](../crates/rocm-engine-protocol/src/stdio_client.rs),
shared by `rocm` and `rocmd`; the version rule is in
[`protocol_version.rs`](../crates/rocm-engine-protocol/src/protocol_version.rs).

**Every envelope names its protocol.** Requests and responses carry
`protocol_version`, the `MAJOR.MINOR.PATCH` version of the engine protocol the
side that wrote it speaks (`ENGINE_PROTOCOL_VERSION`, `0.1.0` today; `rocm
examine` prints it as `protocol:`). `0.1.0` is the baseline: the release that
introduced the field speaks the same format as the messages written before it.
So an envelope without the field is read as `0.1.0`, and an adapter or a daemon
from an earlier release keeps working with this one.

**Compatibility is the caret rule.** While the major is `0`, two versions are
compatible when their major and minor match (`0.1.0` and `0.1.9` are, `0.1.0`
and `0.2.0` are not); from `1.0.0` on, when their major matches. The patch never
decides. A version must be three dot-separated plain decimal numbers, each
without a leading zero (`0` itself aside) and fitting in 64 bits; anything else
is unreadable and incompatible with everything.

**When the version is raised.**

- An addition an older peer can ignore (a new optional field, a new method
  that only an engine ever offers) raises the patch.
- A new method a client may send raises the minor while the major is `0`, and
  the major after that, because an older engine cannot decode it. The exception
  is a method the client sends only to engines whose capabilities say they have
  it.
- A removal or a type change (a field dropped, renamed, made required or given
  another type) raises the minor while the major is `0`, and the major after
  that.

**What each side does with a mismatch.**

- An adapter reads the request's version before anything else in it, its method
  included. A request it cannot serve, including one whose `protocol_version`
  is present and neither a string nor `null`, gets an error envelope with code
  `incompatible_protocol`, whatever its method, and nothing else happens. A
  `protocol_version` of `null` is read like a missing one, as `0.1.0`. A
  request in a protocol it can serve that names a method it does not have gets
  `unknown_method`, and input that is not a request envelope (including JSON
  that is not an object) gets `invalid_request`. Each of these is an answer on
  stdout. Input that is not valid UTF-8 is not read as a request: the adapter
  exits with an error and prints no answer.
- The client checks a response's version before anything else in it, `ok`
  and the shape of the rest included, whenever stdout is a single JSON object.
  An answer in a protocol it cannot use is an error that names both versions,
  even when it reports success or does not otherwise decode.
- Before an `install` or a `launch`, the requests that install or start
  something, the client first sends `capabilities` (with an empty object as its
  payload) through the same path, on the same environment minus the request
  for progress, and checks the version of that answer. The `install` or
  `launch` is sent only when that answer is in a compatible protocol and is not
  an error; otherwise the error says the request was not sent and why (the
  engine's version, or the reason its capabilities could not be read). Nothing
  is cached: each such request checks again. The check is made by the shared
  client, for an engine binary it runs: `rocm` answers its built-in engines in
  its own process, with no check, and no shipped command sends an install or
  launch to an engine outside it today. The other methods are sent as they are
  and only their answer is checked. `stop` is one of them on purpose: it starts
  no extra child for a check first, so a stop is never held behind one. An
  adapter that checks the version refuses a `stop` made for a protocol it
  cannot serve, before reading its payload. An adapter from before the field
  ignores the version and acts on the `stop`; its answer names no version and
  reads as `0.1.0`, so the client accepts it while this release's protocol is
  `0.1.x` and refuses it once the protocol is past that. When the daemon stops
  a server and the engine's answer is refused, the report carries that error
  as the engine's result; the daemon
  still signals the server's recorded processes itself, so for a `vllm` server
  the stop completes, while for an `aim` server, whose stop only the engine can
  confirm, it is left unconfirmed.

**Stdout holds only the answer.** The whole of stdout must parse as one
response envelope, a JSON object, pretty-printed over many lines or not. Anything else there,
such as a log line before the answer, makes the answer unreadable, and the
error says so, quoting the other output (or, when there is no answer at all,
the start of stdout). The rule for an adapter is to write its logs and progress
to stderr, which the client collects and shows when a request fails. The
first-party adapters do not hold it everywhere yet. Known cases where a child
process they start can write to the adapter's stdout:

- `aim` runs `docker pull` with inherited stdout when it starts a container.
- `lemonade` runs its llama.cpp backend install with inherited stdout. When
  it is given no log file, the Lemonade server it starts, the Lemonade CLI it
  runs to pull or load a model, and a packaged `llama-server` it runs directly
  on a GGUF file all inherit stdout too.
- `vllm` leaves the server's stdout inherited when it is given no log file.

These are known and not yet changed.

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
shipped an asset for the active ROCm version — the second lists the 30 newest
`lemonade-sdk/llama.cpp` releases on GitHub (an unauthenticated
`api.github.com` call, subject to GitHub's public rate limit) and installs the
most recently published one that ships the llama.cpp archive for the active
ROCm version's `major.minor` on this platform
(`llama-<tag>-bin-ubuntu-rocm-<major.minor>-x64.tar.gz` on Linux,
`llama-<tag>-bin-win-rocm-<major.minor>-x64.zip` on Windows). The newest
release alone is not used, because a release can be published without its
ROCm archives. The second attempt is skipped when none of the 30 ships that
archive, when the listing cannot be fetched, and when the build it finds is the
one Lemonade already pins (the first attempt tried that build). That second
build is a moving target, not a pin: which llama.cpp commit actually gets
installed depends on when the install ran, and the version installed is not
recorded anywhere `rocm examine`/`rocm version` report. An attempt that cannot
be verified against the installed backend's shared libraries reverts to whichever version
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
derived from it, so the tree cannot end up holding two different versions of
the same runtime. A Lemonade bump also records the new release's archive
digests (`EMBEDDABLE_LINUX_SHA256` and `EMBEDDABLE_WINDOWS_SHA256` in
[install.rs](../engines/lemonade/src/install.rs)): the downloaded archive is
checked against them, and a digest cannot be derived from the version. It also
re-reads what the new Linux `lemond` needs from the host, the newest `GLIBC_*`
and `OPENSSL_*` versions `objdump -T lemond` lists, into `LINUX_LEMOND_NEEDS`
beside those digests; a unit test fails until that record names the new Linux
archive. On Linux, `rocm engines install lemonade` refuses a host whose glibc
or OpenSSL is older than that before downloading anything, and the engine
refuses to start that `lemond` on such a host
([host_libraries.rs](../engines/lemonade/src/host_libraries.rs)). For each
library that falls short, the message names the version needed and what the
host has: its version, read from the library the `ldconfig` cache lists, or
that the cache has no such library at all. A host with only OpenSSL 1.1 has no
`libssl.so.3`, so it reads that way. A host whose cache cannot be read, or
whose library does not parse, is not refused. A refused install says what it
left undone: "Nothing was downloaded." for a fresh install, or that the
Lemonade already installed there stays as it is. Because only the `ldconfig`
cache is read, a host that provides OpenSSL 3 or glibc outside it
(`LD_LIBRARY_PATH`, a Nix or Guix profile, a user prefix, a stale cache) is
refused although `lemond` could load there; setting
`ROCM_CLI_DISABLE_LEMONADE_HOST_LIBRARY_CHECK` (any value) turns both refusals
into a warning, and the refusal message names it. Lemonade
2026.44.1, the current pin, needs glibc 2.38 and OpenSSL 3.0.0, which Ubuntu
24.04 meets; 2026.45.3's `lemond` needs glibc 2.43 and OpenSSL 3.2.0, which it
does not.

`rocm engines list` shows the exact plugin directories for the current host.
The same output is available in the TUI with `/engine`.

Installer policy:

- `install.sh` and `install.ps1` update only the rocm-cli binary install
  directory (its binaries, `.rocm-cli-manifest`, and the `.rocm-cli-path-edit`
  record of the `PATH` change they made) and that `PATH` change itself: a
  marked block in the login shell's profile, plus fish's `config.fish`
  when fish is installed, or the user `PATH` on Windows. When fish is
  installed, `install.sh` also writes the fish completion script to
  `completions/rocm.fish` in fish's config folder (`$XDG_CONFIG_HOME/fish`,
  else `~/.config/fish`), starting with a line naming the install folder, and
  records it, so `rocm uninstall` removes it with the binaries. Uninstall
  leaves a `rocm.fish` that does not start with that line for its own folder.
  The installer itself leaves an existing `rocm.fish` (or link) that does not
  start with that marker as it is, prints a note, and records nothing for it.
  A custom `ROCM_CLI_SHELL_PROFILE` replaces only the login shell's profile;
  fish's is still updated.
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
