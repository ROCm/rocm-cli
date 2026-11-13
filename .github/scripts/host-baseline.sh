#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Host baseline for the self-hosted Linux GPU lanes: fail with a named reason
# when the runner's C and C++ runtimes are older than the product supports,
# instead of letting the suite fail later and obscurely when the Lemonade
# engine cannot load.
#
# THE FLOOR IS THE PRODUCT'S OWN. README.md states the minimum supported Linux
# release as Ubuntu 24.04, or on other distributions glibc 2.38 with
# GLIBCXX_3.4.32 — what every published Lemonade build is linked against. CI
# asserts exactly that floor, on purpose: a runner below it cannot run what
# users are told is supported, and a runner floor above it would hide a
# regression on the oldest supported host. The two must not drift silently,
# so `host_baseline_contract.rs` in xtask fails when these constants and the
# README sentence disagree. Raising the CI floor is a separate decision from
# raising the product floor; make it there, not here.
#
# WHAT IS CHECKED, AND WHAT IS NOT. glibc through `getconf GNU_LIBC_VERSION`,
# and the newest GLIBCXX symbol version in the first x86-64 libstdc++ the
# dynamic loader's cache lists (`ldconfig -p`) — the host's system copy. A
# process whose LD_LIBRARY_PATH or RPATH/RUNPATH names another directory can
# load a different libstdc++, and this does not see that one: it judges the
# host, not any one engine's resolved libraries. The OS name is logged for the
# reader and never gated on: the floor is the libraries, and a non-Ubuntu host
# that provides them passes. Only x86-64 is checked, because every GPU runner
# this gates is x86-64; an x86-64 libstdc++ missing from the loader cache fails
# rather than passes.
#
# Exit status: 0 when the floor is met, 1 otherwise — including every value
# that could not be read. An unreadable runtime is not evidence of a good one.
#
# Env:
#   HOST_BASELINE_OS_RELEASE    os-release file to log the OS name from.
#                               Default: /etc/os-release, then
#                               /usr/lib/os-release.
#   HOST_BASELINE_LDCONFIG_DIRS space-separated directories searched for
#                               ldconfig when it is not on PATH. Default:
#                               /usr/sbin /sbin.
#   Both are set by the contract tests only; the lanes set neither.

set -euo pipefail

# Keep equal to README.md's "glibc X with `GLIBCXX_Y`"; a test ties them.
MIN_GLIBC=2.38
MIN_GLIBCXX=3.4.32

fail() {
  echo "::error::host baseline not met on ${os_name}: $1 (floor glibc ${MIN_GLIBC}, GLIBCXX_${MIN_GLIBCXX}; see README.md)"
  exit 1
}

# True when version $1 >= version $2. `sort -V` compares each dotted component
# as a number; a string comparison would put 2.100 below 2.38 and 3.4.4 above
# 3.4.32.
version_ge() {
  local lowest
  lowest="$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)"
  [[ "${lowest}" == "$2" ]]
}

os_name="unknown"
if [[ -n "${HOST_BASELINE_OS_RELEASE:-}" ]]; then
  os_release_files=("${HOST_BASELINE_OS_RELEASE}")
else
  os_release_files=(/etc/os-release /usr/lib/os-release)
fi
for f in "${os_release_files[@]}"; do
  if [[ -r "${f}" ]]; then
    name="$(sed -n 's/^PRETTY_NAME=//p' "${f}" | head -n 1 | tr -d '"'"'")"
    [[ -n "${name}" ]] && os_name="${name}"
    break
  fi
done

if ! command -v getconf >/dev/null 2>&1; then
  fail "getconf is not installed, so the glibc version cannot be read"
fi
libc_rc=0
libc_out="$(getconf GNU_LIBC_VERSION 2>/dev/null)" || libc_rc=$?
if [[ "${libc_rc}" -ne 0 ]]; then
  fail "getconf GNU_LIBC_VERSION failed (exit ${libc_rc}) — not a glibc host"
fi
if [[ ! "${libc_out}" =~ ^glibc\ ([0-9]+\.[0-9]+(\.[0-9]+)?)$ ]]; then
  fail "getconf GNU_LIBC_VERSION printed '${libc_out}', not a glibc version"
fi
glibc="${BASH_REMATCH[1]}"

# The loader cache, not a fixed path: libstdc++ sits in a different directory
# on different distributions, and the cache is where the loader looks it up
# once LD_LIBRARY_PATH and any RPATH/RUNPATH have not supplied it. Filtered to
# x86-64 because hosts with multilib list the /libx32 and /lib32 copies too,
# often first, and those are not the copy a 64-bit engine maps. Among the
# x86-64 entries the loader takes the first one, so that is the one judged.
#
# ldconfig is often in an sbin directory that a non-root PATH leaves out, so
# those are searched as well before concluding it is missing.
ldconfig_bin="$(command -v ldconfig || true)"
if [[ -z "${ldconfig_bin}" ]]; then
  read -r -a ldconfig_dirs <<<"${HOST_BASELINE_LDCONFIG_DIRS:-/usr/sbin /sbin}"
  for d in "${ldconfig_dirs[@]}"; do
    if [[ -x "${d}/ldconfig" ]]; then
      ldconfig_bin="${d}/ldconfig"
      break
    fi
  done
fi
if [[ -z "${ldconfig_bin}" ]]; then
  fail "ldconfig is not installed, so libstdc++ cannot be located"
fi
cache="$("${ldconfig_bin}" -p 2>/dev/null)" ||
  fail "ldconfig -p failed, so libstdc++ cannot be located"
# A here-string rather than a pipe: `sed` stops at the first match, and under
# pipefail a writer still feeding it a large cache would die of SIGPIPE (141)
# and take the script down with no `::error::` line. A here-string is written
# in full before `sed` starts, so there is no writer left to kill.
libstdcxx="$(sed -n '/^[[:space:]]*libstdc++\.so\.6 (libc6,x86-64[^)]*) => /{s///p;q}' <<<"${cache}")"
if [[ -z "${libstdcxx}" ]]; then
  fail "no x86-64 libstdc++.so.6 in the loader cache (ldconfig -p)"
fi
if [[ ! -r "${libstdcxx}" ]]; then
  fail "the loader cache names ${libstdcxx}, which cannot be read"
fi
# grep's status on its own: 1 is "no such string", anything else is a read
# that failed, and the two must not share a message.
grep_rc=0
symbols="$(grep -aoE 'GLIBCXX_3\.4(\.[0-9]+)?' "${libstdcxx}")" || grep_rc=$?
if [[ "${grep_rc}" -gt 1 ]]; then
  fail "grep could not read ${libstdcxx} (exit ${grep_rc})"
fi
glibcxx="$(printf '%s\n' "${symbols}" | sed -n 's/^GLIBCXX_//p' | sort -uV | tail -n 1)"
if [[ -z "${glibcxx}" ]]; then
  fail "${libstdcxx} carries no GLIBCXX_3.4 symbol version"
fi

short=()
version_ge "${glibc}" "${MIN_GLIBC}" ||
  short+=("glibc ${glibc} is below ${MIN_GLIBC}")
version_ge "${glibcxx}" "${MIN_GLIBCXX}" ||
  short+=("GLIBCXX_${glibcxx} (${libstdcxx}) is below GLIBCXX_${MIN_GLIBCXX}")
if [[ "${#short[@]}" -gt 0 ]]; then
  joined="$(printf '%s; ' "${short[@]}")"
  fail "${joined%; }"
fi

echo "host baseline: ${os_name}, glibc ${glibc}, GLIBCXX_${glibcxx} (floor glibc ${MIN_GLIBC}, GLIBCXX_${MIN_GLIBCXX})"
