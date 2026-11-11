#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# `apt-get update` plus `apt-get install -y PACKAGE...` for a GitHub-hosted
# Ubuntu runner, falling back to a public mirror when Canonical's own hosts do
# not resolve.
#
# The hosted images point apt at azure.archive.ubuntu.com, archive.ubuntu.com
# and security.ubuntu.com. When Canonical's DNS is down every one of them fails
# to resolve and apt exits 100, however many retries it is given: the
# 2026-11-11 outage (https://github.com/actions/runner-images/issues/15296)
# failed every Linux job at this step before any repository code ran. A mirror
# under a different domain still resolves.
#
# The fallback applies only when a probe host fails to resolve, so a healthy
# run takes exactly the path it took before. Package integrity does not depend
# on the mirror: apt still verifies the Release files against the Ubuntu
# archive keyring named in the sources.
#
# Inputs, all optional:
#   APT_FALLBACK_MIRRORS  space-separated mirror base URLs, tried in order
#                         (default: see MIRRORS below)
#   APT_PROBE_HOSTS       hosts that must resolve to skip the fallback
#                         (default "archive.ubuntu.com security.ubuntu.com")
#   APT_SOURCES_FILE      deb822 sources file to rewrite
#                         (default /etc/apt/sources.list.d/ubuntu.sources)
#   APT_MIRRORS_FILE      the image's mirror list, rewritten when present
#                         (default /etc/apt/apt-mirrors.txt)
#
# `--self-test` runs the offline checks of the rewrite and exits;
# `--rewrite-only FILE MIRROR` applies the rewrite alone, to try it on a copy.

set -euo pipefail

sources_file="${APT_SOURCES_FILE:-/etc/apt/sources.list.d/ubuntu.sources}"
mirrors_file="${APT_MIRRORS_FILE:-/etc/apt/apt-mirrors.txt}"
probe_hosts="${APT_PROBE_HOSTS:-archive.ubuntu.com security.ubuntu.com}"
# Full amd64 archives, with every pocket, in different domains and networks.
MIRRORS="${APT_FALLBACK_MIRRORS:-https://mirrors.mit.edu/ubuntu https://mirror.math.princeton.edu/pub/ubuntu https://mirrors.kernel.org/ubuntu}"

# Run a command as root, through sudo when we are not.
as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then "$@"; else sudo "$@"; fi
}

# True when every probe host resolves.
hosts_resolve() {
  local host
  for host in ${probe_hosts}; do
    getent hosts "${host}" >/dev/null || return 1
  done
}

# Print the first mirror that serves the Release file for $1 (a suite).
pick_mirror() {
  local suite="$1" mirror
  for mirror in ${MIRRORS}; do
    if curl -fsS --max-time 15 -o /dev/null "${mirror}/dists/${suite}/InRelease"; then
      echo "${mirror}"
      return 0
    fi
  done
  return 1
}

# Rewrite every Canonical archive URI in the deb822 file $1, and in the image's
# mirror list $3 when it exists, to the mirror $2. The `mirror+file:` URI is how
# the hosted images name their mirror list; the security stanza names
# security.ubuntu.com on its own, so the mirror list alone is not enough.
rewrite_sources() {
  local file="$1" mirror="$2" list="$3"
  sed -i -E \
    -e "s#(mirror\\+file:/etc/apt/apt-mirrors\\.txt|https?://(azure\\.archive|archive|security)\\.ubuntu\\.com/ubuntu/?)#${mirror}/#" \
    "${file}"
  if [[ -f "${list}" ]]; then
    printf '%s/\tpriority:1\n' "${mirror}" >"${list}"
  fi
}

self_test() (
  local dir
  dir="$(mktemp -d)"
  trap 'rm -rf "${dir}"' EXIT
  cat >"${dir}/ubuntu.sources" <<'EOF'
Types: deb
URIs: mirror+file:/etc/apt/apt-mirrors.txt
Suites: noble noble-updates
Components: main universe

Types: deb
URIs: http://security.ubuntu.com/ubuntu/
Suites: noble-security
Components: main universe
EOF
  printf 'http://azure.archive.ubuntu.com/ubuntu/\tpriority:1\n' >"${dir}/apt-mirrors.txt"
  rewrite_sources "${dir}/ubuntu.sources" https://m.example/ubuntu "${dir}/apt-mirrors.txt"
  [[ "$(grep -c '^URIs: https://m.example/ubuntu/$' "${dir}/ubuntu.sources")" -eq 2 ]] ||
    { echo "self-test: both stanzas should name the mirror" >&2; return 1; }
  ! grep -q 'ubuntu\.com' "${dir}/ubuntu.sources" ||
    { echo "self-test: a Canonical host is left in the sources" >&2; return 1; }
  grep -q '^https://m.example/ubuntu/' "${dir}/apt-mirrors.txt" ||
    { echo "self-test: the mirror list was not rewritten" >&2; return 1; }
  # Suites and the keyring line must be untouched.
  grep -q '^Suites: noble-security$' "${dir}/ubuntu.sources" ||
    { echo "self-test: suites changed" >&2; return 1; }
  probe_hosts="no-such-host.invalid"
  ! hosts_resolve || { echo "self-test: an unresolvable host passed" >&2; return 1; }
  probe_hosts="localhost"
  hosts_resolve || { echo "self-test: localhost did not resolve" >&2; return 1; }
  echo "ci-apt-install self-test: ok"
)

main() {
  if [[ "${1:-}" = "--self-test" ]]; then
    self_test
    return
  fi
  if [[ "${1:-}" = "--rewrite-only" ]]; then
    rewrite_sources "$2" "$3" /nonexistent
    return
  fi
  if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 PACKAGE..." >&2
    return 2
  fi

  if hosts_resolve; then
    as_root apt-get update
  else
    echo "::warning::Ubuntu archive hosts do not resolve; falling back to a public mirror"
    # Only the amd64 archive is mirrored under /ubuntu; arm64 is served from
    # ports.ubuntu.com on 22.04 and 24.04, which this does not cover.
    local arch codename mirror
    arch="$(dpkg --print-architecture)"
    if [[ "${arch}" != "amd64" ]]; then
      echo "no fallback mirror for ${arch}; trying apt as configured" >&2
    else
      codename="$(awk -F= '$1 == "VERSION_CODENAME" { print $2 }' /etc/os-release)"
      if mirror="$(pick_mirror "${codename}")"; then
        echo "using ${mirror}"
        as_root bash -c "$(declare -f rewrite_sources); rewrite_sources '${sources_file}' '${mirror}' '${mirrors_file}'"
      else
        echo "no fallback mirror serves ${codename}; trying apt as configured" >&2
      fi
    fi
    as_root apt-get update
  fi
  as_root apt-get install -y "$@"
}

main "$@"
