#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

ROCDXG_VERSION="${ROCDXG_VERSION:-1.2.0}"
ROCDXG_DEB="rocdxg-roct_${ROCDXG_VERSION}_amd64.deb"
ROCDXG_URL="${ROCDXG_URL:-https://github.com/ROCm/librocdxg/releases/download/v${ROCDXG_VERSION}/${ROCDXG_DEB}}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tmp}"
DEB_PATH="${DOWNLOAD_DIR}/${ROCDXG_DEB}"

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print tolower($1)}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${path}" | awk '{print tolower($1)}'
  else
    echo "error: missing sha256sum or shasum for ROCDXG checksum verification" >&2
    return 1
  fi
}

rocdxg_expected_sha256() {
  if [[ "${ROCDXG_SHA256}" =~ ^[[:space:]]*([0-9A-Fa-f]{64})[[:space:]]*$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1],,}"
    return 0
  fi

  echo "error: ROCDXG_SHA256 must contain a 64-character hex sha256 digest" >&2
  return 1
}

verify_rocdxg_checksum() {
  local path="$1"
  if [[ -z "${ROCDXG_SHA256:-}" ]]; then
    return 0
  fi

  local expected
  local actual
  expected="$(rocdxg_expected_sha256)" || return 1
  actual="$(sha256_file "${path}")" || return 1

  if [[ "${actual}" != "${expected}" ]]; then
    echo "error: ROCDXG checksum verification failed for ${path}" >&2
    echo "  expected: ${expected}" >&2
    echo "  actual:   ${actual}" >&2
    return 1
  fi

  echo "ROCDXG checksum verified: sha256=${actual}"
}

run_checksum_self_test() {
  local tmp
  local expected
  local status
  tmp="$(mktemp)"
  printf '%s' "rocdxg checksum self-test" > "${tmp}"

  expected="$(sha256_file "${tmp}")" || {
    rm -f "${tmp}"
    return 1
  }

  ROCDXG_SHA256="${expected}" verify_rocdxg_checksum "${tmp}" >/dev/null || {
    rm -f "${tmp}"
    echo "error: ROCDXG checksum self-test valid digest was rejected" >&2
    return 1
  }

  set +e
  ROCDXG_SHA256="${expected} extra-text" verify_rocdxg_checksum "${tmp}" >/dev/null 2>&1
  status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then
    rm -f "${tmp}"
    echo "error: ROCDXG checksum self-test malformed digest unexpectedly succeeded" >&2
    return 1
  fi

  set +e
  ROCDXG_SHA256="0000000000000000000000000000000000000000000000000000000000000000" verify_rocdxg_checksum "${tmp}" >/dev/null 2>&1
  status=$?
  set -e
  rm -f "${tmp}"

  if [[ "${status}" -eq 0 ]]; then
    echo "error: ROCDXG checksum self-test mismatch unexpectedly succeeded" >&2
    return 1
  fi

  echo "ROCDXG checksum self-test ok"
}

if [[ "${ROCDXG_CHECKSUM_SELF_TEST:-0}" == "1" ]]; then
  run_checksum_self_test
  exit $?
fi

if [[ ! -e /dev/dxg ]]; then
  echo "error: /dev/dxg is missing; WSL GPU plumbing is not available" >&2
  exit 1
fi

if [[ ! -e /usr/lib/wsl/lib/libdxcore.so ]]; then
  echo "error: /usr/lib/wsl/lib/libdxcore.so is missing" >&2
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "error: sudo is required to install ROCDXG under /opt/rocm" >&2
  exit 1
fi

echo "Installing ROCDXG ${ROCDXG_VERSION} from ${ROCDXG_URL}"
sudo apt-get update
sudo apt-get install -y ca-certificates curl
curl -L --fail --show-error --output "${DEB_PATH}" "${ROCDXG_URL}"
verify_rocdxg_checksum "${DEB_PATH}" || exit 1
sudo apt install -y "${DEB_PATH}"
sudo ldconfig

if [[ ! -e /opt/rocm/lib/librocdxg.so ]]; then
  echo "error: /opt/rocm/lib/librocdxg.so was not installed" >&2
  exit 1
fi

if ! ldconfig -p | grep -q 'librocdxg\.so'; then
  echo "error: librocdxg.so is not visible through ldconfig" >&2
  exit 1
fi

echo "ROCDXG installed and visible to ldconfig."
