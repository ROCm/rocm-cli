#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "usage: $0 [debug|release] [target-triple]" >&2
  exit 1
fi

PROFILE="${1:-release}"
TARGET_TRIPLE="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

case "${PROFILE}" in
  debug|release) ;;
  *)
    echo "invalid profile: ${PROFILE} (expected debug or release)" >&2
    exit 1
    ;;
esac

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

need_cmd cargo

if [[ "$(uname -s)" == "Linux" ]]; then
  need_cmd pkg-config
  LOCAL_DEV_SYSROOT="${ROCM_CLI_PORTABLE_BUILD_DEPS_ROOT:-${REPO_ROOT}/.rocm-work/tools/wsl-build-deps}/root"
  LOCAL_PKG_CONFIG_PATH="$(
    find "${LOCAL_DEV_SYSROOT}/usr/lib" -path '*/pkgconfig' -type d -print 2>/dev/null | paste -sd: || true
  )"
  if ! pkg-config --exists libcap openssl && [[ -n "${LOCAL_PKG_CONFIG_PATH}" ]]; then
    export PKG_CONFIG_PATH="${LOCAL_PKG_CONFIG_PATH}:${PKG_CONFIG_PATH:-}"
    export PKG_CONFIG_SYSROOT_DIR="${LOCAL_DEV_SYSROOT}"
  fi
  if ! pkg-config --exists libcap openssl; then
    cat >&2 <<'EOF'
vendored Codex build prerequisites are missing on this Linux host.

Install `pkg-config` plus the `libcap` and OpenSSL development packages, then rerun the build.
Examples:
  Debian/Ubuntu: sudo apt install pkg-config libcap-dev libssl-dev
  Fedora/RHEL:  sudo dnf install pkgconf-pkg-config libcap-devel openssl-devel
Without sudo on WSL, run:
  bash scripts/setup-wsl-portable-build-deps.sh
EOF
    exit 1
  fi
fi

CODEX_MANIFEST="${REPO_ROOT}/third_party/openai-codex/codex-rs/Cargo.toml"

if [[ ! -f "${CODEX_MANIFEST}" ]]; then
  echo "vendored Codex manifest not found: ${CODEX_MANIFEST}" >&2
  exit 1
fi

echo "building vendored Codex TUI"
echo "  manifest: ${CODEX_MANIFEST}"
echo "  profile: ${PROFILE}"
if [[ -n "${TARGET_TRIPLE}" ]]; then
  echo "  target: ${TARGET_TRIPLE}"
fi

BUILD_ARGS=(build --manifest-path "${CODEX_MANIFEST}" -p codex-cli --bin codex)
if [[ "${PROFILE}" == "release" ]]; then
  BUILD_ARGS+=(--release)
fi
if [[ -n "${TARGET_TRIPLE}" ]]; then
  BUILD_ARGS+=(--target "${TARGET_TRIPLE}")
fi

(cd "${REPO_ROOT}" && cargo "${BUILD_ARGS[@]}")

if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
  if [[ "${CARGO_TARGET_DIR}" = /* ]]; then
    SHARED_TARGET_DIR="${CARGO_TARGET_DIR}"
  else
    SHARED_TARGET_DIR="${REPO_ROOT}/${CARGO_TARGET_DIR}"
  fi
  CODEX_TARGET_DIR="${SHARED_TARGET_DIR}"
  ROCM_TARGET_DIR="${SHARED_TARGET_DIR}"
else
  CODEX_TARGET_DIR="${REPO_ROOT}/third_party/openai-codex/codex-rs/target"
  ROCM_TARGET_DIR="${REPO_ROOT}/target"
fi

if [[ -n "${TARGET_TRIPLE}" ]]; then
  CODEX_BINARY="${CODEX_TARGET_DIR}/${TARGET_TRIPLE}/${PROFILE}/codex"
  ROCM_PROFILE_DIR="${ROCM_TARGET_DIR}/${TARGET_TRIPLE}/${PROFILE}"
else
  CODEX_BINARY="${CODEX_TARGET_DIR}/${PROFILE}/codex"
  ROCM_PROFILE_DIR="${ROCM_TARGET_DIR}/${PROFILE}"
fi

if [[ ! -x "${CODEX_BINARY}" ]]; then
  echo "vendored Codex binary not found after build: ${CODEX_BINARY}" >&2
  exit 1
fi

mkdir -p "${ROCM_PROFILE_DIR}"
install -m 0755 "${CODEX_BINARY}" "${ROCM_PROFILE_DIR}/rocm-codex"

echo "  installed wrapper binary: ${ROCM_PROFILE_DIR}/rocm-codex"
