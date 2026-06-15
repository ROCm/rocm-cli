#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="${ROCM_CLI_PORTABLE_BUILD_DEPS_ROOT:-${REPO_ROOT}/.rocm-work/tools/wsl-build-deps}"
CACHE_DIR="${ROOT}/cache"
EXTRACT_DIR="${ROOT}/root"
PACKAGES=(
  libcap-dev
  libssl-dev
)

usage() {
  cat >&2 <<'EOF'
usage: setup-wsl-portable-build-deps.sh [--self-test]

Downloads libcap/OpenSSL development packages into a workspace-local WSL sysroot
and prints the pkg-config directory list for callers.

  --self-test  run an offline fake-sysroot test without apt/network access
EOF
}

fail() {
  echo "portable WSL build dependency setup failed: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

pkg_config_path() {
  find "${EXTRACT_DIR}/usr/lib" -path '*/pkgconfig' -type d -print 2>/dev/null \
    | paste -sd:
}

normalize_pkg_config_libdirs() {
  local multiarch_lib
  local multiarch_dir
  local relative_dir
  multiarch_lib="$(find "${EXTRACT_DIR}/usr/lib" -maxdepth 2 -name 'libcap.so' -print -quit 2>/dev/null || true)"
  if [[ -n "${multiarch_lib}" && ! -e "${EXTRACT_DIR}/usr/lib64" ]]; then
    multiarch_dir="$(dirname "${multiarch_lib}")"
    relative_dir="$(realpath --relative-to="${EXTRACT_DIR}/usr" "${multiarch_dir}")"
    ln -s "${relative_dir}" "${EXTRACT_DIR}/usr/lib64"
  fi
}

normalize_openssl_headers() {
  local multiarch_openssl_dir
  multiarch_openssl_dir="$(
    find "${EXTRACT_DIR}/usr/include" -path '*/openssl/opensslconf.h' -print0 -quit 2>/dev/null \
      | xargs -0 -r dirname
  )"
  if [[ -z "${multiarch_openssl_dir}" || "${multiarch_openssl_dir}" == "${EXTRACT_DIR}/usr/include/openssl" ]]; then
    return
  fi
  mkdir -p "${EXTRACT_DIR}/usr/include/openssl"
  for header in configuration.h opensslconf.h; do
    if [[ -f "${multiarch_openssl_dir}/${header}" && ! -e "${EXTRACT_DIR}/usr/include/openssl/${header}" ]]; then
      ln -s "../$(realpath --relative-to="${EXTRACT_DIR}/usr/include" "${multiarch_openssl_dir}")/${header}" \
        "${EXTRACT_DIR}/usr/include/openssl/${header}"
    fi
  done
}

deps_are_ready() {
  local pc_path
  normalize_pkg_config_libdirs
  normalize_openssl_headers
  pc_path="$(pkg_config_path)"
  [[ -n "${pc_path}" ]] || return 1
  [[ -f "${EXTRACT_DIR}/usr/include/sys/capability.h" ]] || return 1
  [[ -f "${EXTRACT_DIR}/usr/include/openssl/ssl.h" ]] || return 1
  PKG_CONFIG_LIBDIR="${pc_path}" PKG_CONFIG_PATH="${pc_path}" PKG_CONFIG_SYSROOT_DIR="${EXTRACT_DIR}" pkg-config --exists libcap openssl
}

run_self_test() {
  need_cmd pkg-config
  need_cmd realpath

  ROOT="${ROCM_CLI_PORTABLE_BUILD_DEPS_SELF_TEST_ROOT:-${REPO_ROOT}/.rocm-work/script-self-tests/wsl-build-deps}"
  CACHE_DIR="${ROOT}/cache"
  EXTRACT_DIR="${ROOT}/root"
  rm -rf "${ROOT}"

  local pc_dir
  local openssl_multiarch_dir
  pc_dir="${EXTRACT_DIR}/usr/lib/x86_64-linux-gnu/pkgconfig"
  openssl_multiarch_dir="${EXTRACT_DIR}/usr/include/x86_64-linux-gnu/openssl"
  mkdir -p \
    "${CACHE_DIR}" \
    "${pc_dir}" \
    "${EXTRACT_DIR}/usr/lib/x86_64-linux-gnu" \
    "${EXTRACT_DIR}/usr/include/sys" \
    "${EXTRACT_DIR}/usr/include/openssl" \
    "${openssl_multiarch_dir}"

  printf '%s\n' '/* fake libcap header */' >"${EXTRACT_DIR}/usr/include/sys/capability.h"
  printf '%s\n' '/* fake ssl header */' >"${EXTRACT_DIR}/usr/include/openssl/ssl.h"
  printf '%s\n' '/* fake opensslconf header */' >"${openssl_multiarch_dir}/opensslconf.h"
  printf '%s\n' '/* fake configuration header */' >"${openssl_multiarch_dir}/configuration.h"
  printf '%s\n' 'fake libcap' >"${EXTRACT_DIR}/usr/lib/x86_64-linux-gnu/libcap.so"

  cat >"${pc_dir}/libcap.pc" <<'EOF'
prefix=/usr
libdir=${prefix}/lib64
includedir=${prefix}/include

Name: libcap
Description: fake libcap for rocm-cli self-test
Version: 2.0
Libs: -L${libdir} -lcap
Cflags: -I${includedir}
EOF

  cat >"${pc_dir}/openssl.pc" <<'EOF'
prefix=/usr
libdir=${prefix}/lib64
includedir=${prefix}/include

Name: OpenSSL
Description: fake OpenSSL for rocm-cli self-test
Version: 3.0.0
Libs: -L${libdir} -lssl -lcrypto
Cflags: -I${includedir}
EOF

  deps_are_ready || fail "self-test fake sysroot was not detected as ready"
  [[ -L "${EXTRACT_DIR}/usr/lib64" ]] || fail "self-test did not create usr/lib64 multiarch link"
  [[ -L "${EXTRACT_DIR}/usr/include/openssl/opensslconf.h" ]] || fail "self-test did not create opensslconf.h link"
  [[ -L "${EXTRACT_DIR}/usr/include/openssl/configuration.h" ]] || fail "self-test did not create configuration.h link"

  rm -f "${pc_dir}/openssl.pc"
  if deps_are_ready; then
    fail "self-test missing openssl.pc was incorrectly accepted"
  fi

  rm -rf "${ROOT}"
  echo "portable WSL build dependency self-test ok"
}

if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi

case "${1:-}" in
  --self-test)
    run_self_test
    exit 0
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    usage
    exit 2
    ;;
esac

if deps_are_ready; then
  pkg_config_path
  exit 0
fi

need_cmd apt-get
need_cmd dpkg-deb
need_cmd pkg-config
need_cmd realpath

rm -rf "${ROOT}"
mkdir -p "${CACHE_DIR}" "${EXTRACT_DIR}"

for package_name in "${PACKAGES[@]}"; do
  (
    cd "${CACHE_DIR}"
    apt-get download "${package_name}" >"${package_name}.download.log" 2>&1 || {
      cat "${package_name}.download.log" >&2
      exit 1
    }
  )
done

while IFS= read -r deb_path; do
  dpkg-deb -x "${deb_path}" "${EXTRACT_DIR}"
done < <(find "${CACHE_DIR}" -maxdepth 1 -name '*.deb' -print)

if deps_are_ready; then
  pkg_config_path
  exit 0
fi

echo "portable WSL build dependencies did not expose libcap and openssl through pkg-config" >&2
exit 1
