#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

fail() {
  echo "acceptance failed: $*" >&2
  exit 1
}

assert_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "expected file: ${path}"
}

assert_missing() {
  local path="$1"
  [[ ! -e "${path}" ]] || fail "expected path to be removed: ${path}"
}

need_cmd cargo
need_cmd script
need_cmd grep
need_cmd openssl
need_cmd pkg-config
need_cmd timeout

ensure_linux_build_deps_pkg_config() {
  if pkg-config --exists libcap openssl; then
    return
  fi

  local pc_dir
  local deps_root
  deps_root="${ROCM_CLI_PORTABLE_BUILD_DEPS_ROOT:-${REPO_ROOT}/.rocm-work/tools/wsl-build-deps}"
  pc_dir="$(bash "${REPO_ROOT}/scripts/setup-wsl-portable-build-deps.sh")"
  export PKG_CONFIG_PATH="${pc_dir}:${PKG_CONFIG_PATH:-}"
  export PKG_CONFIG_SYSROOT_DIR="${deps_root}/root"

  if ! pkg-config --exists libcap openssl; then
    fail "portable WSL build dependency setup did not make libcap and openssl visible to pkg-config"
  fi
}

ensure_linux_build_deps_pkg_config

TMP_ROOT="${ROCM_CLI_ACCEPTANCE_ROOT:-${REPO_ROOT}/.rocm-work/acceptance-linux}"
rm -rf "${TMP_ROOT}"
mkdir -p "${TMP_ROOT}"
cleanup() {
  if [[ "${ROCM_CLI_KEEP_ACCEPTANCE_ROOT:-0}" != "1" ]]; then
    rm -rf "${TMP_ROOT}"
  fi
}
trap cleanup EXIT INT TERM

DIST_NAME="rocm-cli-linux-amd64"
DIST_DIR="${TMP_ROOT}/dist"
PEM_DIST_NAME="${DIST_NAME}"
PEM_DIST_DIR="${TMP_ROOT}/pem-dist"
INSTALL_DIR="${TMP_ROOT}/install/bin"
PEM_INSTALL_DIR="${TMP_ROOT}/pem-install/bin"
HOME_DIR="${TMP_ROOT}/home"
CONFIG_DIR="${TMP_ROOT}/rocm-config"
DATA_DIR="${TMP_ROOT}/rocm-data"
CACHE_DIR="${TMP_ROOT}/rocm-cache"
XDG_CONFIG_HOME="${TMP_ROOT}/xdg/config"
XDG_DATA_HOME="${TMP_ROOT}/xdg/data"
XDG_CACHE_HOME="${TMP_ROOT}/xdg/cache"
DOWNLOAD_BASE="file://${DIST_DIR}"
TUI_LOG="${TMP_ROOT}/tui.log"
INSTALL_LOG_1="${TMP_ROOT}/install-1.log"
INSTALL_LOG_2="${TMP_ROOT}/install-2.log"
CHECKSUM_LOG="${TMP_ROOT}/checksum-failure.log"
SIGNATURE_LOG="${TMP_ROOT}/signature-failure.log"
MISSING_SIGNATURE_LOG="${TMP_ROOT}/missing-signature-failure.log"
NO_PUBLIC_KEY_LOG="${TMP_ROOT}/no-public-key-failure.log"
PEM_INSTALL_LOG="${TMP_ROOT}/pem-install.log"
UNINSTALL_LOG="${TMP_ROOT}/uninstall.log"
CONFIG_FILE="${CONFIG_DIR}/config.json"
INSTALL_CONFIG_FILE="${HOME_DIR}/.rocm/config.json"
BASHRC_FILE="${HOME_DIR}/.bashrc"
SIGNING_PRIVATE_KEY="${TMP_ROOT}/signing-private.pem"
SIGNING_PUBLIC_KEY="${TMP_ROOT}/signing-public.pem"

expect_failure() {
  local label="$1"
  local log="$2"
  shift 2
  set +e
  "$@" >"${log}" 2>&1
  local status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then
    cat "${log}" >&2
    fail "${label} unexpectedly succeeded"
  fi
}

echo "acceptance: build release binaries"
(cd "${REPO_ROOT}" && cargo build --release -p rocm -p rocmd -p rocm-engine-pytorch -p rocm-engine-llama-cpp -p rocm-engine-lemonade -p rocm-engine-atom -p rocm-engine-vllm -p rocm-engine-sglang)

echo "acceptance: generate signing key"
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "${SIGNING_PRIVATE_KEY}" >/dev/null 2>&1 \
  || fail "failed to generate acceptance signing private key"
openssl rsa -in "${SIGNING_PRIVATE_KEY}" -pubout -out "${SIGNING_PUBLIC_KEY}" >/dev/null 2>&1 \
  || fail "failed to generate acceptance signing public key"

echo "acceptance: package local release bundle"
(cd "${REPO_ROOT}" && \
  ROCM_CLI_SIGNING_PRIVATE_KEY_PATH="${SIGNING_PRIVATE_KEY}" \
  ROCM_CLI_REQUIRE_SIGNATURE=1 \
  ./scripts/package-linux-release.sh "${DIST_NAME}" "${DIST_DIR}")

echo "acceptance: package with generated private key PEM"
(cd "${REPO_ROOT}" && \
  ROCM_CLI_SIGNING_PRIVATE_KEY_PEM="$(cat "${SIGNING_PRIVATE_KEY}")" \
  ROCM_CLI_REQUIRE_SIGNATURE=1 \
  ./scripts/package-linux-release.sh "${PEM_DIST_NAME}" "${PEM_DIST_DIR}")

run_installer() {
  local download_base="${1:-${DOWNLOAD_BASE}}"
  local install_dir="${2:-${INSTALL_DIR}}"
  (
    cd "${REPO_ROOT}"
    HOME="${HOME_DIR}" \
    SHELL="/bin/bash" \
    PATH="/usr/bin:/bin" \
    ROCM_CLI_DOWNLOAD_BASE="${download_base}" \
    ROCM_CLI_INSTALL_DIR="${install_dir}" \
    ROCM_CLI_SIGNING_PUBLIC_KEY_PATH="${SIGNING_PUBLIC_KEY}" \
    ROCM_CLI_REQUIRE_SIGNATURE=1 \
    sh ./install.sh release
  )
}

run_installer_with_public_key_pem() {
  local download_base="$1"
  local install_dir="$2"
  (
    cd "${REPO_ROOT}"
    HOME="${HOME_DIR}" \
    SHELL="/bin/bash" \
    PATH="/usr/bin:/bin" \
    ROCM_CLI_DOWNLOAD_BASE="${download_base}" \
    ROCM_CLI_INSTALL_DIR="${install_dir}" \
    ROCM_CLI_SIGNING_PUBLIC_KEY_PEM="$(cat "${SIGNING_PUBLIC_KEY}")" \
    ROCM_CLI_REQUIRE_SIGNATURE=1 \
    ROCM_CLI_UPDATE_SHELL_PATH=0 \
    sh ./install.sh release
  )
}

run_installer_without_public_key() {
  local download_base="$1"
  local install_dir="$2"
  (
    cd "${REPO_ROOT}"
    env -u ROCM_CLI_SIGNING_PUBLIC_KEY_PATH -u ROCM_CLI_SIGNING_PUBLIC_KEY_PEM \
      HOME="${HOME_DIR}" \
      SHELL="/bin/bash" \
      PATH="/usr/bin:/bin" \
      ROCM_CLI_DOWNLOAD_BASE="${download_base}" \
      ROCM_CLI_INSTALL_DIR="${install_dir}" \
      ROCM_CLI_REQUIRE_SIGNATURE=1 \
      ROCM_CLI_UPDATE_SHELL_PATH=0 \
      sh ./install.sh release
  )
}

echo "acceptance: install with generated public key PEM"
PEM_DOWNLOAD_BASE="file://${PEM_DIST_DIR}"
run_installer_with_public_key_pem "${PEM_DOWNLOAD_BASE}" "${PEM_INSTALL_DIR}" | tee "${PEM_INSTALL_LOG}"
grep -q "signature verified" "${PEM_INSTALL_LOG}" \
  || fail "installer did not report PEM signature verification"
assert_file "${PEM_INSTALL_DIR}/rocm"
assert_file "${PEM_INSTALL_DIR}/.rocm-cli-manifest"

echo "acceptance: reject required signature without public key"
NO_PUBLIC_KEY_INSTALL_DIR="${TMP_ROOT}/no-public-key-install/bin"
expect_failure \
  "acceptance: required signature no public key install" \
  "${NO_PUBLIC_KEY_LOG}" \
  run_installer_without_public_key "${DOWNLOAD_BASE}" "${NO_PUBLIC_KEY_INSTALL_DIR}"
grep -q "signature verification requires ROCM_CLI_SIGNING_PUBLIC_KEY_PATH or ROCM_CLI_SIGNING_PUBLIC_KEY_PEM" "${NO_PUBLIC_KEY_LOG}" \
  || fail "installer did not report missing public key for required signature"
assert_missing "${NO_PUBLIC_KEY_INSTALL_DIR}/rocm"
assert_missing "${NO_PUBLIC_KEY_INSTALL_DIR}/.rocm-cli-manifest"

echo "acceptance: reject mismatched checksum before activation"
BAD_CHECKSUM_DIST_DIR="${TMP_ROOT}/bad-checksum-dist"
BAD_CHECKSUM_INSTALL_DIR="${TMP_ROOT}/bad-checksum-install/bin"
mkdir -p "${BAD_CHECKSUM_DIST_DIR}"
cp "${DIST_DIR}/${DIST_NAME}.tar.gz" "${BAD_CHECKSUM_DIST_DIR}/${DIST_NAME}.tar.gz"
cp "${DIST_DIR}/${DIST_NAME}.tar.gz.sig" "${BAD_CHECKSUM_DIST_DIR}/${DIST_NAME}.tar.gz.sig"
printf '%064d  %s.tar.gz\n' 0 "${DIST_NAME}" > "${BAD_CHECKSUM_DIST_DIR}/${DIST_NAME}.tar.gz.sha256"
expect_failure \
  "acceptance: checksum mismatch install" \
  "${CHECKSUM_LOG}" \
  run_installer "file://${BAD_CHECKSUM_DIST_DIR}" "${BAD_CHECKSUM_INSTALL_DIR}"
grep -q "checksum verification failed" "${CHECKSUM_LOG}" \
  || fail "installer did not report checksum verification failure"
assert_missing "${BAD_CHECKSUM_INSTALL_DIR}/rocm"
assert_missing "${BAD_CHECKSUM_INSTALL_DIR}/.rocm-cli-manifest"

echo "acceptance: reject mismatched signature before activation"
BAD_SIGNATURE_DIST_DIR="${TMP_ROOT}/bad-signature-dist"
BAD_SIGNATURE_INSTALL_DIR="${TMP_ROOT}/bad-signature-install/bin"
mkdir -p "${BAD_SIGNATURE_DIST_DIR}"
cp "${DIST_DIR}/${DIST_NAME}.tar.gz" "${BAD_SIGNATURE_DIST_DIR}/${DIST_NAME}.tar.gz"
cp "${DIST_DIR}/${DIST_NAME}.tar.gz.sha256" "${BAD_SIGNATURE_DIST_DIR}/${DIST_NAME}.tar.gz.sha256"
printf '%s\n' "not a real signature" > "${BAD_SIGNATURE_DIST_DIR}/${DIST_NAME}.tar.gz.sig"
expect_failure \
  "acceptance: signature mismatch install" \
  "${SIGNATURE_LOG}" \
  run_installer "file://${BAD_SIGNATURE_DIST_DIR}" "${BAD_SIGNATURE_INSTALL_DIR}"
grep -q "signature verification failed" "${SIGNATURE_LOG}" \
  || fail "installer did not report signature verification failure"
assert_missing "${BAD_SIGNATURE_INSTALL_DIR}/rocm"
assert_missing "${BAD_SIGNATURE_INSTALL_DIR}/.rocm-cli-manifest"

echo "acceptance: reject missing signature before activation"
MISSING_SIGNATURE_DIST_DIR="${TMP_ROOT}/missing-signature-dist"
MISSING_SIGNATURE_INSTALL_DIR="${TMP_ROOT}/missing-signature-install/bin"
mkdir -p "${MISSING_SIGNATURE_DIST_DIR}"
cp "${DIST_DIR}/${DIST_NAME}.tar.gz" "${MISSING_SIGNATURE_DIST_DIR}/${DIST_NAME}.tar.gz"
cp "${DIST_DIR}/${DIST_NAME}.tar.gz.sha256" "${MISSING_SIGNATURE_DIST_DIR}/${DIST_NAME}.tar.gz.sha256"
expect_failure \
  "acceptance: missing signature install" \
  "${MISSING_SIGNATURE_LOG}" \
  run_installer "file://${MISSING_SIGNATURE_DIST_DIR}" "${MISSING_SIGNATURE_INSTALL_DIR}"
grep -q "required signature sidecar is missing or unavailable" "${MISSING_SIGNATURE_LOG}" \
  || fail "installer did not report missing signature download failure"
assert_missing "${MISSING_SIGNATURE_INSTALL_DIR}/rocm"
assert_missing "${MISSING_SIGNATURE_INSTALL_DIR}/.rocm-cli-manifest"

echo "acceptance: first install"
run_installer | tee "${INSTALL_LOG_1}"
grep -q "signature verified" "${INSTALL_LOG_1}" \
  || fail "installer did not report signature verification"
grep -q "shell profile updated" "${INSTALL_LOG_1}" \
  || fail "installer did not report automatic shell profile setup"
assert_file "${INSTALL_DIR}/rocm"
assert_file "${INSTALL_DIR}/rocmd"
assert_file "${INSTALL_DIR}/rocm-engine-pytorch"
assert_file "${INSTALL_DIR}/rocm-engine-llama-cpp"
assert_file "${INSTALL_DIR}/rocm-engine-lemonade"
assert_file "${INSTALL_DIR}/rocm-engine-atom"
assert_file "${INSTALL_DIR}/rocm-engine-vllm"
assert_file "${INSTALL_DIR}/rocm-engine-sglang"
assert_file "${INSTALL_DIR}/.rocm-cli-manifest"
assert_file "${INSTALL_CONFIG_FILE}"
grep -q '"default_engine"[[:space:]]*:[[:space:]]*"pytorch"' "${INSTALL_CONFIG_FILE}" \
  || fail "installer did not seed minimal config with the expected default engine"
assert_file "${BASHRC_FILE}"
grep -F "${INSTALL_DIR}" "${BASHRC_FILE}" >/dev/null \
  || fail "installer did not add install dir to the shell profile"

echo "acceptance: simulate stale prior install entry and reinstall"
printf '%s\n' '{"default_engine":"llama.cpp"}' > "${INSTALL_CONFIG_FILE}"
echo "stale" > "${INSTALL_DIR}/rocm-engine-stale"
echo "${INSTALL_DIR}/rocm-engine-stale" >> "${INSTALL_DIR}/.rocm-cli-manifest"
run_installer | tee "${INSTALL_LOG_2}"
assert_missing "${INSTALL_DIR}/rocm-engine-stale"
assert_file "${INSTALL_DIR}/.rocm-cli-manifest"
grep -q '"default_engine"[[:space:]]*:[[:space:]]*"llama.cpp"' "${INSTALL_CONFIG_FILE}" \
  || fail "installer overwrote an existing config file"
grep -q "removing previous rocm-cli install" "${INSTALL_LOG_2}" \
  || fail "installer did not report removal of previous install"
MARKER_COUNT="$(grep -c '# >>> rocm-cli path >>>' "${BASHRC_FILE}")"
[[ "${MARKER_COUNT}" -eq 1 ]] || fail "installer duplicated the shell PATH snippet"

echo "acceptance: drive the TUI through a pseudo-terminal"
env \
  ROCM_CLI_CONFIG_DIR="${CONFIG_DIR}" \
  ROCM_CLI_DATA_DIR="${DATA_DIR}" \
  ROCM_CLI_CACHE_DIR="${CACHE_DIR}" \
  XDG_CONFIG_HOME="${XDG_CONFIG_HOME}" \
  XDG_DATA_HOME="${XDG_DATA_HOME}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  "${INSTALL_DIR}/rocm" config set-default-engine pytorch >/dev/null

tui_command="$(
  printf '%q ' \
    env \
    "ROCM_CLI_CONFIG_DIR=${CONFIG_DIR}" \
    "ROCM_CLI_DATA_DIR=${DATA_DIR}" \
    "ROCM_CLI_CACHE_DIR=${CACHE_DIR}" \
    "XDG_CONFIG_HOME=${XDG_CONFIG_HOME}" \
    "XDG_DATA_HOME=${XDG_DATA_HOME}" \
    "XDG_CACHE_HOME=${XDG_CACHE_HOME}" \
    "${INSTALL_DIR}/rocm" \
    chat \
    --provider \
    openai
)"
tui_command="stty rows 40 cols 120; exec ${tui_command}"
set +e
(
  sleep 1
  printf 'q'
) | timeout 20s script -q -e -f -c "${tui_command}" "${TUI_LOG}"
tui_status=$?
set -e
if [[ "${tui_status}" -ne 0 ]]; then
  fail "TUI smoke exited with status ${tui_status}"
fi

assert_file "${CONFIG_FILE}"
grep -q '"default_engine"[[:space:]]*:[[:space:]]*"pytorch"' "${CONFIG_FILE}" \
  || fail "config smoke did not persist the expected default engine"
assert_file "${TUI_LOG}"
[[ -s "${TUI_LOG}" ]] || fail "TUI smoke log was empty"

echo "acceptance: uninstall from the installed binary"
env \
  ROCM_CLI_CONFIG_DIR="${CONFIG_DIR}" \
  ROCM_CLI_DATA_DIR="${DATA_DIR}" \
  ROCM_CLI_CACHE_DIR="${CACHE_DIR}" \
  XDG_CONFIG_HOME="${XDG_CONFIG_HOME}" \
  XDG_DATA_HOME="${XDG_DATA_HOME}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  "${INSTALL_DIR}/rocm" uninstall --yes | tee "${UNINSTALL_LOG}"

assert_missing "${INSTALL_DIR}/rocm"
assert_missing "${INSTALL_DIR}/rocmd"
assert_missing "${INSTALL_DIR}/rocm-engine-pytorch"
assert_missing "${INSTALL_DIR}/rocm-engine-llama-cpp"
assert_missing "${INSTALL_DIR}/rocm-engine-lemonade"
assert_missing "${INSTALL_DIR}/rocm-engine-atom"
assert_missing "${INSTALL_DIR}/rocm-engine-vllm"
assert_missing "${INSTALL_DIR}/rocm-engine-sglang"
assert_missing "${INSTALL_DIR}/.rocm-cli-manifest"
assert_missing "${XDG_CONFIG_HOME}/rocm-cli"
assert_missing "${XDG_DATA_HOME}/rocm-cli"
assert_missing "${XDG_CACHE_HOME}/rocm-cli"

echo "acceptance: ok"
