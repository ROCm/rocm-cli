#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 <dist-name> [output-dir] [target-triple]" >&2
  exit 1
fi

DIST_NAME="$1"
OUTPUT_DIR="${2:-dist}"
TARGET_TRIPLE="${3:-}"
ROOT_DIR="${OUTPUT_DIR}/${DIST_NAME}"
ARCHIVE_PATH="${OUTPUT_DIR}/${DIST_NAME}.tar.gz"
TAR_PATH="${OUTPUT_DIR}/${DIST_NAME}.tar"
CARGO_TARGET_ROOT="${CARGO_TARGET_DIR:-target}"
BINARY_DIR="${CARGO_TARGET_ROOT}/release"

if [[ -n "${TARGET_TRIPLE}" ]]; then
  BINARY_DIR="${CARGO_TARGET_ROOT}/${TARGET_TRIPLE}/release"
fi

mkdir -p "${OUTPUT_DIR}"
rm -rf "${ROOT_DIR}"
rm -f "${ARCHIVE_PATH}" "${ARCHIVE_PATH}.sha256" "${ARCHIVE_PATH}.sig" "${TAR_PATH}"
mkdir -p "${ROOT_DIR}/bin"

if [[ ! -x "${BINARY_DIR}/rocm-codex" ]]; then
  chmod +x scripts/build-vendored-codex.sh
  PROFILE="release"
  if [[ "${BINARY_DIR}" == *"/debug" ]]; then
    PROFILE="debug"
  fi
  ./scripts/build-vendored-codex.sh "${PROFILE}" "${TARGET_TRIPLE}"
fi

cp "${BINARY_DIR}/rocm" "${ROOT_DIR}/bin/"
cp "${BINARY_DIR}/rocmd" "${ROOT_DIR}/bin/"
cp "${BINARY_DIR}/rocm-engine-pytorch" "${ROOT_DIR}/bin/"
cp "${BINARY_DIR}/rocm-engine-llama-cpp" "${ROOT_DIR}/bin/"
cp "${BINARY_DIR}/rocm-engine-lemonade" "${ROOT_DIR}/bin/"
cp "${BINARY_DIR}/rocm-engine-atom" "${ROOT_DIR}/bin/"
cp "${BINARY_DIR}/rocm-engine-vllm" "${ROOT_DIR}/bin/"
cp "${BINARY_DIR}/rocm-engine-sglang" "${ROOT_DIR}/bin/"
cp "${BINARY_DIR}/rocm-codex" "${ROOT_DIR}/bin/"
cp README.md LICENSE install.sh "${ROOT_DIR}/"

(cd "${OUTPUT_DIR}" && tar -cf "${DIST_NAME}.tar" "${DIST_NAME}")
gzip -c "${TAR_PATH}" > "${ARCHIVE_PATH}"
rm -f "${TAR_PATH}"
(cd "${OUTPUT_DIR}" && sha256sum "${DIST_NAME}.tar.gz" > "${DIST_NAME}.tar.gz.sha256")

SIGNATURE_REQUIRED="${ROCM_CLI_REQUIRE_SIGNATURE:-0}"
SIGNATURE_AVAILABLE=0
if [[ -n "${ROCM_CLI_SIGNING_PRIVATE_KEY_PATH:-}" || -n "${ROCM_CLI_SIGNING_PRIVATE_KEY_PEM:-}" ]]; then
  SIGNATURE_AVAILABLE=1
fi
if [[ "${SIGNATURE_REQUIRED}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ && "${SIGNATURE_AVAILABLE}" -eq 0 ]]; then
  echo "signature is required but ROCM_CLI_SIGNING_PRIVATE_KEY_PATH/PEM is not configured" >&2
  exit 1
fi

if [[ "${SIGNATURE_AVAILABLE}" -eq 1 ]]; then
  command -v openssl >/dev/null 2>&1 || {
    echo "missing required command for signing: openssl" >&2
    exit 1
  }
  SIGNING_TMP_DIR="${OUTPUT_DIR}/.signing-tmp-$$"
  rm -rf "${SIGNING_TMP_DIR}"
  mkdir -p "${SIGNING_TMP_DIR}"
  if [[ -n "${ROCM_CLI_SIGNING_PRIVATE_KEY_PATH:-}" ]]; then
    SIGNING_PRIVATE_KEY="${ROCM_CLI_SIGNING_PRIVATE_KEY_PATH}"
  else
    SIGNING_PRIVATE_KEY="${SIGNING_TMP_DIR}/rocm-cli-signing-private-key.pem"
    printf '%s\n' "${ROCM_CLI_SIGNING_PRIVATE_KEY_PEM}" > "${SIGNING_PRIVATE_KEY}"
  fi
  openssl dgst -sha256 -sign "${SIGNING_PRIVATE_KEY}" -out "${ARCHIVE_PATH}.sig" "${ARCHIVE_PATH}"
  rm -rf "${SIGNING_TMP_DIR}"
fi

if [[ "${SIGNATURE_REQUIRED}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ && ! -f "${ARCHIVE_PATH}.sig" ]]; then
  echo "signature is required but ${ARCHIVE_PATH}.sig was not produced" >&2
  exit 1
fi

echo "created:"
echo "  ${ARCHIVE_PATH}"
echo "  ${ARCHIVE_PATH}.sha256"
if [[ -f "${ARCHIVE_PATH}.sig" ]]; then
  echo "  ${ARCHIVE_PATH}.sig"
fi
