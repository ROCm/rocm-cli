#!/usr/bin/env bash
set -euo pipefail

# Download a workspace-local Cosmopolitan compiler.
#
# The default URL follows Cosmopolitan's documented getting-started path. Keep
# this tool under .rocm-work so the repo and user home directory stay clean.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROCM_CLI_COSMOCC_TOOLS_DIR:-"${ROOT_DIR}/.rocm-work/tools/cosmocc"}"
WSL_ELF_DIR="${ROCM_CLI_COSMOCC_WSL_ELF_DIR:-"${ROOT_DIR}/.rocm-work/tools/cosmocc-wsl-elf"}"
URL="${ROCM_CLI_COSMOCC_URL:-"https://cosmo.zip/pub/cosmocc/cosmocc.zip"}"
ZIP_PATH="${TOOLS_DIR}/cosmocc.zip"

if [[ "${1:-}" == "--self-test" ]]; then
  echo "setup-cosmocc: tools dir: ${TOOLS_DIR}"
  echo "setup-cosmocc: wsl elf dir: ${WSL_ELF_DIR}"
  echo "setup-cosmocc: url: ${URL}"
  exit 0
fi

is_wsl() {
  [[ -n "${WSL_INTEROP:-}" || -n "${WSL_DISTRO_NAME:-}" ]] && return 0
  [[ -r /proc/sys/kernel/osrelease ]] && grep -qiE 'microsoft|wsl' /proc/sys/kernel/osrelease
}

is_ape_file() {
  local path="$1"
  local magic
  magic="$(od -An -tx1 -N6 "${path}" 2>/dev/null | tr -d ' \n')"
  [[ "${magic}" == "4d5a71467044" || "${magic}" == "6a6172747372" ]]
}

restore_tool_permissions() {
  local root="$1"
  if [[ -d "${root}/bin" ]]; then
    find "${root}/bin" -type f -exec chmod +x {} +
  fi
  if [[ -d "${root}/libexec" ]]; then
    find "${root}/libexec" -type f -exec chmod +x {} +
  fi
}

needs_elf_toolchain() {
  local root="$1"
  local tool
  for tool in \
    "${root}/bin/x86_64-linux-cosmo-ar" \
    "${root}/bin/x86_64-unknown-cosmo-cc" \
    "${root}/bin/cosmocross" \
    "${root}/bin/assimilate"; do
    if [[ -f "${tool}" ]] && is_ape_file "${tool}"; then
      return 0
    fi
  done
  return 1
}

prepare_wsl_elf_toolchain() {
  local src="$1"
  local dest="$2"
  local ape="${src}/bin/ape-x86_64.elf"
  local assimilate="${src}/bin/assimilate"
  local marker="${dest}/.rocm-cosmocc-wsl-elf-source"
  local expected="source=${src} url=${URL}"

  if [[ -f "${dest}/bin/cosmocc" && -f "${marker}" ]] && grep -qxF "${expected}" "${marker}"; then
    echo "${dest}/bin/cosmocc"
    return 0
  fi
  if [[ ! -x "${ape}" || ! -f "${assimilate}" ]]; then
    echo "setup-cosmocc: missing APE loader or assimilate in ${src}/bin" >&2
    exit 1
  fi

  echo "setup-cosmocc: preparing executable ELF toolchain in ${dest}" >&2
  rm -rf "${dest}"
  mkdir -p "${dest}"
  tar --exclude=./cosmocc.zip -C "${src}" -cf - . | tar -C "${dest}" -xf -

  while IFS= read -r -d '' file; do
    if is_ape_file "${file}"; then
      "${ape}" "${assimilate}" -e -c "${file}" >/dev/null
    fi
  done < <(find "${dest}/bin" "${dest}/libexec" -type f -perm /111 -print0)

  printf '%s\n' "${expected}" >"${marker}"
  chmod +x "${dest}/bin/cosmocc"
  echo "${dest}/bin/cosmocc"
}

mkdir -p "${TOOLS_DIR}"
if [[ ! -f "${TOOLS_DIR}/bin/cosmocc" && ! -f "${TOOLS_DIR}/bin/x86_64-unknown-cosmo-cc" ]]; then
  echo "setup-cosmocc: downloading ${URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "${ZIP_PATH}" "${URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${ZIP_PATH}" "${URL}"
  else
    echo "setup-cosmocc: curl or wget is required" >&2
    exit 1
  fi
  echo "setup-cosmocc: extracting into ${TOOLS_DIR}"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "${ZIP_PATH}" -d "${TOOLS_DIR}"
  else
    python3 - "${ZIP_PATH}" "${TOOLS_DIR}" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
dest = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as archive:
    for info in archive.infolist():
        archive.extract(info, dest)
        mode = (info.external_attr >> 16) & 0o777
        if mode:
            path = dest / info.filename
            if path.exists():
                path.chmod(mode)
PY
  fi
fi

restore_tool_permissions "${TOOLS_DIR}"

if [[ -f "${TOOLS_DIR}/bin/cosmocc" ]]; then
  COSMOCC="${TOOLS_DIR}/bin/cosmocc"
elif [[ -f "${TOOLS_DIR}/bin/x86_64-unknown-cosmo-cc" ]]; then
  COSMOCC="${TOOLS_DIR}/bin/x86_64-unknown-cosmo-cc"
else
  echo "setup-cosmocc: could not find cosmocc in ${TOOLS_DIR}/bin" >&2
  exit 1
fi

chmod +x "${COSMOCC}"
if is_wsl || needs_elf_toolchain "${TOOLS_DIR}"; then
  prepare_wsl_elf_toolchain "${TOOLS_DIR}" "${WSL_ELF_DIR}"
else
  echo "${COSMOCC}"
fi
