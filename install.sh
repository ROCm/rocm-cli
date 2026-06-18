#!/bin/sh
set -eu

REPO="${ROCM_CLI_GITHUB_REPO:-powderluv/rocm-cli}"
CHANNEL="${1:-release}"
INSTALL_DIR="${ROCM_CLI_INSTALL_DIR:-${HOME}/.local/bin}"
UPDATE_SHELL_PATH="${ROCM_CLI_UPDATE_SHELL_PATH:-1}"

fail() {
  echo "rocm-cli installer: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

sha256_file() {
  file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${file}" | awk '{print $1}'
  else
    fail "missing sha256sum or shasum for checksum verification"
  fi
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

fetch() {
  url="$1"
  output="$2"
  failure_message="${3:-failed to download ${url}}"
  if command -v curl >/dev/null 2>&1; then
    if [ -t 2 ]; then
      curl -fL --progress-bar "${url}" -o "${output}" || {
        rm -f "${output}"
        fail "${failure_message}"
      }
    else
      curl -fsSL "${url}" -o "${output}" || {
        rm -f "${output}"
        fail "${failure_message}"
      }
    fi
  elif command -v wget >/dev/null 2>&1; then
    if [ -t 2 ]; then
      wget --show-progress -O "${output}" "${url}" || {
        rm -f "${output}"
        fail "${failure_message}"
      }
    else
      wget -qO "${output}" "${url}" || {
        rm -f "${output}"
        fail "${failure_message}"
      }
    fi
  else
    fail "missing curl or wget"
  fi
}

signing_public_key_path() {
  if [ -n "${ROCM_CLI_SIGNING_PUBLIC_KEY_PATH:-}" ]; then
    printf '%s\n' "${ROCM_CLI_SIGNING_PUBLIC_KEY_PATH}"
    return 0
  fi

  if [ -n "${ROCM_CLI_SIGNING_PUBLIC_KEY_PEM:-}" ]; then
    key_path="${tmp_dir}/rocm-cli-signing-public-key.pem"
    printf '%s\n' "${ROCM_CLI_SIGNING_PUBLIC_KEY_PEM}" > "${key_path}"
    printf '%s\n' "${key_path}"
    return 0
  fi

  printf '%s\n' ""
}

verify_signature() {
  archive="$1"
  signature="$2"
  public_key="$3"
  need_cmd openssl
  openssl dgst -sha256 -verify "${public_key}" -signature "${signature}" "${archive}" >/dev/null 2>&1 \
    || fail "signature verification failed"
}

installer_config_dir() {
  if [ -n "${ROCM_CLI_CONFIG_DIR:-}" ]; then
    printf '%s\n' "${ROCM_CLI_CONFIG_DIR}"
  else
    [ -n "${HOME:-}" ] || fail "unable to determine the user home directory for rocm-cli config"
    printf '%s\n' "${HOME}/.rocm"
  fi
}

write_minimal_config_if_missing() {
  config_dir="$(installer_config_dir)"
  config_path="${config_dir}/config.json"
  if [ -f "${config_path}" ]; then
    echo "config: existing ${config_path}"
    return
  fi

  mkdir -p "${config_dir}"
  config_tmp="${tmp_dir}/config.json"
  cat > "${config_tmp}" <<'JSON'
{
  "default_engine": "pytorch",
  "telemetry": {
    "mode": "local"
  },
  "permissions": {
    "mode": "ask"
  },
  "setup": {
    "completed": false
  }
}
JSON
  install -m 0600 "${config_tmp}" "${config_path}"
  echo "config: created ${config_path}"
}

need_cmd tar
need_cmd mkdir
need_cmd mktemp
need_cmd install
need_cmd rm
need_cmd grep
need_cmd sed

shell_name() {
  if [ -n "${ROCM_CLI_SHELL_NAME:-}" ]; then
    printf '%s\n' "${ROCM_CLI_SHELL_NAME}"
    return
  fi

  shell_path="${SHELL:-}"
  if [ -z "${shell_path}" ]; then
    printf '%s\n' "sh"
    return
  fi
  printf '%s\n' "${shell_path##*/}"
}

profile_path_for_shell() {
  if [ -n "${ROCM_CLI_SHELL_PROFILE:-}" ]; then
    printf '%s\n' "${ROCM_CLI_SHELL_PROFILE}"
    return
  fi

  case "$(shell_name)" in
    bash) printf '%s\n' "${HOME}/.bashrc" ;;
    zsh) printf '%s\n' "${HOME}/.zshrc" ;;
    fish) printf '%s\n' "${HOME}/.config/fish/config.fish" ;;
    ksh) printf '%s\n' "${HOME}/.kshrc" ;;
    *) printf '%s\n' "${HOME}/.profile" ;;
  esac
}

path_expr_for_profile() {
  path="$1"
  case "${path}" in
    "${HOME}")
      # Emit the literal string $HOME so the user's shell expands it later.
      # shellcheck disable=SC2016
      printf '%s\n' '$HOME'
      ;;
    "${HOME}"/*)
      printf '%s\n' "\$HOME/${path#"${HOME}"/}"
      ;;
    *)
      printf '%s\n' "${path}"
      ;;
  esac
}

escape_for_double_quotes() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

profile_has_path_entry() {
  profile="$1"
  path_expr="$2"
  [ -f "${profile}" ] || return 1
  grep -F "# >>> rocm-cli path >>>" "${profile}" >/dev/null 2>&1 && return 0
  grep -F "${path_expr}" "${profile}" >/dev/null 2>&1 && return 0
  grep -F "${INSTALL_DIR}" "${profile}" >/dev/null 2>&1 && return 0
  return 1
}

append_path_snippet() {
  profile="$1"
  shell_kind="$2"
  path_expr="$3"
  escaped_path_expr="$(escape_for_double_quotes "${path_expr}")"

  profile_dir="${profile%/*}"
  if [ "${profile_dir}" != "${profile}" ]; then
    mkdir -p "${profile_dir}"
  fi
  [ -f "${profile}" ] || : > "${profile}"

  if profile_has_path_entry "${profile}" "${path_expr}"; then
    printf '%s\n' "unchanged:${profile}"
    return 0
  fi

  case "${shell_kind}" in
    fish)
      cat >> "${profile}" <<EOF

# >>> rocm-cli path >>>
if not contains -- "${escaped_path_expr}" \$PATH
    set -gx PATH "${escaped_path_expr}" \$PATH
end
# <<< rocm-cli path <<<
EOF
      ;;
    *)
      cat >> "${profile}" <<EOF

# >>> rocm-cli path >>>
case ":\$PATH:" in
  *:"${escaped_path_expr}":*) ;;
  *) export PATH="${escaped_path_expr}:\$PATH" ;;
esac
# <<< rocm-cli path <<<
EOF
      ;;
  esac

  printf '%s\n' "updated:${profile}"
}

ensure_installer_process_path() {
  case ":${PATH}:" in
    *:"${INSTALL_DIR}":*)
      return 0
      ;;
    *)
      export PATH="${INSTALL_DIR}:${PATH}"
      return 0
      ;;
  esac
}

os="$(uname -s)"
arch="$(uname -m)"

case "${os}" in
  Linux) platform_os="linux" ;;
  *)
    fail "unsupported OS: ${os} (installer currently supports Linux x86_64 only)"
    ;;
esac

case "${arch}" in
  x86_64|amd64) platform_arch="amd64" ;;
  *)
    fail "unsupported architecture: ${arch} (installer currently supports Linux x86_64 only)"
    ;;
esac

case "${CHANNEL}" in
  nightly)
    asset_base="rocm-cli-nightly-${platform_os}-${platform_arch}.tar.gz"
    release_path="releases/download/nightly"
    ;;
  release)
    asset_base="rocm-cli-${platform_os}-${platform_arch}.tar.gz"
    release_path="releases/latest/download"
    ;;
  *)
    asset_base="rocm-cli-${platform_os}-${platform_arch}.tar.gz"
    release_path="releases/download/${CHANNEL}"
    ;;
esac

download_base="${ROCM_CLI_DOWNLOAD_BASE:-https://github.com/${REPO}/${release_path}}"
archive_url="${download_base}/${asset_base}"
sha_url="${archive_url}.sha256"
sig_url="${archive_url}.sig"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT INT TERM

manifest_path="${INSTALL_DIR}/.rocm-cli-manifest"

archive_path="${tmp_dir}/${asset_base}"
sha_path="${archive_path}.sha256"
sig_path="${archive_path}.sig"

echo "rocm-cli installer"
echo "  repo: ${REPO}"
echo "  channel: ${CHANNEL}"
echo "  install_dir: ${INSTALL_DIR}"
echo "  download: ${archive_url}"

fetch "${archive_url}" "${archive_path}"
fetch "${sha_url}" "${sha_path}"

expected="$(awk '{print $1}' "${sha_path}" | head -n1)"
[ -n "${expected}" ] || fail "checksum file did not contain a sha256 digest"
actual="$(sha256_file "${archive_path}")"
[ "${expected}" = "${actual}" ] || fail "checksum verification failed"

public_key_path="$(signing_public_key_path)"
if truthy "${ROCM_CLI_REQUIRE_SIGNATURE:-0}" || [ -n "${public_key_path}" ]; then
  [ -n "${public_key_path}" ] || fail "signature verification requires ROCM_CLI_SIGNING_PUBLIC_KEY_PATH or ROCM_CLI_SIGNING_PUBLIC_KEY_PEM"
  fetch "${sig_url}" "${sig_path}" "required signature sidecar is missing or unavailable: ${sig_url}"
  verify_signature "${archive_path}" "${sig_path}" "${public_key_path}"
  echo "signature verified"
fi

extract_dir="${tmp_dir}/extract"
mkdir -p "${extract_dir}"
tar -xzf "${archive_path}" -C "${extract_dir}"

bundle_dir="$(find "${extract_dir}" -mindepth 1 -maxdepth 1 -type d | head -n1)"
[ -n "${bundle_dir}" ] || fail "unable to locate extracted bundle directory"

[ -f "${bundle_dir}/bin/rocm" ] || fail "bundle did not contain bin/rocm"
[ -f "${bundle_dir}/bin/rocmd" ] || fail "bundle did not contain bin/rocmd"
[ -f "${bundle_dir}/bin/rocm-engine-pytorch" ] || fail "bundle did not contain bin/rocm-engine-pytorch"
[ -f "${bundle_dir}/bin/rocm-engine-llama-cpp" ] || fail "bundle did not contain bin/rocm-engine-llama-cpp"
[ -f "${bundle_dir}/bin/rocm-engine-lemonade" ] || fail "bundle did not contain bin/rocm-engine-lemonade"
[ -f "${bundle_dir}/bin/rocm-engine-atom" ] || fail "bundle did not contain bin/rocm-engine-atom"
[ -f "${bundle_dir}/bin/rocm-engine-vllm" ] || fail "bundle did not contain bin/rocm-engine-vllm"
[ -f "${bundle_dir}/bin/rocm-engine-sglang" ] || fail "bundle did not contain bin/rocm-engine-sglang"

mkdir -p "${INSTALL_DIR}"
write_minimal_config_if_missing

if [ -f "${manifest_path}" ]; then
  echo "removing previous rocm-cli install"
  while IFS= read -r installed_path; do
    [ -n "${installed_path}" ] || continue
    case "${installed_path}" in
      "${INSTALL_DIR}"/*)
        rm -f "${installed_path}"
        ;;
      *)
        echo "warning: skipping manifest entry outside install dir: ${installed_path}" >&2
        ;;
    esac
  done < "${manifest_path}"
  rm -f "${manifest_path}"
fi

manifest_tmp="${tmp_dir}/install-manifest"
: > "${manifest_tmp}"
for bin_path in "${bundle_dir}"/bin/*; do
  [ -f "${bin_path}" ] || continue
  bin_name="${bin_path##*/}"
  rm -f "${INSTALL_DIR}/${bin_name}"
  install -m 0755 "${bin_path}" "${INSTALL_DIR}/${bin_name}"
  echo "${INSTALL_DIR}/${bin_name}" >> "${manifest_tmp}"
done
install -m 0644 "${manifest_tmp}" "${manifest_path}"

echo "installed:"
while IFS= read -r installed_path; do
  [ -n "${installed_path}" ] || continue
  echo "  ${installed_path}"
done < "${manifest_path}"

ensure_installer_process_path

case ":${PATH}:" in
  *:"${INSTALL_DIR}":*)
    if [ "${UPDATE_SHELL_PATH}" = "1" ]; then
      profile_path="$(profile_path_for_shell)"
      path_expr="$(path_expr_for_profile "${INSTALL_DIR}")"
      profile_result="$(append_path_snippet "${profile_path}" "$(shell_name)" "${path_expr}")" || true
      case "${profile_result}" in
        updated:*)
          echo "shell profile updated:"
          echo "  profile: ${profile_result#updated:}"
          echo "  new terminals can run: rocm"
          ;;
        unchanged:*)
          echo "shell profile already configured:"
          echo "  profile: ${profile_result#unchanged:}"
          ;;
        *)
          echo "note: ${INSTALL_DIR} is not saved in your shell profile"
          echo "  rocm is installed here: ${INSTALL_DIR}/rocm"
          ;;
      esac
    else
      echo "shell profile update skipped"
      echo "  rocm is installed here: ${INSTALL_DIR}/rocm"
    fi
    ;;
  *)
    echo "note: rocm is installed but this shell could not update PATH"
    echo "  run: ${INSTALL_DIR}/rocm doctor"
    ;;
esac

echo "next:"
if [ "${UPDATE_SHELL_PATH}" = "1" ]; then
  echo "  open a new terminal, then run: rocm doctor"
else
  echo "  ${INSTALL_DIR}/rocm doctor"
fi
