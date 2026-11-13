#!/bin/sh
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set -eu

REPO="${ROCM_CLI_GITHUB_REPO:-ROCm/rocm-cli}"
CHANNEL="${1:-release}"
INSTALL_DIR="${ROCM_CLI_INSTALL_DIR:-${HOME}/.local/bin}"
UPDATE_SHELL_PATH="${ROCM_CLI_UPDATE_SHELL_PATH:-1}"

fail() {
  printf '%s\n' "${C_ERR}!!${C_RESET} rocm-cli installer: $*" >&2
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

use_color=0
# Both stdout and stderr must be TTYs: step()/ok() only ever color stdout
# and fail() only stderr, and gating on stdout alone would leak raw ANSI
# escapes into a redirected stderr (e.g. `sh install.sh 2>error.log`, stdout
# still attached to a terminal) — dropping color entirely is simpler and
# safer than tracking which fd each helper writes to.
if [ -t 1 ] && [ -t 2 ] && [ -z "${NO_COLOR:-}" ]; then
  use_color=1
fi
if [ "${use_color}" -eq 1 ]; then
  C_STEP=$(printf '\033[1;36m')
  C_OK=$(printf '\033[32m')
  C_ERR=$(printf '\033[31m')
  C_RESET=$(printf '\033[0m')
else
  C_STEP=""
  C_OK=""
  C_ERR=""
  C_RESET=""
fi

step() {
  printf '%s\n' "${C_STEP}==>${C_RESET} $*"
}

info() {
  printf '%s\n' "  $*"
}

ok() {
  printf '%s\n' "${C_OK}::${C_RESET} $*"
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

# Pinned production release signing public keys (trust roots). These stay empty
# until the repository owner publishes production keys (see docs/release-trust.md,
# "Remaining Owner Step"). While empty, release installs keep the opt-in behavior
# below: a signature is verified only when a key is supplied via the env vars or
# ROCM_CLI_REQUIRE_SIGNATURE=1. Once populated, release-channel installs verify
# signatures by default with these keys as trust roots. Two slots support
# zero-downtime key rotation: a release signed with either the current or the
# pre-staged next key verifies, so the next key is trusted before its first use.
PINNED_RELEASE_PUBLIC_KEY_CURRENT="-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxuyScR/BzV+kuXqWAHtE
+9xiPCWURUYnsio9MOrf2Xe01mBngP7qPcF13+5nrfT3EnuxOn5rSCYwjOndlS+c
KzOw6GZXJD/ZqeojnbXxxsxlftAQHHEke1WCtga5ZEFxOauTeB5nTV/IbjMAl2Xc
M4PaudpFFH/6j/E3gongDmt0hWdpMLbaCcd3i1vMTEsaHZooNoAbJ/dIAHR/dDNM
pScZAZoy0LL3Afhn5Hiv71trfbfnnboVSdhnCoMmisl6/sK55zR7VM8hWDTTowl3
ultUtiz4emTfXDCb2RptOgoydBA+mu9z6O4eVF8S5dVr/S834SK6dD2fWHNnT0dc
JwIDAQAB
-----END PUBLIC KEY-----"
PINNED_RELEASE_PUBLIC_KEY_NEXT=""
# Keep each slot to the single assignment above: `cargo xtask verify-pinned-keys`
# compares exactly that assignment with docs/keys/ and requires this readonly
# line, which makes any later write fail instead of silently replacing the
# trust root.
readonly PINNED_RELEASE_PUBLIC_KEY_CURRENT PINNED_RELEASE_PUBLIC_KEY_NEXT

has_pinned_release_keys() {
  [ -n "${PINNED_RELEASE_PUBLIC_KEY_CURRENT}" ] || [ -n "${PINNED_RELEASE_PUBLIC_KEY_NEXT}" ]
}

# Emit the candidate signing public keys as newline-separated file paths. An
# explicit env-provided key wins (an escape hatch for private mirrors); otherwise
# the pinned production trust roots are used. Requires ${tmp_dir} to exist.
resolve_public_keys() {
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

  pinned_index=0
  for pinned_pem in "${PINNED_RELEASE_PUBLIC_KEY_CURRENT}" "${PINNED_RELEASE_PUBLIC_KEY_NEXT}"; do
    [ -n "${pinned_pem}" ] || continue
    pinned_index=$((pinned_index + 1))
    key_path="${tmp_dir}/rocm-cli-pinned-key-${pinned_index}.pem"
    printf '%s\n' "${pinned_pem}" > "${key_path}"
    printf '%s\n' "${key_path}"
  done
}

# Verify ${signature} over ${archive} against any of the newline-separated public
# key file paths in ${keys}; succeed on the first match, fail if none verify.
verify_signature() {
  archive="$1"
  signature="$2"
  keys="$3"
  need_cmd openssl
  saved_ifs="${IFS}"
  IFS='
'
  for key in ${keys}; do
    IFS="${saved_ifs}"
    [ -n "${key}" ] || continue
    if openssl dgst -sha256 -verify "${key}" -signature "${signature}" "${archive}" >/dev/null 2>&1; then
      return 0
    fi
    IFS='
'
  done
  IFS="${saved_ifs}"
  fail "signature verification failed"
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
    info "config: existing ${config_path}"
    return
  fi

  mkdir -p "${config_dir}"
  config_tmp="${tmp_dir}/config.json"
  # Deliberately no "default_engine": the CLI picks the serving engine from the
  # host GPU (vLLM on Instinct data-center parts, Lemonade elsewhere). Seeding a
  # value here would be treated as an explicit user choice and short-circuit that
  # detection -- which is how every installed Instinct box ended up on Lemonade.
  cat > "${config_tmp}" <<'JSON'
{
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
  info "config: created ${config_path}"
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
    fish) fish_config_path ;;
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

  # A profile that cannot be created leaves nothing for the `cat >>` below to
  # append to, and that write returns non-zero before "updated" is printed, so
  # the caller reports that the folder is not saved instead of claiming it is.
  profile_dir="${profile%/*}"
  if [ "${profile_dir}" != "${profile}" ]; then
    mkdir -p "${profile_dir}" 2>/dev/null || true
  fi
  [ -f "${profile}" ] || { : > "${profile}"; } 2>/dev/null || true

  if profile_has_path_entry "${profile}" "${path_expr}"; then
    printf '%s\n' "unchanged:${profile}"
    return 0
  fi

  case "${shell_kind}" in
    fish)
      cat >> "${profile}" <<EOF || return 1

# >>> rocm-cli path >>>
if not contains -- "${escaped_path_expr}" \$PATH
    set -gx PATH "${escaped_path_expr}" \$PATH
end
# <<< rocm-cli path <<<
EOF
      ;;
    *)
      cat >> "${profile}" <<EOF || return 1

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

# Whether profile $1 holds the installer's marked block for path expression $2,
# spelled the way append_path_snippet writes it, and closed by its end marker,
# as `rocm uninstall` requires before it removes one. A block for another
# folder, or a PATH line outside the markers, does not count.
profile_has_own_block() {
  [ -f "$1" ] || return 1
  escaped="$(escape_for_double_quotes "$2")"
  # Passed through the environment: awk -v would undo the escaping above.
  ROCM_CLI_BLOCK_QUOTED="\"${escaped}\"" ROCM_CLI_BLOCK_LISTED="\"${escaped}:" awk '
    { sub(/[ \t\r]+$/, "") }
    $0 == "# >>> rocm-cli path >>>" { inside = 1; hit = 0; next }
    $0 == "# <<< rocm-cli path <<<" { if (inside && hit) found = 1; inside = 0; next }
    inside && (index($0, ENVIRON["ROCM_CLI_BLOCK_QUOTED"]) || index($0, ENVIRON["ROCM_CLI_BLOCK_LISTED"])) { hit = 1 }
    END { exit found ? 0 : 1 }
  ' "$1"
}

# Add one line to the record of the PATH change this installer made, unless it
# is already there.
record_path_edit() {
  record="${INSTALL_DIR}/.rocm-cli-path-edit"
  if [ -f "${record}" ] && grep -qxF "$1" "${record}"; then
    return 0
  fi
  printf '%s\n' "$1" >> "${record}"
}

# The folder fish reads its config from: $XDG_CONFIG_HOME/fish when that is
# set to an absolute path, else ~/.config/fish. fish would also take a relative
# XDG_CONFIG_HOME, but against its own working directory, so it names a
# different folder from each directory; the installer does not follow that.
fish_config_dir() {
  case "${XDG_CONFIG_HOME:-}" in
    /*) printf '%s\n' "${XDG_CONFIG_HOME}/fish" ;;
    *) printf '%s\n' "${HOME}/.config/fish" ;;
  esac
}

fish_config_path() {
  printf '%s\n' "$(fish_config_dir)/config.fish"
}

# Whether fish is set up on this machine, whatever the login shell is: the
# binary is on PATH, or the user already has a fish config directory. A bash
# login shell is common even for people who run fish interactively.
fish_is_installed() {
  command -v fish >/dev/null 2>&1 && return 0
  [ -d "$(fish_config_dir)" ] && return 0
  return 1
}

# Append the fish PATH block to fish's config.fish and record it so
# `rocm uninstall` can take it out again. The caller skips this when that file
# is the profile the login-shell step just wrote.
configure_fish_path() {
  fish_profile="$(fish_config_path)"
  fish_expr="$1"
  fish_result="$(append_path_snippet "${fish_profile}" fish "${fish_expr}")" || true
  if profile_has_own_block "${fish_profile}" "${fish_expr}"; then
    record_path_edit "profile ${fish_profile}"
    record_path_edit "path-expr ${fish_expr}"
  fi
  case "${fish_result}" in
    updated:*)
      ok "fish shell profile updated:"
      info "profile: ${fish_result#updated:}"
      ;;
    unchanged:*)
      info "fish shell profile already configured:"
      info "profile: ${fish_result#unchanged:}"
      ;;
    *)
      info "note: ${INSTALL_DIR} is not saved in your fish config"
      ;;
  esac
}

# The first line of the fish completion script this installer writes, followed
# by the install folder. `rocm uninstall` removes the script only when it still
# starts with this line naming its own folder, so a script the user wrote, or
# one another install folder wrote over this one, stays. Keep it in step with
# FISH_COMPLETIONS_MARKER in apps/rocm/src/uninstall.rs.
FISH_COMPLETIONS_MARKER='# rocm-cli fish completions for '

# The same for the rocm(1) manual page: a roff comment line (`.\"`), which man
# does not show, followed by the install folder. Keep it in step with
# MAN_PAGE_MARKER in apps/rocm/src/uninstall.rs.
MAN_PAGE_MARKER='.\" rocm-cli manual page for '

# How the first line of the page `rocm __man-page` prints starts. A rocm built
# before that verb existed reads it as a natural-language request: it prints a
# request plan and exits 0, so success and output alone do not mean a page.
# Keep it in step with TITLE_PREFIX in apps/rocm/src/man_page.rs.
MAN_PAGE_TITLE='.TH ROCM 1 '

# $1 quoted for a POSIX shell, so a printed command line can be run as printed
# whatever the path holds: each ' becomes '\'' inside single quotes.
sh_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

# Whether file $1 is a regular file this installer (any install folder) wrote:
# not a link, and starting with marker line $2.
is_installer_file() {
  [ -f "$1" ] && [ ! -L "$1" ] || return 1
  first_line=""
  IFS= read -r first_line < "$1" || [ -n "${first_line}" ] || return 1
  case "${first_line}" in
    "$2"?*) return 0 ;;
    *) return 1 ;;
  esac
}

# Install fish completions from the binary that was just placed, so they match
# it, and record the file so `rocm uninstall` removes it with the binary. Best
# effort: a binary that cannot print them leaves the install intact. The binary
# reads nothing from stdin, which under `curl | sh` is the rest of this script.
#
# rocm.fish is one fixed path, so a file or link that is already there and did
# not come from this installer is the user's: it is left alone, never written
# through, and not recorded, so `rocm uninstall` leaves it too.
install_fish_completions() {
  completions_dir="$(fish_config_dir)/completions"
  completions_path="${completions_dir}/rocm.fish"
  completions_tmp="${tmp_dir}/rocm.fish"
  completions_marked="${tmp_dir}/rocm.fish.marked"
  if { [ -e "${completions_path}" ] || [ -L "${completions_path}" ]; } \
    && ! is_installer_file "${completions_path}" "${FISH_COMPLETIONS_MARKER}"; then
    info "note: fish completions not installed: ${completions_path} already exists and was not written by this installer"
    info "it is left as it is; remove it and run the installer again to replace it"
    return 0
  fi
  if "${INSTALL_DIR}/rocm" completions fish < /dev/null > "${completions_tmp}" \
    && [ -s "${completions_tmp}" ] \
    && { printf '%s%s\n' "${FISH_COMPLETIONS_MARKER}" "${INSTALL_DIR}" \
      && cat "${completions_tmp}"; } > "${completions_marked}" \
    && mkdir -p "${completions_dir}" \
    && install -m 0644 "${completions_marked}" "${completions_path}"; then
    record_path_edit "completions ${completions_path}"
    ok "fish completions installed:"
    info "${completions_path}"
  else
    info "note: could not install fish completions"
    info "to try again, run the installer again, or run:"
    info "mkdir -p $(sh_quote "${completions_dir}")"
    info "$(sh_quote "${INSTALL_DIR}/rocm") completions fish > $(sh_quote "${completions_path}")"
    info "(a file written that way has no installer marker, so rocm uninstall leaves it)"
  fi
}

# Install the rocm(1) manual page from the binary that was just placed, so it
# matches it, and record it so `rocm uninstall` removes it with the binary. It
# goes in share/man/man1 under the folder above the install folder
# (~/.local/share/man/man1 for the default ~/.local/bin), where man looks for
# the pages of any folder on PATH, so it needs no shell profile change and is
# installed whatever ROCM_CLI_UPDATE_SHELL_PATH says. Best effort, and a rocm.1
# that did not come from this installer is left alone, as for the fish
# completions above. Output that is not a page (see MAN_PAGE_TITLE) means a
# rocm too old to have one, which no retry fixes, so it gets a note only.
man_page_installed=0
install_man_page() {
  case "${INSTALL_DIR}" in
    /*) ;;
    *)
      info "note: manual page not installed: the install folder ${INSTALL_DIR} is not an absolute path"
      return 0
      ;;
  esac
  man_dir="$(dirname -- "${INSTALL_DIR}")/share/man/man1"
  man_path="${man_dir}/rocm.1"
  man_tmp="${tmp_dir}/rocm.1"
  man_marked="${tmp_dir}/rocm.1.marked"
  if { [ -e "${man_path}" ] || [ -L "${man_path}" ]; } \
    && ! is_installer_file "${man_path}" "${MAN_PAGE_MARKER}"; then
    info "note: manual page not installed: ${man_path} already exists and was not written by this installer"
    info "it is left as it is; remove it and run the installer again to replace it"
    return 0
  fi
  if ! "${INSTALL_DIR}/rocm" __man-page < /dev/null > "${man_tmp}" \
    || [ ! -s "${man_tmp}" ]; then
    man_page_status=failed
  else
    man_first_line=""
    IFS= read -r man_first_line < "${man_tmp}" || true
    case "${man_first_line}" in
      "${MAN_PAGE_TITLE}"*) man_page_status=page ;;
      *) man_page_status=not-a-page ;;
    esac
  fi
  if [ "${man_page_status}" = not-a-page ]; then
    # Nothing to try again: this rocm has no manual page to print.
    info "note: manual page not installed: this version of rocm does not provide one"
    return 0
  fi
  if [ "${man_page_status}" = page ] \
    && { printf '%s%s\n' "${MAN_PAGE_MARKER}" "${INSTALL_DIR}" \
      && cat "${man_tmp}"; } > "${man_marked}" \
    && mkdir -p "${man_dir}" \
    && install -m 0644 "${man_marked}" "${man_path}"; then
    record_path_edit "man ${man_path}"
    man_page_installed=1
    ok "manual page installed:"
    info "${man_path}"
  else
    # Run as printed, these lines write the page as the installer would,
    # marker included, so the next install or uninstall treats it as its own.
    info "note: could not install the manual page"
    info "to try again, run the installer again, or run:"
    info "mkdir -p $(sh_quote "${man_dir}")"
    info "printf '%s%s\\n' $(sh_quote "${MAN_PAGE_MARKER}") $(sh_quote "${INSTALL_DIR}") > $(sh_quote "${man_path}")"
    info "$(sh_quote "${INSTALL_DIR}/rocm") __man-page >> $(sh_quote "${man_path}")"
    info "(rocm uninstall removes a page written that way only after the installer runs again and records it)"
  fi
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

print_completion_banner() {
  cmd="$1"
  next_hint="$2"
  # The box-drawing art is only for an interactive terminal on a UTF-8 locale;
  # piped, logged, or non-UTF-8 output gets the plain "Get started" block only.
  if [ -t 1 ]; then
    case "${LC_ALL:-${LC_CTYPE:-${LANG:-}}}" in
      *[Uu][Tt][Ff]-8* | *[Uu][Tt][Ff]8*)
        printf '%s' "${C_STEP}"
        cat <<'ROCM_CLI_BANNER'
 ██████╗  ██████╗  ██████╗███╗   ███╗     ██████╗██╗     ██╗
 ██╔══██╗██╔═══██╗██╔════╝████╗ ████║    ██╔════╝██║     ██║
 ██████╔╝██║   ██║██║     ██╔████╔██║    ██║     ██║     ██║
 ██╔══██╗██║   ██║██║     ██║╚██╔╝██║    ██║     ██║     ██║
 ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║    ╚██████╗███████╗██║
 ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝     ╚═════╝╚══════╝╚═╝

        Local AI on AMD GPUs — one binary, zero setup
ROCM_CLI_BANNER
        printf '%s\n' "${C_RESET}"
        ;;
      *) ;;
    esac
  fi
  echo "Get started:"
  printf '  %-20s # %s\n' "${cmd} install sdk" "set up a managed ROCm runtime"
  printf '  %-20s # %s\n' "${cmd} serve qwen" "serve a small built-in assistant model"
  printf '  %-20s # %s\n' "${cmd}" "open the launcher"
  # man finds the page through the install folder on PATH, which only the
  # shell profile change puts there in a new terminal, so the hint needs that
  # change to have been saved, not only attempted.
  if [ "${man_page_installed}" -eq 1 ] && [ "${profile_saved}" -eq 1 ]; then
    printf '  %-20s # %s\n' "man rocm" "read the manual, offline"
  fi
  echo
  echo "${next_hint}"
}

# Which machine the artifact is *for*. Normally this machine, but the target can
# be named explicitly so an artifact can be fetched here on behalf of a different
# one -- see ROCM_CLI_DOWNLOAD_ONLY below. Detection is the only thing that
# changes; naming a target does not relax any verification.
os="${ROCM_CLI_TARGET_OS:-$(uname -s)}"
arch="${ROCM_CLI_TARGET_ARCH:-$(uname -m)}"

case "${os}" in
  Linux|linux) platform_os="linux" ;;
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

# Fetch and verify an artifact, then stop without installing it. Used to obtain a
# build for a machine that cannot reach the release host itself: this machine
# downloads it, verifies it, and hands the verified files over for delivery.
#
# The artifact is emitted with its checksum and signature sidecars so that
# whatever installs it later can repeat every check performed here. Nothing is
# skipped on either side -- this splits the trust chain across two machines, it
# does not shorten it.
DOWNLOAD_ONLY=0
if truthy "${ROCM_CLI_DOWNLOAD_ONLY:-0}"; then
  DOWNLOAD_ONLY=1
fi
DOWNLOAD_DIR="${ROCM_CLI_DOWNLOAD_DIR:-.}"

# Install from an artifact already on disk instead of downloading one. The
# receiving half of the split above. Its .sha256 sidecar is required, not
# optional: an artifact that arrived over the network is exactly the one whose
# integrity still has to be proven.
LOCAL_ARCHIVE="${ROCM_CLI_ARCHIVE:-}"
if [ -n "${LOCAL_ARCHIVE}" ] && [ "${DOWNLOAD_ONLY}" -eq 1 ]; then
  fail "ROCM_CLI_ARCHIVE and ROCM_CLI_DOWNLOAD_ONLY are mutually exclusive"
fi

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

step "rocm-cli installer"
info "repo: ${REPO}"
info "channel: ${CHANNEL}"
info "platform: ${platform_os}-${platform_arch}"
if [ -n "${LOCAL_ARCHIVE}" ]; then
  info "archive: ${LOCAL_ARCHIVE}"
elif [ "${DOWNLOAD_ONLY}" -eq 1 ]; then
  info "download_only: ${DOWNLOAD_DIR}"
  info "download: ${archive_url}"
else
  info "install_dir: ${INSTALL_DIR}"
  info "download: ${archive_url}"
fi

if [ -n "${LOCAL_ARCHIVE}" ]; then
  [ -f "${LOCAL_ARCHIVE}" ] || fail "archive not found: ${LOCAL_ARCHIVE}"
  [ -f "${LOCAL_ARCHIVE}.sha256" ] || fail "archive checksum not found: ${LOCAL_ARCHIVE}.sha256"
  cp "${LOCAL_ARCHIVE}" "${archive_path}"
  cp "${LOCAL_ARCHIVE}.sha256" "${sha_path}"
  if [ -f "${LOCAL_ARCHIVE}.sig" ]; then
    cp "${LOCAL_ARCHIVE}.sig" "${sig_path}"
  fi
else
  fetch "${archive_url}" "${archive_path}"
  fetch "${sha_url}" "${sha_path}"
fi

expected="$(awk '{print $1}' "${sha_path}" | head -n1)"
[ -n "${expected}" ] || fail "checksum file did not contain a sha256 digest"
actual="$(sha256_file "${archive_path}")"
[ "${expected}" = "${actual}" ] || fail "checksum verification failed"

public_keys="$(resolve_public_keys)"

require_sig=0
if truthy "${ROCM_CLI_REQUIRE_SIGNATURE:-0}"; then
  require_sig=1
fi
# Production default: once release trust roots are pinned, release-channel
# installs always verify. An unset or 0 ROCM_CLI_REQUIRE_SIGNATURE does not lower
# this floor; use ROCM_CLI_SIGNING_PUBLIC_KEY_PATH/PEM to point at an alternate
# key (e.g. a private mirror) instead.
if [ "${CHANNEL}" = "release" ] && has_pinned_release_keys; then
  require_sig=1
fi

if [ "${require_sig}" -eq 1 ] || [ -n "${public_keys}" ]; then
  [ -n "${public_keys}" ] || fail "signature verification requires ROCM_CLI_SIGNING_PUBLIC_KEY_PATH or ROCM_CLI_SIGNING_PUBLIC_KEY_PEM"
  if [ -n "${LOCAL_ARCHIVE}" ]; then
    # A locally-supplied archive has no URL to fall back on: the signature had to
    # travel with it. Refusing here is the point -- an artifact delivered out of
    # band is precisely the one whose provenance cannot be assumed.
    [ -f "${sig_path}" ] || fail "required signature sidecar is missing: ${LOCAL_ARCHIVE}.sig"
  else
    fetch "${sig_url}" "${sig_path}" "required signature sidecar is missing or unavailable: ${sig_url}"
  fi
  verify_signature "${archive_path}" "${sig_path}" "${public_keys}"
  ok "signature verified"
fi

# Everything above ran unchanged. Only now, with the artifact proven, is it
# handed over rather than installed.
if [ "${DOWNLOAD_ONLY}" -eq 1 ]; then
  mkdir -p "${DOWNLOAD_DIR}"
  install -m 0644 "${archive_path}" "${DOWNLOAD_DIR}/${asset_base}"
  install -m 0644 "${sha_path}" "${DOWNLOAD_DIR}/${asset_base}.sha256"
  if [ -f "${sig_path}" ]; then
    install -m 0644 "${sig_path}" "${DOWNLOAD_DIR}/${asset_base}.sig"
  fi
  echo "downloaded: ${DOWNLOAD_DIR}/${asset_base}"
  exit 0
fi

extract_dir="${tmp_dir}/extract"
mkdir -p "${extract_dir}"
tar -xzf "${archive_path}" -C "${extract_dir}"

bundle_dir="$(find "${extract_dir}" -mindepth 1 -maxdepth 1 -type d | head -n1)"
[ -n "${bundle_dir}" ] || fail "unable to locate extracted bundle directory"

# First-party engines are built into rocm and run in-process; the standalone
# rocm-engine-* binaries are an external plugin fallback, not shipped here.
[ -f "${bundle_dir}/bin/rocm" ] || fail "bundle did not contain bin/rocm"
[ -f "${bundle_dir}/bin/rocmd" ] || fail "bundle did not contain bin/rocmd"

mkdir -p "${INSTALL_DIR}"
write_minimal_config_if_missing

if [ -f "${manifest_path}" ]; then
  step "removing previous rocm-cli install"
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

step "installed:"
while IFS= read -r installed_path; do
  [ -n "${installed_path}" ] || continue
  info "${installed_path}"
done < "${manifest_path}"

ensure_installer_process_path

# Whether the shell profile now puts the install folder on PATH in a new
# terminal, which is how man finds the manual page.
profile_saved=0
case ":${PATH}:" in
  *:"${INSTALL_DIR}":*)
    if [ "${UPDATE_SHELL_PATH}" = "1" ]; then
      profile_path="$(profile_path_for_shell)"
      path_expr="$(path_expr_for_profile "${INSTALL_DIR}")"
      profile_result="$(append_path_snippet "${profile_path}" "$(shell_name)" "${path_expr}")" || true
      # Record where the marked block lives, and how it spells this folder, so
      # `rocm uninstall` can take it out again. Only a marked block for this
      # folder is recorded: a PATH line the user wrote themselves, or a block
      # another install wrote for a different folder, is not this install's.
      # Lines are added, not replaced, so a block an earlier install wrote from
      # another shell is still on record.
      if profile_has_own_block "${profile_path}" "${path_expr}"; then
        record_path_edit "profile ${profile_path}"
        record_path_edit "path-expr ${path_expr}"
      fi
      case "${profile_result}" in
        updated:* | unchanged:*) profile_saved=1 ;;
        *) ;;
      esac
      case "${profile_result}" in
        updated:*)
          ok "shell profile updated:"
          info "profile: ${profile_result#updated:}"
          info "new terminals can run: rocm"
          ;;
        unchanged:*)
          info "shell profile already configured:"
          info "profile: ${profile_result#unchanged:}"
          ;;
        *)
          info "note: ${INSTALL_DIR} is not saved in your shell profile"
          info "rocm is installed here: ${INSTALL_DIR}/rocm"
          ;;
      esac
      # Fish gets its own snippet when it is installed, including when the
      # login shell is bash, zsh, or ksh. Skip the second write when that
      # login shell already targeted config.fish.
      if fish_is_installed; then
        if [ "${profile_path}" != "$(fish_config_path)" ]; then
          configure_fish_path "${path_expr}"
        fi
        install_fish_completions
      fi
    else
      info "shell profile update skipped"
      info "rocm is installed here: ${INSTALL_DIR}/rocm"
    fi
    ;;
  *)
    info "note: rocm is installed but this shell could not update PATH"
    info "run: ${INSTALL_DIR}/rocm examine"
    ;;
esac

install_man_page

if [ "${UPDATE_SHELL_PATH}" = "1" ]; then
  cmd="rocm"
  next_hint="Open a new terminal, then run: rocm examine"
else
  cmd="${INSTALL_DIR}/rocm"
  next_hint="Run: ${cmd} examine"
fi
print_completion_banner "${cmd}" "${next_hint}"
