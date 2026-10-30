#!/bin/sh
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Drive install.sh with a locally signed bundle and a stub `rocm`.
#
# The case that matters: a bash (or zsh) login shell on a machine where fish
# is installed must get the fish PATH block, fish completions, and an uninstall
# record for that profile. A bash login shell with no fish install must not
# grow a fish config. Re-running must not append a second PATH block.
#
# Usage: tests/install-sh/run.sh
# Set INSTALL_SH to test another copy of the installer (mutation checks).
# Requires: openssl, tar, sha256sum, and a POSIX sh. No network and no GPU.

set -eu

ROOT="$(CDPATH='' cd -- "$(dirname "$0")/../.." && pwd)"
INSTALLER="${INSTALL_SH:-${ROOT}/install.sh}"
# The installer executes the binary it just wrote. /tmp is often mounted
# noexec, so the scratch directory lives on this repo's filesystem.
mkdir -p "${ROOT}/.rocm-work"
WORK="$(mktemp -d "${ROOT}/.rocm-work/install-sh.XXXXXX")"
FAILURES=0

cleanup() {
  rm -rf "${WORK}"
}
trap cleanup EXIT INT TERM

fail() {
  echo "FAIL: $*" >&2
  FAILURES=$((FAILURES + 1))
}

pass() {
  echo "ok: $*"
}

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

for cmd in openssl tar gzip sha256sum awk grep sed find mktemp install; do
  need "${cmd}"
done

TOOLS="${WORK}/tools"
FISH_BIN="${WORK}/fish-bin"
mkdir -p "${TOOLS}" "${FISH_BIN}" "${WORK}/tmp"

link_tool() {
  target="$(command -v "$1")" || {
    echo "missing required command: $1" >&2
    exit 1
  }
  ln -s "${target}" "${TOOLS}/$1"
}

for cmd in sh cat awk sed grep rm mkdir mktemp tar gzip openssl sha256sum find head cp uname install dirname; do
  link_tool "${cmd}"
done

# A name on PATH is enough: the installer only tests `command -v fish`.
cat > "${FISH_BIN}/fish" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod 755 "${FISH_BIN}/fish"

BUNDLE="${WORK}/bundle/rocm-cli-linux-amd64/bin"
mkdir -p "${BUNDLE}"
cat > "${BUNDLE}/rocm" <<'EOF'
#!/bin/sh
if [ "${1:-}" = "completions" ] && [ "${2:-}" = "fish" ]; then
  printf '%s\n' '# rocm-cli fish completions fixture'
  exit 0
fi
printf '%s\n' "unexpected rocm invocation: $*" >&2
exit 1
EOF
cat > "${BUNDLE}/rocmd" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod 755 "${BUNDLE}/rocm" "${BUNDLE}/rocmd"

ARCHIVE="${WORK}/rocm-cli-linux-amd64.tar.gz"
tar -C "${WORK}/bundle" -czf "${ARCHIVE}" rocm-cli-linux-amd64
sha256sum "${ARCHIVE}" | awk '{print $1}' > "${ARCHIVE}.sha256"
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "${WORK}/private.pem" 2>/dev/null
openssl pkey -in "${WORK}/private.pem" -pubout -out "${WORK}/public.pem"
openssl dgst -sha256 -sign "${WORK}/private.pem" -out "${ARCHIVE}.sig" "${ARCHIVE}"

BASE_PATH="${TOOLS}"
FISH_PATH="${FISH_BIN}:${TOOLS}"
MARKER='# >>> rocm-cli path >>>'
# Literal installer text, including the unexpanded $HOME and $PATH.
# shellcheck disable=SC2016
FISH_SNIPPET='if not contains -- "$HOME/.local/bin" $PATH'
# shellcheck disable=SC2016
POSIX_SNIPPET='case ":$PATH:" in'
COMPLETIONS_FIXTURE='# rocm-cli fish completions fixture'

install_failed() {
  fail "installer exited non-zero"
  echo "---- installer log ----" >&2
  cat "$1" >&2
}

# install_case <log> <home> <shell> <path> <update-shell-path> [profile]
install_case() {
  log="$1"
  home="$2"
  shell="$3"
  path="$4"
  update="$5"
  profile="${6:-}"
  mkdir -p "${home}"
  if [ -n "${profile}" ]; then
    env -i \
      HOME="${home}" \
      SHELL="${shell}" \
      PATH="${path}" \
      TMPDIR="${WORK}/tmp" \
      ROCM_CLI_INSTALL_DIR="${home}/.local/bin" \
      ROCM_CLI_ARCHIVE="${ARCHIVE}" \
      ROCM_CLI_SIGNING_PUBLIC_KEY_PATH="${WORK}/public.pem" \
      ROCM_CLI_UPDATE_SHELL_PATH="${update}" \
      ROCM_CLI_SHELL_PROFILE="${profile}" \
      "${TOOLS}/sh" "${INSTALLER}" release > "${log}" 2>&1 || install_failed "${log}"
  else
    env -i \
      HOME="${home}" \
      SHELL="${shell}" \
      PATH="${path}" \
      TMPDIR="${WORK}/tmp" \
      ROCM_CLI_INSTALL_DIR="${home}/.local/bin" \
      ROCM_CLI_ARCHIVE="${ARCHIVE}" \
      ROCM_CLI_SIGNING_PUBLIC_KEY_PATH="${WORK}/public.pem" \
      ROCM_CLI_UPDATE_SHELL_PATH="${update}" \
      "${TOOLS}/sh" "${INSTALLER}" release > "${log}" 2>&1 || install_failed "${log}"
  fi
}

marker_count() {
  if [ -f "$1" ]; then
    grep -F -c "${MARKER}" "$1" || true
  else
    printf '%s\n' 0
  fi
}

assert_contains() {
  file="$1"
  needle="$2"
  label="$3"
  if [ ! -f "${file}" ] || ! grep -F -q -- "${needle}" "${file}"; then
    fail "${label}"
    if [ -f "${file}" ]; then
      echo "---- ${file} ----" >&2
      cat "${file}" >&2
    else
      echo "---- ${file} (missing) ----" >&2
    fi
  fi
}

assert_absent_file() {
  if [ -e "$1" ]; then
    fail "$2"
    echo "---- $1 ----" >&2
    cat "$1" >&2 || true
  fi
}

assert_eq() {
  if [ "$2" != "$3" ]; then
    fail "$1: expected '$3', got '$2'"
  fi
}

assert_file_eq() {
  label="$1"
  file="$2"
  expected="$3"
  if [ ! -f "${file}" ]; then
    fail "${label}: missing ${file}"
    return
  fi
  actual="$(cat "${file}")"
  assert_eq "${label}" "${actual}" "${expected}"
}

assert_log_contains() {
  if ! grep -F -q -- "$2" "$1"; then
    fail "$3"
    echo "---- log ----" >&2
    cat "$1" >&2
  fi
}

# Bash login shell, fish on PATH: both profiles, completions, and the record.
bash_home="${WORK}/homes/bash-fish"
install_case "${WORK}/bash-fish.log" "${bash_home}" /bin/bash "${FISH_PATH}" 1
bash_profile="${bash_home}/.bashrc"
fish_profile="${bash_home}/.config/fish/config.fish"
fish_completions="${bash_home}/.config/fish/completions/rocm.fish"
record="${bash_home}/.local/bin/.rocm-cli-path-edit"
assert_contains "${bash_profile}" "${POSIX_SNIPPET}" "bash profile missing the POSIX PATH block"
if grep -F -q -- "${FISH_SNIPPET}" "${bash_profile}"; then
  fail "bash profile received the fish PATH snippet"
fi
assert_contains "${fish_profile}" "${FISH_SNIPPET}" "fish config missing the fish PATH snippet when SHELL is bash"
if grep -F -q -- "${POSIX_SNIPPET}" "${fish_profile}"; then
  fail "fish config received the POSIX PATH snippet"
fi
assert_eq "bash marker count" "$(marker_count "${bash_profile}")" 1
assert_eq "fish marker count" "$(marker_count "${fish_profile}")" 1
assert_contains "${record}" "profile ${bash_profile}" "uninstall record missing the bash profile"
assert_contains "${record}" "profile ${fish_profile}" "uninstall record missing the fish profile"
# shellcheck disable=SC2016
assert_contains "${record}" 'path-expr $HOME/.local/bin' "uninstall record missing the path expression"
assert_eq "completions file" "$(cat "${fish_completions}")" "${COMPLETIONS_FIXTURE}"
assert_log_contains "${WORK}/bash-fish.log" "shell profile updated:" "installer did not report a shell profile update"
assert_log_contains "${WORK}/bash-fish.log" "profile: ${fish_profile}" "installer did not report the fish profile"
assert_log_contains "${WORK}/bash-fish.log" "fish completions installed:" "installer did not report fish completions"
assert_log_contains "${WORK}/bash-fish.log" "${fish_completions}" "installer did not report the completions path"
if [ "${FAILURES}" -eq 0 ]; then
  pass "bash login shell also configures fish"
fi

# A second install must not append another marked block or another record line.
install_case "${WORK}/bash-fish-again.log" "${bash_home}" /bin/bash "${FISH_PATH}" 1
assert_eq "bash marker count after reinstall" "$(marker_count "${bash_profile}")" 1
assert_eq "fish marker count after reinstall" "$(marker_count "${fish_profile}")" 1
assert_eq "recorded bash profile lines" "$(grep -F -c "profile ${bash_profile}" "${record}" || true)" 1
assert_eq "recorded fish profile lines" "$(grep -F -c "profile ${fish_profile}" "${record}" || true)" 1
assert_eq "completions file after reinstall" "$(cat "${fish_completions}")" "${COMPLETIONS_FIXTURE}"
assert_log_contains "${WORK}/bash-fish-again.log" "shell profile already configured:" "reinstall did not report the existing profile"
if [ "${FAILURES}" -eq 0 ]; then
  pass "reinstall does not duplicate the fish block"
fi

# Bash, no fish binary and no fish config directory.
plain_home="${WORK}/homes/bash-plain"
install_case "${WORK}/bash-plain.log" "${plain_home}" /bin/bash "${BASE_PATH}" 1
assert_contains "${plain_home}/.bashrc" "${POSIX_SNIPPET}" "bash profile missing when fish is absent"
assert_absent_file "${plain_home}/.config/fish/config.fish" "bash user with no fish install grew a fish config"
assert_absent_file "${plain_home}/.config/fish/completions/rocm.fish" "bash user with no fish install grew fish completions"
if grep -F -q -- "config.fish" "${plain_home}/.local/bin/.rocm-cli-path-edit"; then
  fail "uninstall record names a fish profile that was not written"
fi
if [ "${FAILURES}" -eq 0 ]; then
  pass "bash without fish does not grow a fish config"
fi

# Fish config directory exists, but fish is not on PATH.
dir_home="${WORK}/homes/fish-dir"
mkdir -p "${dir_home}/.config/fish"
printf '%s\n' 'set -gx EDITOR vim' > "${dir_home}/.config/fish/config.fish"
install_case "${WORK}/fish-dir.log" "${dir_home}" /bin/bash "${BASE_PATH}" 1
assert_contains "${dir_home}/.config/fish/config.fish" "set -gx EDITOR vim" "existing fish config lost its own line"
assert_contains "${dir_home}/.config/fish/config.fish" "${FISH_SNIPPET}" "existing fish config was not given the PATH snippet"
assert_eq "fish marker count for an existing config dir" "$(marker_count "${dir_home}/.config/fish/config.fish")" 1
assert_contains "${dir_home}/.local/bin/.rocm-cli-path-edit" "profile ${dir_home}/.config/fish/config.fish" "uninstall record missing fish when only the config dir exists"
assert_eq "completions from a config dir" "$(cat "${dir_home}/.config/fish/completions/rocm.fish")" "${COMPLETIONS_FIXTURE}"
if [ "${FAILURES}" -eq 0 ]; then
  pass "an existing fish config dir is configured when fish is not on PATH"
fi

# Login shell is fish. The fish profile is the one the login-shell path writes,
# and completions are still installed. Bash is left alone.
fish_home="${WORK}/homes/fish-login"
install_case "${WORK}/fish-login.log" "${fish_home}" "${FISH_BIN}/fish" "${FISH_PATH}" 1
assert_contains "${fish_home}/.config/fish/config.fish" "${FISH_SNIPPET}" "fish login shell did not get the fish snippet"
assert_eq "fish login marker count" "$(marker_count "${fish_home}/.config/fish/config.fish")" 1
assert_absent_file "${fish_home}/.bashrc" "fish login shell also wrote a bash profile"
assert_eq "fish login completions" "$(cat "${fish_home}/.config/fish/completions/rocm.fish")" "${COMPLETIONS_FIXTURE}"
assert_contains "${fish_home}/.local/bin/.rocm-cli-path-edit" "profile ${fish_home}/.config/fish/config.fish" "fish login shell was not recorded"
if [ "${FAILURES}" -eq 0 ]; then
  pass "fish login shell still configures fish once"
fi

# zsh keeps the POSIX snippet, and fish is configured beside it.
zsh_home="${WORK}/homes/zsh-fish"
install_case "${WORK}/zsh-fish.log" "${zsh_home}" /bin/zsh "${FISH_PATH}" 1
assert_contains "${zsh_home}/.zshrc" "${POSIX_SNIPPET}" "zsh profile missing the POSIX PATH block"
assert_contains "${zsh_home}/.config/fish/config.fish" "${FISH_SNIPPET}" "zsh login shell skipped fish"
assert_absent_file "${zsh_home}/.bashrc" "zsh login shell also wrote a bash profile"
assert_contains "${zsh_home}/.local/bin/.rocm-cli-path-edit" "profile ${zsh_home}/.zshrc" "uninstall record missing the zsh profile"
assert_contains "${zsh_home}/.local/bin/.rocm-cli-path-edit" "profile ${zsh_home}/.config/fish/config.fish" "uninstall record missing fish beside zsh"
if [ "${FAILURES}" -eq 0 ]; then
  pass "zsh login shell keeps its profile and also configures fish"
fi

# ksh keeps the POSIX snippet in .kshrc, and fish is configured beside it.
ksh_home="${WORK}/homes/ksh-fish"
install_case "${WORK}/ksh-fish.log" "${ksh_home}" /bin/ksh "${FISH_PATH}" 1
assert_contains "${ksh_home}/.kshrc" "${POSIX_SNIPPET}" "ksh profile missing the POSIX PATH block"
assert_contains "${ksh_home}/.config/fish/config.fish" "${FISH_SNIPPET}" "ksh login shell skipped fish"
assert_absent_file "${ksh_home}/.bashrc" "ksh login shell also wrote a bash profile"
assert_absent_file "${ksh_home}/.profile" "ksh login shell also wrote a generic profile"
assert_contains "${ksh_home}/.local/bin/.rocm-cli-path-edit" "profile ${ksh_home}/.kshrc" "uninstall record missing the ksh profile"
assert_contains "${ksh_home}/.local/bin/.rocm-cli-path-edit" "profile ${ksh_home}/.config/fish/config.fish" "uninstall record missing fish beside ksh"
if [ "${FAILURES}" -eq 0 ]; then
  pass "ksh login shell keeps its profile and also configures fish"
fi

# Opting out of shell profile updates writes neither profile.
skip_home="${WORK}/homes/skip"
install_case "${WORK}/skip.log" "${skip_home}" /bin/bash "${FISH_PATH}" 0
assert_absent_file "${skip_home}/.bashrc" "shell update opt-out still wrote bashrc"
assert_absent_file "${skip_home}/.config/fish/config.fish" "shell update opt-out still wrote fish config"
assert_absent_file "${skip_home}/.config/fish/completions/rocm.fish" "shell update opt-out still wrote fish completions"
assert_log_contains "${WORK}/skip.log" "shell profile update skipped" "opt-out did not report the skip"
if [ "${FAILURES}" -eq 0 ]; then
  pass "shell profile opt-out leaves fish unconfigured"
fi

# A profile the user named still receives the login-shell snippet, and fish
# still receives its own.
custom_home="${WORK}/homes/custom"
custom_profile="${custom_home}/dotfiles/rocm-path.sh"
install_case "${WORK}/custom.log" "${custom_home}" /bin/bash "${FISH_PATH}" 1 "${custom_profile}"
assert_contains "${custom_profile}" "${POSIX_SNIPPET}" "custom profile missing the POSIX PATH block"
assert_contains "${custom_home}/.config/fish/config.fish" "${FISH_SNIPPET}" "custom profile path skipped fish"
assert_absent_file "${custom_home}/.bashrc" "custom profile path also wrote bashrc"
assert_contains "${custom_home}/.local/bin/.rocm-cli-path-edit" "profile ${custom_profile}" "uninstall record missing the custom profile"
assert_contains "${custom_home}/.local/bin/.rocm-cli-path-edit" "profile ${custom_home}/.config/fish/config.fish" "uninstall record missing fish beside the custom profile"
if [ "${FAILURES}" -eq 0 ]; then
  pass "a chosen login profile and fish are both recorded"
fi

if [ "${FAILURES}" -ne 0 ]; then
  echo "${FAILURES} failed" >&2
  exit 1
fi
echo "all install.sh fish checks passed"
