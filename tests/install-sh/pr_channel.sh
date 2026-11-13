#!/bin/sh
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Drive install.sh's `pr-<number>` channel with a stub `gh`.
#
# The channel installs the bundle CI built for a pull request. That bundle is a
# workflow artifact, so the installer resolves the PR to a CI run through `gh`
# and then runs the same checks as for any archive on disk. What matters here:
#
# - the run that supplies the bundle is the newest one that still has an
#   unexpired artifact, found by run id, not the newest run or the first listed
# - the pinned release signing keys do not apply (CI does not sign PR bundles),
#   so the bundle installs on checksum alone, and asking for a signature
#   anyway fails instead of being skipped
# - a bad checksum, an unknown PR, a PR with no CI run or only expired builds,
#   a malformed channel and a missing `gh` each stop the install with a message
#   naming the cause, and nothing is installed
#
# The stub answers the three `gh` calls the installer makes with what the real
# `--jq` filters would print, so this checks the installer's decisions, not
# those filters: they were run against the live API when the channel was written.
#
# Usage: tests/install-sh/pr_channel.sh
# Set INSTALL_SH to test another copy of the installer (mutation checks).
# Requires: tar, sha256sum, and a POSIX sh. No network and no GPU.

set -eu

ROOT="$(CDPATH='' cd -- "$(dirname "$0")/../.." && pwd)"
INSTALLER="${INSTALL_SH:-${ROOT}/install.sh}"
mkdir -p "${ROOT}/.rocm-work"
WORK="$(mktemp -d "${ROOT}/.rocm-work/install-pr.XXXXXX")"
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

for cmd in tar gzip sha256sum awk grep sed find mktemp install; do
  command -v "${cmd}" >/dev/null 2>&1 || {
    echo "missing required command: ${cmd}" >&2
    exit 1
  }
done

TOOLS="${WORK}/tools"
GH_BIN="${WORK}/gh-bin"
mkdir -p "${TOOLS}" "${GH_BIN}" "${WORK}/tmp"
for cmd in sh cat awk sed grep rm mkdir mktemp tar gzip sha256sum find head cp uname install dirname; do
  ln -s "$(command -v "${cmd}")" "${TOOLS}/${cmd}"
done

# What the installer reads from `gh`, taken from files in ${GH_STUB_DIR}:
#   head              the pull request's head commit (missing: `gh pr view` fails;
#                     empty, with head-empty present: it succeeds and prints nothing)
#   runs              one "<id> <status> <conclusion> [<workflow path>]" line per
#                     run, newest first; the path defaults to the CI workflow
#   artifacts-<id>    how many unexpired artifacts that run still has
#   expired-<id>      how many expired ones it also lists
#   artifacts-fail-<id>  present: listing that run's artifacts fails
#   bundle            directory whose files `gh run download` copies into -D
#   calls             every invocation, appended
#
# The stub is as strict as the real API: it fails a call that names another
# repository than ${GH_STUB_REPO} (default ROCm/rocm-cli), lists runs for any
# commit but the pull request's head, asks for another artifact than
# rocm-cli-pr-linux-amd64, or downloads without naming it. An installer that
# drops one of those fails here instead of passing against a lenient stub.
# It also drains its stdin, as a real `gh` may: under `curl | sh` that is the
# rest of the installer, which an unredirected call would swallow.
cat > "${GH_BIN}/gh" <<'EOF'
#!/bin/sh
cat > /dev/null
printf '%s\n' "$*" >> "${GH_STUB_DIR}/calls"
repo="${GH_STUB_REPO:-ROCm/rocm-cli}"
artifact="rocm-cli-pr-linux-amd64"
die() { echo "gh stub: $*" >&2; exit 1; }
arg_after() {
  flag="$1"
  shift
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "${flag}" ]; then
      printf '%s\n' "${2:-}"
      return 0
    fi
    shift
  done
}
case "$1 $2" in
  "pr view")
    [ "$(arg_after -R "$@")" = "${repo}" ] || die "pr view without -R ${repo}: $*"
    # Without these a real gh prints a human summary, which would become the commit.
    [ "$(arg_after --json "$@")" = "headRefOid" ] || die "pr view without --json headRefOid: $*"
    [ "$(arg_after --jq "$@")" = ".headRefOid" ] || die "pr view without --jq .headRefOid: $*"
    if [ -e "${GH_STUB_DIR}/head-empty" ]; then
      exit 0
    fi
    [ -s "${GH_STUB_DIR}/head" ] || { echo "GraphQL: Could not resolve to a PullRequest" >&2; exit 1; }
    cat "${GH_STUB_DIR}/head"
    ;;
  "api repos/"*)
    case "$2" in
      "repos/${repo}/actions/runs?"*)
        case "$2" in
          *"head_sha=$(cat "${GH_STUB_DIR}/head")"*) ;;
          *) die "run listing for another commit than the pull request's head: $2" ;;
        esac
        case "$2" in
          *event=pull_request*) ;;
          *) die "run listing not limited to pull request runs: $2" ;;
        esac
        # GitHub lists 30 runs by default; a commit with more could hide the one with a build.
        case "$2" in
          *per_page=100*) ;;
          *) die "run listing without per_page=100: $2" ;;
        esac
        # What `--jq` would leave of the listing: the workflow filter is applied
        # only when the installer's expression carries it, so an installer that
        # drops it is handed the other workflows' runs too.
        jq="$(arg_after --jq "$@")"
        while read -r id state conclusion path; do
          [ -n "${id}" ] || continue
          path="${path:-.github/workflows/ci.yml}"
          case "${jq}" in
            *'.path == ".github/workflows/ci.yml"'*) [ "${path}" = ".github/workflows/ci.yml" ] || continue ;;
          esac
          printf '%s %s %s\n' "${id}" "${state}" "${conclusion}"
        done < "${GH_STUB_DIR}/runs"
        ;;
      "repos/${repo}/actions/runs/"*/artifacts\?name="${artifact}")
        run="${2#*/actions/runs/}"
        run="${run%%/*}"
        [ ! -e "${GH_STUB_DIR}/artifacts-fail-${run}" ] || die "artifact listing failed for run ${run}"
        # The real listing also carries expired artifacts; they are left out
        # only when the installer's expression filters them.
        live="$(cat "${GH_STUB_DIR}/artifacts-${run}" 2>/dev/null || echo 0)"
        expired="$(cat "${GH_STUB_DIR}/expired-${run}" 2>/dev/null || echo 0)"
        case "$(arg_after --jq "$@")" in
          *'select(.expired | not)'*) echo "${live}" ;;
          *) echo $((live + expired)) ;;
        esac
        ;;
      *) die "unexpected gh api path: $2" ;;
    esac
    ;;
  "run download")
    run="$3"
    [ "$(arg_after -R "$@")" = "${repo}" ] || die "run download without -R ${repo}: $*"
    [ "$(arg_after -n "$@")" = "${artifact}" ] || die "run download without -n ${artifact}: $*"
    dest="$(arg_after -D "$@")"
    [ -n "${dest}" ] || die "run download without -D: $*"
    printf '%s\n' "downloaded ${run}" >> "${GH_STUB_DIR}/calls"
    mkdir -p "${dest}"
    cp "${GH_STUB_DIR}/bundle/"* "${dest}/"
    ;;
  *) die "unexpected invocation: $*" ;;
esac
EOF
chmod 755 "${GH_BIN}/gh"

BUNDLE_SRC="${WORK}/bundle-src/rocm-cli-pr-linux-amd64/bin"
mkdir -p "${BUNDLE_SRC}"
printf '%s\n' '#!/bin/sh' 'exit 0' > "${BUNDLE_SRC}/rocm"
printf '%s\n' '#!/bin/sh' 'exit 0' > "${BUNDLE_SRC}/rocmd"
chmod 755 "${BUNDLE_SRC}/rocm" "${BUNDLE_SRC}/rocmd"
GOOD="${WORK}/artifact-good"
mkdir -p "${GOOD}"
tar -C "${WORK}/bundle-src" -czf "${GOOD}/rocm-cli-pr-linux-amd64.tar.gz" rocm-cli-pr-linux-amd64
sha256sum "${GOOD}/rocm-cli-pr-linux-amd64.tar.gz" | awk '{print $1}' > "${GOOD}/rocm-cli-pr-linux-amd64.tar.gz.sha256"
BAD="${WORK}/artifact-bad"
mkdir -p "${BAD}"
cp "${GOOD}/rocm-cli-pr-linux-amd64.tar.gz" "${BAD}/"
printf '%s\n' 0000000000000000000000000000000000000000000000000000000000000000 > "${BAD}/rocm-cli-pr-linux-amd64.tar.gz.sha256"

HEAD_SHA=53546b007a85445a2466b28447307992733829c9

# stub_reset <name>: a fresh stub directory for one case, echoed.
stub_reset() {
  dir="${WORK}/stub-$1"
  rm -rf "${dir}"
  mkdir -p "${dir}"
  printf '%s\n' "${HEAD_SHA}" > "${dir}/head"
  : > "${dir}/runs"
  : > "${dir}/calls"
  ln -s "${GOOD}" "${dir}/bundle"
  printf '%s\n' "${dir}"
}

# run_install <case> <stub-dir> <path> <channel> [NAME=value ...]
# Leaves the output in ${WORK}/<case>.log and the exit status in ${STATUS}.
run_install() {
  name="$1"
  stub="$2"
  path="$3"
  channel="$4"
  shift 4
  home="${WORK}/homes/${name}"
  mkdir -p "${home}"
  STATUS=0
  env -i \
    HOME="${home}" \
    SHELL=/bin/sh \
    PATH="${path}" \
    TMPDIR="${WORK}/tmp" \
    GH_STUB_DIR="${stub}" \
    ROCM_CLI_INSTALL_DIR="${home}/.local/bin" \
    ROCM_CLI_UPDATE_SHELL_PATH=0 \
    "$@" \
    "${TOOLS}/sh" "${INSTALLER}" "${channel}" < /dev/null > "${WORK}/${name}.log" 2>&1 || STATUS=$?
}

# run_install_piped <case> <stub-dir> <channel>: the installer arrives on stdin,
# as under `curl | sh`. Anything that reads that stdin, a `gh` call that is not
# redirected from /dev/null, swallows the rest of the script and the install
# stops half way, so a failing install here names the cause.
run_install_piped() {
  name="$1"
  stub="$2"
  channel="$3"
  home="${WORK}/homes/${name}"
  mkdir -p "${home}"
  STATUS=0
  "${TOOLS}/cat" "${INSTALLER}" | env -i \
    HOME="${home}" \
    SHELL=/bin/sh \
    PATH="${PATH_WITH_GH}" \
    TMPDIR="${WORK}/tmp" \
    GH_STUB_DIR="${stub}" \
    ROCM_CLI_INSTALL_DIR="${home}/.local/bin" \
    ROCM_CLI_UPDATE_SHELL_PATH=0 \
    "${TOOLS}/sh" -s -- "${channel}" > "${WORK}/${name}.log" 2>&1 || STATUS=$?
}

assert_log_contains() {
  if ! grep -F -q -- "$2" "$1"; then
    fail "$3"
    echo "---- log ----" >&2
    cat "$1" >&2
  fi
}

assert_log_lacks() {
  if grep -F -q -- "$2" "$1"; then
    fail "$3"
    echo "---- log ----" >&2
    cat "$1" >&2
  fi
}

assert_installed() {
  if [ ! -x "${WORK}/homes/$1/.local/bin/rocm" ] || [ ! -x "${WORK}/homes/$1/.local/bin/rocmd" ]; then
    fail "$2"
  fi
}

assert_not_installed() {
  if [ -e "${WORK}/homes/$1/.local/bin/rocm" ] || [ -e "${WORK}/homes/$1/.local/bin/.rocm-cli-manifest" ]; then
    fail "$2"
  fi
}

assert_absent_dir() {
  if [ -e "$1" ]; then
    fail "$2"
  fi
}

assert_failed() {
  if [ "${STATUS}" -eq 0 ]; then
    fail "$2"
    cat "${WORK}/$1.log" >&2
  fi
}

PATH_WITH_GH="${GH_BIN}:${TOOLS}"

# Happy path. The newest run's artifacts have expired, the one before has a
# live artifact, and an older one has too: the install must use the newest run
# that still has a build, by its id.
stub="$(stub_reset happy)"
# Also: a newer run of another workflow with a live artifact, which only a
# path filter keeps out, and expired artifacts on the newest CI run, which only
# the expiry filter keeps out.
cat > "${stub}/runs" <<'EOF'
3500 completed success .github/workflows/codeql.yml
3003 completed failure
2002 completed failure
1001 completed success
EOF
printf '%s\n' 1 > "${stub}/artifacts-3500"
printf '%s\n' 0 > "${stub}/artifacts-3003"
printf '%s\n' 2 > "${stub}/expired-3003"
printf '%s\n' 1 > "${stub}/artifacts-2002"
printf '%s\n' 1 > "${stub}/artifacts-1001"
run_install happy "${stub}" "${PATH_WITH_GH}" pr-1048
if [ "${STATUS}" -ne 0 ]; then
  fail "pr-1048 install exited ${STATUS}"
  cat "${WORK}/happy.log" >&2
fi
assert_installed happy "pr-1048 did not install rocm and rocmd"
assert_log_contains "${WORK}/happy.log" "channel: pr-1048" "channel was not reported"
assert_log_contains "${WORK}/happy.log" "pull request: #1048 at 53546b0" "pull request and short commit were not reported"
# The substring above is also in the full commit, so the abbreviation is
# pinned by the full commit not appearing.
assert_log_lacks "${WORK}/happy.log" "${HEAD_SHA}" "the commit was reported in full, not abbreviated"
assert_log_contains "${WORK}/happy.log" "CI run: 2002 (completed/failure)" "installer did not use the newest run that still has an artifact"
assert_log_contains "${WORK}/happy.log" "unsigned build" "installer did not warn that the build is unsigned"
assert_log_lacks "${WORK}/happy.log" "signature verified" "a pull request bundle was verified against the pinned release key"
assert_log_contains "${stub}/calls" "pr view 1048 -R ROCm/rocm-cli" "pull request was not looked up in the default repository"
assert_log_contains "${stub}/calls" "downloaded 2002" "the artifact was not downloaded from run 2002"
assert_log_lacks "${stub}/calls" "downloaded 3003" "an expired run was downloaded"
assert_log_lacks "${stub}/calls" "downloaded 1001" "an older run was downloaded"
if [ "${FAILURES}" -eq 0 ]; then
  pass "pr channel installs the newest unexpired CI build on checksum alone"
fi

# ROCM_CLI_GITHUB_REPO moves every gh call to that repository.
stub="$(stub_reset repo)"
printf '%s\n' '9 completed success' > "${stub}/runs"
printf '%s\n' 1 > "${stub}/artifacts-9"
run_install repo "${stub}" "${PATH_WITH_GH}" pr-7 ROCM_CLI_GITHUB_REPO=someone/fork GH_STUB_REPO=someone/fork
assert_installed repo "a fork repository install did not complete"
assert_log_contains "${stub}/calls" "pr view 7 -R someone/fork" "ROCM_CLI_GITHUB_REPO was ignored for the pull request lookup"
assert_log_contains "${stub}/calls" "run download 9 -R someone/fork" "ROCM_CLI_GITHUB_REPO was ignored for the download"
if [ "${FAILURES}" -eq 0 ]; then
  pass "pr channel follows ROCM_CLI_GITHUB_REPO"
fi

# A checksum that does not match is rejected before anything is installed.
stub="$(stub_reset checksum)"
printf '%s\n' '5 completed success' > "${stub}/runs"
printf '%s\n' 1 > "${stub}/artifacts-5"
rm "${stub}/bundle"
ln -s "${BAD}" "${stub}/bundle"
run_install checksum "${stub}" "${PATH_WITH_GH}" pr-1048
assert_failed checksum "a bundle with the wrong checksum installed"
assert_log_contains "${WORK}/checksum.log" "checksum verification failed" "a wrong checksum was not reported"
assert_not_installed checksum "a bundle with the wrong checksum was activated"
if [ "${FAILURES}" -eq 0 ]; then
  pass "pr channel rejects a bundle whose checksum does not match"
fi

# Asking for a signature on a bundle that CI never signs fails; it is not
# skipped, and nothing is installed.
stub="$(stub_reset require-sig)"
printf '%s\n' '5 completed success' > "${stub}/runs"
printf '%s\n' 1 > "${stub}/artifacts-5"
run_install require-sig "${stub}" "${PATH_WITH_GH}" pr-1048 ROCM_CLI_REQUIRE_SIGNATURE=1
assert_failed require-sig "a required signature was skipped for a pull request bundle"
assert_log_contains "${WORK}/require-sig.log" "a signature cannot be required for the pr-1048 channel" "a required signature on a pull request bundle was not refused with a cause the user can act on"
assert_log_lacks "${WORK}/require-sig.log" "ROCM_CLI_SIGNING_PUBLIC_KEY" "a required signature on a pull request bundle pointed at a key, which cannot help"
assert_not_installed require-sig "a pull request bundle was installed although a signature was required"
if [ "${FAILURES}" -eq 0 ]; then
  pass "pr channel refuses a required signature it cannot get"
fi

# A key the user names is not cleared along with the pinned ones: it asks for
# verification, and the bundle has no signature to give it, so the install
# fails there. Each variable is pinned on its own, because the code that spares
# a named key tests both. The key is a placeholder: the install stops on the
# missing signature before any key is read.
KEY_PEM='-----BEGIN PUBLIC KEY-----
placeholder
-----END PUBLIC KEY-----'
printf '%s\n' "${KEY_PEM}" > "${WORK}/placeholder.pem"
for variant in path pem; do
  stub="$(stub_reset "key-${variant}")"
  printf '%s\n' '5 completed success' > "${stub}/runs"
  printf '%s\n' 1 > "${stub}/artifacts-5"
  if [ "${variant}" = path ]; then
    run_install "key-${variant}" "${stub}" "${PATH_WITH_GH}" pr-1048 ROCM_CLI_SIGNING_PUBLIC_KEY_PATH="${WORK}/placeholder.pem"
  else
    run_install "key-${variant}" "${stub}" "${PATH_WITH_GH}" pr-1048 ROCM_CLI_SIGNING_PUBLIC_KEY_PEM="${KEY_PEM}"
  fi
  assert_failed "key-${variant}" "a pull request bundle installed although a public key (${variant}) was supplied"
  assert_log_contains "${WORK}/key-${variant}.log" "required signature sidecar is missing" "a supplied public key (${variant}) did not ask for the signature the bundle lacks"
  assert_not_installed "key-${variant}" "a pull request bundle was installed although a public key (${variant}) was supplied"
done
if [ "${FAILURES}" -eq 0 ]; then
  pass "pr channel does not skip a public key the user supplied"
fi

# No such pull request.
stub="$(stub_reset unknown-pr)"
: > "${stub}/head"
run_install unknown-pr "${stub}" "${PATH_WITH_GH}" pr-99999
assert_failed unknown-pr "an unknown pull request installed"
assert_log_contains "${WORK}/unknown-pr.log" "could not look up pull request #99999" "an unknown pull request was not reported"
assert_not_installed unknown-pr "an unknown pull request left an install"

# The pull request has no CI run yet.
stub="$(stub_reset no-run)"
run_install no-run "${stub}" "${PATH_WITH_GH}" pr-1048
assert_failed no-run "a pull request with no CI run installed"
assert_log_contains "${WORK}/no-run.log" "no CI run found for pull request #1048 at 53546b0" "a pull request with no CI run was not reported"
assert_not_installed no-run "a pull request with no CI run left an install"

# Every build has expired, or none was made.
stub="$(stub_reset expired)"
printf '%s\n' '4 completed success' '3 completed success' > "${stub}/runs"
printf '%s\n' 0 > "${stub}/artifacts-4"
run_install expired "${stub}" "${PATH_WITH_GH}" pr-1048
assert_failed expired "a pull request whose builds expired installed"
assert_log_contains "${WORK}/expired.log" "no downloadable rocm-cli-pr-linux-amd64 build for pull request #1048" "expired builds were not reported"
assert_log_contains "${WORK}/expired.log" "7 days" "the 7 day retention was not explained"
assert_not_installed expired "a pull request whose builds expired left an install"
if [ "${FAILURES}" -eq 0 ]; then
  pass "pr channel reports an unknown pull request, no CI run and expired builds"
fi

# A channel that is not pr-<number> is refused before anything is looked up,
# not treated as a release tag.
for bad in pr-abc pr- pr-12x; do
  stub="$(stub_reset "bad-${bad}")"
  run_install "bad-${bad}" "${stub}" "${PATH_WITH_GH}" "${bad}"
  assert_failed "bad-${bad}" "channel ${bad} installed"
  assert_log_contains "${WORK}/bad-${bad}.log" "invalid channel: ${bad}" "channel ${bad} was not refused as malformed"
  if [ -s "${stub}/calls" ]; then
    fail "channel ${bad} reached gh"
  fi
done
if [ "${FAILURES}" -eq 0 ]; then
  pass "pr channel refuses a malformed number"
fi

# A pull request with a head commit that gh reports as empty.
stub="$(stub_reset empty-head)"
: > "${stub}/head-empty"
run_install empty-head "${stub}" "${PATH_WITH_GH}" pr-1048
assert_failed empty-head "an empty head commit installed"
assert_log_contains "${WORK}/empty-head.log" "has no head commit" "an empty head commit was not reported"
assert_not_installed empty-head "an empty head commit left an install"

# A run whose artifact list cannot be read is skipped, never taken as having a
# build: the next run, which has one, supplies the bundle.
stub="$(stub_reset lookup-fails)"
printf '%s\n' '6 completed success' '5 completed success' > "${stub}/runs"
: > "${stub}/artifacts-fail-6"
printf '%s\n' 1 > "${stub}/artifacts-5"
run_install lookup-fails "${stub}" "${PATH_WITH_GH}" pr-1048
assert_installed lookup-fails "a failed artifact listing stopped the install instead of trying the next run"
assert_log_contains "${WORK}/lookup-fails.log" "CI run: 5 (" "the run after the failed listing was not used"
assert_log_contains "${WORK}/lookup-fails.log" "could not list the artifacts of CI run 6" "a failed artifact listing was skipped without saying so"
assert_log_lacks "${stub}/calls" "downloaded 6" "a run whose artifact listing failed was downloaded"
if [ "${FAILURES}" -eq 0 ]; then
  pass "pr channel refuses an empty head and skips a run it cannot list"
fi

# Settings that name another source for the bundle are refused, not ignored.
stub="$(stub_reset archive-conflict)"
printf '%s\n' '5 completed success' > "${stub}/runs"
printf '%s\n' 1 > "${stub}/artifacts-5"
run_install archive-conflict "${stub}" "${PATH_WITH_GH}" pr-1048 ROCM_CLI_ARCHIVE="${GOOD}/rocm-cli-pr-linux-amd64.tar.gz"
assert_failed archive-conflict "ROCM_CLI_ARCHIVE with a pull request channel installed"
assert_log_contains "${WORK}/archive-conflict.log" "ROCM_CLI_ARCHIVE cannot be combined with the pr-1048 channel" "ROCM_CLI_ARCHIVE with a pull request channel was not refused"
assert_not_installed archive-conflict "ROCM_CLI_ARCHIVE with a pull request channel left an install"
stub="$(stub_reset base-conflict)"
printf '%s\n' '5 completed success' > "${stub}/runs"
printf '%s\n' 1 > "${stub}/artifacts-5"
run_install base-conflict "${stub}" "${PATH_WITH_GH}" pr-1048 ROCM_CLI_DOWNLOAD_BASE=https://mirror.example/rocm
assert_failed base-conflict "ROCM_CLI_DOWNLOAD_BASE with a pull request channel installed"
assert_log_contains "${WORK}/base-conflict.log" "ROCM_CLI_DOWNLOAD_BASE cannot be combined with the pr-1048 channel" "ROCM_CLI_DOWNLOAD_BASE with a pull request channel was not refused"
assert_not_installed base-conflict "ROCM_CLI_DOWNLOAD_BASE with a pull request channel left an install"
if [ -s "${stub}/calls" ]; then
  fail "a refused ROCM_CLI_DOWNLOAD_BASE still reached gh"
fi
stub="$(stub_reset download-only)"
printf '%s\n' '5 completed success' > "${stub}/runs"
printf '%s\n' 1 > "${stub}/artifacts-5"
run_install download-only "${stub}" "${PATH_WITH_GH}" pr-1048 ROCM_CLI_DOWNLOAD_ONLY=1 ROCM_CLI_DOWNLOAD_DIR="${WORK}/exported"
assert_failed download-only "ROCM_CLI_DOWNLOAD_ONLY with a pull request channel exported a bundle"
assert_log_contains "${WORK}/download-only.log" "ROCM_CLI_DOWNLOAD_ONLY cannot be combined with the pr-1048 channel" "ROCM_CLI_DOWNLOAD_ONLY with a pull request channel was not refused"
assert_absent_dir "${WORK}/exported" "a refused ROCM_CLI_DOWNLOAD_ONLY still wrote a bundle"
if [ -s "${stub}/calls" ]; then
  fail "a refused ROCM_CLI_DOWNLOAD_ONLY still reached gh"
fi
if [ "${FAILURES}" -eq 0 ]; then
  pass "pr channel refuses a second source for the bundle"
fi

# The installer arriving on stdin, as under `curl | sh`, still installs.
stub="$(stub_reset piped)"
printf '%s\n' '5 completed success' > "${stub}/runs"
printf '%s\n' 1 > "${stub}/artifacts-5"
run_install_piped piped "${stub}" pr-1048
if [ "${STATUS}" -ne 0 ]; then
  fail "piped pr-1048 install exited ${STATUS}"
  cat "${WORK}/piped.log" >&2
fi
assert_installed piped "a pr channel install with the installer on stdin did not complete"
assert_log_contains "${WORK}/piped.log" "installed:" "a pr channel install with the installer on stdin stopped before installing"
if [ "${FAILURES}" -eq 0 ]; then
  pass "pr channel works with the installer on stdin"
fi

# gh is required, and its absence is named.
stub="$(stub_reset no-gh)"
run_install no-gh "${stub}" "${TOOLS}" pr-1048
assert_failed no-gh "an install without gh succeeded"
assert_log_contains "${WORK}/no-gh.log" "missing required command: gh" "a missing gh was not named"
assert_not_installed no-gh "an install without gh left files"
if [ "${FAILURES}" -eq 0 ]; then
  pass "pr channel names a missing gh"
fi

if [ "${FAILURES}" -ne 0 ]; then
  echo "${FAILURES} failed" >&2
  exit 1
fi
echo "all install.sh pr channel checks passed"
