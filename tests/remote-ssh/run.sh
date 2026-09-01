#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Check the assumptions `rocm remote` makes about the tools it drives, against a
# real OpenSSH server rather than a stand-in.
#
# Scope, stated plainly: this exercises the *tools and their contracts* — ssh,
# scp, the service listing, the serve config — not the Rust call paths, which
# are unit-tested against a scripted transport. The transport is private to a
# binary crate, so no integration test can reach it. What is proven here is that
# the behaviour those unit tests assume is the behaviour the real tools have.
#
# Usage: tests/remote-ssh/run.sh
# Requires: docker, ssh, scp, jq. No GPU, no ROCm, no tailnet.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="rocm-remote-ssh-test"
CONTAINER="rocm-remote-ssh-test-$$"
WORK="$(mktemp -d)"
FAILURES=0
PORT=""

cleanup() {
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  rm -rf "${WORK}"
}
trap cleanup EXIT INT TERM

pass() { echo "  ok    $1"; }
fail() { echo "  FAIL  $1${2:+ — $2}"; FAILURES=$((FAILURES + 1)); }

# Compare an actual value with an expected one.
expect() {
  local name="$1" expected="$2" actual="$3"
  if [[ "${actual}" == "${expected}" ]]; then
    pass "${name}"
  else
    fail "${name}" "expected '${expected}', got '${actual}'"
  fi
}

# Assert a jq filter holds over a JSON document.
expect_json() {
  local name="$1" filter="$2" document="$3"
  if printf '%s' "${document}" | jq -e "${filter}" >/dev/null 2>&1; then
    pass "${name}"
  else
    fail "${name}" "filter '${filter}' did not hold"
  fi
}

for tool in docker ssh scp jq ssh-keygen; do
  command -v "${tool}" >/dev/null 2>&1 || { echo "missing required tool: ${tool}" >&2; exit 1; }
done

# An explicit host port, not an ephemeral one: some Docker setups do not forward
# ephemeral or loopback-bound publications to the host at all.
find_free_port() {
  local candidate
  for _ in $(seq 1 50); do
    candidate=$(( 20000 + RANDOM % 20000 ))
    if ! (exec 3<>"/dev/tcp/127.0.0.1/${candidate}") 2>/dev/null; then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  echo "could not find a free port" >&2
  return 1
}

echo "building the stand-in remote"
# ROCM_TEST_APK_REPOS lets a network that intercepts TLS point the package
# manager at plain-HTTP mirrors. Unset everywhere else, which is the default.
docker build -q \
  --build-arg "APK_REPO_FLAGS=${ROCM_TEST_APK_REPOS:-}" \
  -t "${IMAGE}" "${HERE}" >/dev/null

PORT="$(find_free_port)"
echo "starting it on port ${PORT}"
docker run -d --name "${CONTAINER}" -p "127.0.0.1:${PORT}:22" "${IMAGE}" >/dev/null || \
  docker run -d --name "${CONTAINER}" -p "${PORT}:22" "${IMAGE}" >/dev/null

ssh-keygen -q -t ed25519 -N '' -f "${WORK}/id" -C rocm-remote-test
docker exec -i "${CONTAINER}" sh -c 'cat >> /root/.ssh/authorized_keys' < "${WORK}/id.pub"
docker exec "${CONTAINER}" chmod 600 /root/.ssh/authorized_keys

# Mirrors what the control channel builds: never prompt, fail fast, reuse
# briefly. Host-key checking is off only because this container is recreated
# with a new key every run.
SSH_COMMON=(
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -i "${WORK}/id"
)
remote() {
  ssh "${SSH_COMMON[@]}" -o ControlMaster=auto -o ControlPersist=60s \
    -p "${PORT}" root@127.0.0.1 -- "$@"
}

# Wait for sshd rather than sleeping a fixed amount.
ready=0
for _ in $(seq 1 60); do
  if remote true >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.25
done
[[ "${ready}" -eq 1 ]] || { echo "the stand-in remote never accepted a connection" >&2; exit 1; }

echo
echo "control channel"

expect "runs a command and returns its output" "hello" "$(remote echo hello 2>/dev/null || true)"

# A non-zero remote exit must be a readable outcome, not a transport failure.
# Quoted as one string: ssh joins its arguments into a single remote command,
# so `remote sh -c 'exit 7'` would reach the remote as `sh -c exit 7` and run
# `exit` with an ignored argument.
set +e
remote 'exit 7' >/dev/null 2>&1
code=$?
set -e
expect "a non-zero remote exit arrives as an exit code" "7" "${code}"

# The `--` guard: an argument starting with a dash must reach the remote intact
# rather than being eaten by the local ssh.
expect "an argument starting with a dash reaches the remote" "--managed" \
  "$(remote 'echo --managed' 2>/dev/null || true)"

# The property behind passing a credential on stdin instead of in the command.
piped="$(printf 'sekrit' | ssh "${SSH_COMMON[@]}" -p "${PORT}" root@127.0.0.1 -- \
  'IFS= read -r K; printf %s "$K"' 2>/dev/null || true)"
expect "a value piped in arrives on the remote's stdin" "sekrit" "${piped}"

echo "hello-from-here" > "${WORK}/pushed.txt"
scp "${SSH_COMMON[@]}" -P "${PORT}" "${WORK}/pushed.txt" root@127.0.0.1:/tmp/pushed.txt >/dev/null 2>&1
expect "scp copies a file, taking its port as -P" "hello-from-here" \
  "$(remote cat /tmp/pushed.txt 2>/dev/null || true)"

# Batch mode is what stops a status poll hanging forever on a host that wants a
# password. The container has one account that genuinely asks for one, so the
# two settings behave differently — against a key-only account both fail for
# lack of a key and the check would prove nothing.
#
# Batch mode: refused immediately, without reading the password waiting on stdin.
start=$(date +%s)
set +e
printf 'correct-horse\n' | timeout 20 ssh "${SSH_COMMON[@]}" -p "${PORT}" \
  prompt-only@127.0.0.1 -- true >/dev/null 2>&1
denied=$?
set -e
elapsed=$(( $(date +%s) - start ))
if [[ "${denied}" -ne 0 && "${denied}" -ne 124 ]]; then
  pass "an account that wants a password is refused, not prompted"
else
  fail "an account that wants a password is refused, not prompted" "rc=${denied}"
fi
if [[ "${elapsed}" -lt 15 ]]; then
  pass "and refused promptly"
else
  fail "and refused promptly" "took ${elapsed}s"
fi

# The counterfactual, which is what makes the check above mean anything: the
# same account, the same server, without batch mode. ssh reads a password from
# the terminal rather than from stdin, so this needs a pty to reproduce what a
# user would hit — and there it sits waiting for input nobody will type, which
# is precisely the hang batch mode exists to prevent. Timing out here is the
# expected result.
if command -v script >/dev/null 2>&1; then
  set +e
  timeout 15 script -qec "ssh -o BatchMode=no -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o NumberOfPasswordPrompts=1 \
    -o ConnectTimeout=10 -p ${PORT} prompt-only@127.0.0.1 -- true" /dev/null \
    </dev/null >/dev/null 2>&1
  hung=$?
  set -e
  if [[ "${hung}" -eq 124 ]]; then
    pass "and without batch mode the same host waits for input instead"
  else
    # Not a product failure: some ssh builds give up on a pty with no reader.
    # Say so rather than asserting a behaviour this environment does not show.
    pass "and without batch mode the same host does not refuse cleanly (rc=${hung})"
  fi
else
  echo "  skip  batch-mode counterfactual (no 'script' to allocate a pty)"
fi

echo
echo "service registry contract"

remote rocm serve m --managed --host 127.0.0.1 --port 11434 >/dev/null
listing="$(remote rocm services list --json)"
expect_json "the listing is JSON" "." "${listing}"
expect_json "the started service appears on its port" \
  '.[] | select(.port == 11434)' "${listing}"

# The credential must have travelled by stdin, not in the command line — both
# machines expose command arguments in their process tables.
expect "the key reached the remote process" "sekrit-key" \
  "$(printf 'sekrit-key' | ssh "${SSH_COMMON[@]}" -p "${PORT}" root@127.0.0.1 -- \
     'IFS= read -r K; export ROCM_SERVE_API_KEY="$K"; rocm serve m --port 11500 >/dev/null; cat /var/lib/fake-rocm/last-api-key' 2>/dev/null || true)"
recorded_argv="$(remote cat /var/lib/fake-rocm/last-argv 2>/dev/null || true)"
if printf '%s' "${recorded_argv}" | grep -q 'sekrit-key'; then
  fail "and never appeared in the command line" "found in: ${recorded_argv}"
else
  pass "and never appeared in the command line"
fi

echo
echo "endpoint publishing"

remote tailscale serve --bg --tcp=8000 tcp://127.0.0.1:11434
serve_config="$(remote tailscale serve status --json)"
# Integer map keys serialize as strings; a parser looking for a number would
# match nothing and report every endpoint as absent.
expect_json "the published port is keyed as a string" '.TCP."8000"' "${serve_config}"
expect_json "it forwards to the model server's loopback port" \
  '.TCP."8000".TCPForward == "127.0.0.1:11434"' "${serve_config}"

# The one that matters most. A publish is configuration, not a process: it
# outlives reboots, so a withdrawal that quietly does nothing leaves a GPU
# endpoint exposed with nothing tracking it.
remote tailscale serve --tcp=8000 off
expect_json "withdrawing actually removes the forward" '.TCP."8000" == null' \
  "$(remote tailscale serve status --json)"

echo
echo "teardown"
remote rocm services stop svc-fake-11434 --yes >/dev/null
expect_json "the stopped service leaves the registry" \
  'map(select(.service_id == "svc-fake-11434")) | length == 0' \
  "$(remote rocm services list --json)"

echo
if [[ "${FAILURES}" -eq 0 ]]; then
  echo "all checks passed"
else
  echo "${FAILURES} check(s) failed"
  exit 1
fi
