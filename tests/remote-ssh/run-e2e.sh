#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Drive the real `rocm remote` end to end against a stand-in GPU machine.
#
# Unlike run.sh, which checks that the tools behave as assumed, this runs the
# actual binary through the whole orchestration: discover a target, probe it,
# start a model, publish an endpoint, reconcile status, re-publish after the
# endpoint is withdrawn out of band, and tear down. Real `ssh`, real argument
# building, real session records on disk.
#
# What is faked, and therefore what this does NOT prove: there is no GPU, no
# model, and no tailnet. `tailscale` is a stub on both sides, so the published
# endpoint does not carry traffic — reachability is the one thing here that
# still needs a real two-node tailnet to confirm.
#
# Usage: tests/remote-ssh/run-e2e.sh
# Requires: docker, ssh, jq, and a built `rocm` (cargo build -p rocm).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
IMAGE="rocm-remote-ssh-test"
CONTAINER="rocm-remote-e2e-$$"
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

expect_contains() {
  local name="$1" needle="$2" haystack="$3"
  if [[ "${haystack}" == *"${needle}"* ]]; then
    pass "${name}"
  else
    fail "${name}" "expected to find '${needle}' in:"$'\n'"${haystack}"
  fi
}

expect_absent() {
  local name="$1" needle="$2" haystack="$3"
  if [[ "${haystack}" != *"${needle}"* ]]; then
    pass "${name}"
  else
    fail "${name}" "did not expect '${needle}' in:"$'\n'"${haystack}"
  fi
}

for tool in docker ssh jq ssh-keygen; do
  command -v "${tool}" >/dev/null 2>&1 || { echo "missing required tool: ${tool}" >&2; exit 1; }
done

ROCM_BIN="${ROCM_BIN:-${REPO}/target/debug/rocm}"
[[ -x "${ROCM_BIN}" ]] || { echo "no rocm binary at ${ROCM_BIN}; run: cargo build -p rocm" >&2; exit 1; }

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
docker build -q --build-arg "APK_REPO_FLAGS=${ROCM_TEST_APK_REPOS:-}" -t "${IMAGE}" "${HERE}" >/dev/null

PORT="$(find_free_port)"
echo "starting it on port ${PORT}"
docker run -d --name "${CONTAINER}" -p "127.0.0.1:${PORT}:22" "${IMAGE}" >/dev/null

ssh-keygen -q -t ed25519 -N '' -f "${WORK}/id" -C rocm-remote-e2e
docker exec -i "${CONTAINER}" sh -c 'cat >> /root/.ssh/authorized_keys' < "${WORK}/id.pub"
docker exec "${CONTAINER}" chmod 600 /root/.ssh/authorized_keys

# An isolated HOME: the CLI reads ~/.ssh/config through the system ssh, and puts
# its own session records under ~/.rocm. Both should be this run's, not the
# developer's.
export HOME="${WORK}/home"
mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"
cat > "${HOME}/.ssh/config" <<EOF
Host gpu-box
  HostName 127.0.0.1
  Port ${PORT}
  User root
  IdentityFile ${WORK}/id
  IdentitiesOnly yes
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
EOF
chmod 600 "${HOME}/.ssh/config"
# ssh resolves ~/.ssh/config from the account database, not from HOME, so an
# isolated config has to be named explicitly or the real user's would be used.
export ROCM_REMOTE_SSH_CONFIG="${HOME}/.ssh/config"

# The local half of the tailnet: one online peer named exactly as the ssh alias,
# so the CLI's own resolver has something to find.
cat > "${WORK}/tailscale-status.json" <<'EOF'
{
  "BackendState": "Running",
  "TUN": true,
  "MagicDNSSuffix": "example-tailnet.ts.net",
  "Self": { "HostName": "laptop", "DNSName": "laptop.example-tailnet.ts.net.", "Online": true },
  "Peer": {
    "nodekey:aaa": {
      "HostName": "gpu-box",
      "DNSName": "gpu-box.example-tailnet.ts.net.",
      "OS": "linux",
      "TailscaleIPs": ["100.88.14.21"],
      "Tags": ["tag:gpu"],
      "Online": true
    }
  }
}
EOF
mkdir -p "${WORK}/bin"
cp "${HERE}/local-tailscale.sh" "${WORK}/bin/tailscale"
chmod +x "${WORK}/bin/tailscale"
export FAKE_LOCAL_TAILSCALE_STATUS="${WORK}/tailscale-status.json"
export PATH="${WORK}/bin:${PATH}"

rocm() { "${ROCM_BIN}" "$@" 2>&1; }
in_container() { docker exec "${CONTAINER}" "$@"; }

for _ in $(seq 1 60); do
  ssh -o BatchMode=yes -o ConnectTimeout=5 -F "${ROCM_REMOTE_SSH_CONFIG}" gpu-box true >/dev/null 2>&1 && break
  sleep 0.25
done

echo
echo "discovery"
out="$(rocm remote targets)"
expect_contains "the machine is discovered" "- gpu-box" "${out}"
expect_contains "and reported online" "online: yes" "${out}"
out="$(rocm remote targets --tag gpu)"
expect_contains "and can be narrowed by tag" "- gpu-box" "${out}"

echo
echo "health"
out="$(rocm remote doctor gpu-box || true)"
expect_contains "the remote's own state is fetched and scored here" "Health of gpu-box" "${out}"

echo
echo "serve"
out="$(rocm remote serve gpu-box test-model)"
expect_contains "a model is served" "Model serving on gpu-box" "${out}"
expect_contains "and an endpoint is printed" "http://gpu-box.example-tailnet.ts.net:8000/v1" "${out}"
expect_contains "with a credential" "api key:" "${out}"
expect_contains "and who can reach it is stated" "every machine on your tailnet" "${out}"

serve_config="$(in_container tailscale serve status --json)"
expect_contains "the remote published the endpoint" '"8000"' "${serve_config}"

# The credential must have arrived by stdin, not on a command line either
# machine exposes in its process table.
recorded_argv="$(in_container cat /var/lib/fake-rocm/last-argv)"
recorded_key="$(in_container cat /var/lib/fake-rocm/last-api-key)"
if [[ -n "${recorded_key}" && "${recorded_argv}" != *"${recorded_key}"* ]]; then
  pass "the credential reached the model without appearing in its arguments"
else
  fail "the credential reached the model without appearing in its arguments" \
    "key='${recorded_key}' argv='${recorded_argv}'"
fi

echo
echo "status"
out="$(rocm remote status)"
expect_contains "the model is reported healthy" "model server: healthy" "${out}"
expect_contains "and the endpoint published" "endpoint published: yes" "${out}"

echo
echo "the endpoint is withdrawn behind our back"
in_container tailscale serve --tcp=8000 off
out="$(rocm remote status)"
expect_contains "the model is still healthy" "model server: healthy" "${out}"
expect_contains "but the endpoint is reported gone" "endpoint published: no" "${out}"
expect_contains "and re-publishing is the suggested fix" "rocm remote attach" "${out}"

session_id="$(rocm remote status | sed -n 's/^- \(remote-.*\)$/\1/p' | head -n1)"
out="$(rocm remote attach "${session_id}")"
expect_contains "attaching restores the endpoint" "Endpoint re-published" "${out}"
expect_contains "without restarting the model" "not restarted" "${out}"

echo
echo "teardown"
out="$(rocm remote stop "${session_id}")"
expect_contains "the endpoint is withdrawn" "endpoint withdrawn: yes" "${out}"
expect_contains "and the model stopped" "model server stopped: yes" "${out}"

serve_config="$(in_container tailscale serve status --json)"
expect_absent "nothing is left published on the machine" '"8000"' "${serve_config}"
out="$(rocm remote status)"
expect_contains "and the session is gone from this machine" "No remote sessions" "${out}"

echo
if [[ "${FAILURES}" -eq 0 ]]; then
  echo "all checks passed"
else
  echo "${FAILURES} check(s) failed"
  exit 1
fi
