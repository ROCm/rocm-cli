
# WIP: Persistent app-dev MI300X CI runner

**Stage:** 4-design
**Pipeline:** standard
**Branch:** (none yet — infra/k8s, not a code branch)
**Last Updated:** 2026-07-10

---

## Problem

The `app-dev-gpu` self-hosted GitHub Actions runner (MI300X gfx942) that runs the
`@gpu` E2E jobs is NOT durable. It lives entirely inside the personal VS Code dev
workspace pod and will vanish when that pod is shut down.

Findings (verified 2026-07-10 on cluster context `app-dev`, ns `rocm-cli`):
- Host pod: `wb-dev-workspace-vscode-1782742332-03bb-*` (Deployment→ReplicaSet, the
  personal dev workspace — NOT a dedicated runner).
- Runner install: `/workload/actions-runner`, launched by a **bare `./run.sh`** in a
  shell (not a managed service). Registered agent name `app-dev-gpu`.
- **`/workload` is `emptyDir` (ephemeral)** — the runner install, its OAuth identity
  (`.credentials` + `.credentials_rsaparams` RSA key), and `.runner` config are all
  on ephemeral storage. Only `/workload/fredrik-espinoza-amd-com` is a real PVC
  (`pvc-fredrik-espinoza-amd-com`); the runner is NOT on it.
- The `e2e-test-runner` namespace is EMPTY — a red herring; the real runner is the
  bare process in the rocm-cli dev pod.

**Consequence:** shutting down the vscode workspace destroys the runner AND its RSA
identity (ephemeral). A replacement can't reuse the identity — it must **re-register**,
which needs a fresh registration token.

## How the current runner authenticates

Scheme = **OAuth** (not a stored PAT). At first `config.sh` registration a short-lived
registration token was consumed once; that produced an RSA keypair
(`.credentials_rsaparams`) the runner uses to sign OAuth requests for short-lived
access tokens. So the running runner holds only a runner-scoped RSA identity, not a
reusable GitHub credential — but that identity is on ephemeral storage and dies with
the pod.

## Solution (design — NOT yet built, user paused)

Dedicated runner **Deployment** (not tied to the vscode workspace), self-healing so a
pod restart re-registers automatically.

**Hardware spec to replicate (from the current dev pod):**
- image: `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.8.0`
- resources: `amd.com/gpu: 1`, cpu 1, memory 32Gi
- nodeSelector: `kaiwo/worker: "true"` (or pin `kaiwo/gpu-model: mi300x`)
- node = MI300X (`AMD_Instinct_MI300X_OAM`, gfx942, 192G, driver 6.14.14)
- runner labels must be `[self-hosted, linux, amd-gpu]` to match ci.yml jobs.

**The gating item — a registration credential must exist in the cluster** (none does
today). Options, least-privilege first:
- **GitHub App** (recommended for a standing runner): `Administration: write` on
  `ROCm/rocm-cli` ONLY; store private key as a k8s Secret; runner entrypoint mints a
  registration token on each start. Revocable independently, not tied to a user,
  auditable.
- **Fine-grained PAT**: repo-scoped `Administration: read & write`. Faster, but tied
  to the user's account + expiry to rotate. Fine to prototype with.
- User IS repo admin, so can create either. **Decision deferred** (user said "not now").

**Alternative: ARC (Actions Runner Controller)** — ephemeral per-job runners, handles
token exchange + lifecycle. Best isolation (per-job ephemeral filesystem) but more to
stand up. Overkill for one fixed runner; revisit if we want autoscaling.

## Security implications (why this needs care)

- Standing runner on a **public repo** + **self-hosted GPU** = fork-PR code could run
  on the MI300X. Already mitigated at repo level: fork-PR approval = "all external
  contributors" (see [[test-add-e2e-robot-framework]] hardening / [[ci-harden-actions]]).
- The **registration credential at rest** (PAT/App key as a Secret) is broader than the
  runner's own RSA key — anyone compromising the namespace gets it. GitHub App scoped to
  one repo minimizes blast radius vs a PAT.
- A persistent Deployment runner **reuses its filesystem across jobs** → one poisoned
  job can affect the next. ARC's per-job ephemerality is the mitigation if this matters.

## Next Steps
- 📋 User decides credential: GitHub App (recommended) vs PAT.
- 📋 Create credential, store as k8s Secret in `app-dev` ns.
- 📋 Write runner Deployment (spec above) + entrypoint that mints a registration token
  from the credential and runs `config.sh` + `run.sh` on start; `restartPolicy` via the
  Deployment for self-heal.
- 📋 Verify it registers as `app-dev-gpu` with labels `[self-hosted, linux, amd-gpu]`
  and picks up a dispatched `platform=app-dev-gpu` run.
- 📋 THEN safe to shut down the vscode workspace.

## Notes
- Manifests are infra, likely NOT in the rocm-cli repo (no runner manifests exist there
  today) — probably a separate infra repo / applied directly. Confirm where such k8s
  manifests belong before creating.
- Relates to the manual-dispatch loop ([[ci-manual-e2e]]): dispatch is proven, but
  app-dev dispatches queue forever with no runner online — this WIP unblocks that.
