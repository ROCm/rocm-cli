
# WIP: Persistent app-dev MI300X CI runner

**Stage:** 4-design
**Pipeline:** standard
**Branch:** (none yet — infra/k8s, not a code branch)
**Last Updated:** 2026-07-16

---

## Work Log

---

## 2026-07-16 — Runner was offline; manually restored (stopgap)

The vscode workspace pod had cycled (now `wb-dev-workspace-vscode-1784186242-6168-755bf86c47-b9rsw`,
was `...1782742332-03bb`). As predicted, ephemeral `/workload/actions-runner` was gone and
`app-dev-gpu` showed **offline** on GitHub. Manual fix applied (same fragile bare-process scheme):
1. Reinstalled runner **v2.335.1** into `/workload/actions-runner` (curl+tar).
2. `config.sh --unattended --replace --name app-dev-gpu --labels self-hosted,linux,amd-gpu`
   with a freshly minted registration token; `RUNNER_ALLOW_RUNASROOT=1` (pod is root).
3. `nohup ./run.sh > runner.log 2>&1 &` — now "Listening for Jobs", GitHub status **online**.
4. Immediately picked up queued run `29433972050` (PR test-e2e-tui-cucumber @gpu).

**Still ephemeral** — this dies again on the next pod cycle. The durable Deployment below is the
real fix; this was a right-now unblock.

### 2026-07-16 (same day) — Moved install onto the personal PVC (semi-durable)

Migrated the runner off ephemeral `/workload` onto the persistent PVC so it survives pod cycles:
- **New location:** `/workload/fredrik-espinoza-amd-com/actions-runner` (PVC
  `pvc-386271a1-...`, 98G, mounted at `/workload/fredrik-espinoza-amd-com`).
- Copied the whole install **with identity** (`.runner`, `.credentials`,
  `.credentials_rsaparams`) via `cp -a`, so the registration is preserved — reconnected as
  `app-dev-gpu` with NO re-register and picked up the re-queued GPU job. Removed the ephemeral
  `/workload/actions-runner` copy.
- **After a future pod cycle, no token needed** — just restart `run.sh`. Helper on the PVC:
  `/workload/fredrik-espinoza-amd-com/actions-runner/start-runner.sh` (sets
  `RUNNER_ALLOW_RUNASROOT=1`, nohups run.sh, idempotent).

**Gotcha hit during migration:** `pkill -f Runner.Listener` / `pkill -f run.sh` inside
`kubectl exec` **matches the exec shell's own argv** and kills your own command (exit 137/143).
Kill by exact PID, or match a more specific string like `bin/Runner.Listener run`.

**Caveat — still not fully durable:** identity survives on the PVC, but the **process** does not
auto-start; a pod cycle still needs a manual `start-runner.sh`. The self-healing Deployment below
remains the real fix (auto-restart on pod start). This just removes the reinstall+re-register step.

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

## Next Steps — plan for the durable Deployment (updated 2026-07-16)

Current state after today: runner runs from the PVC, identity persists, but the **process**
still needs a manual `start-runner.sh` after a pod cycle, and it's still parasitic on the
vscode workspace pod. Goal of the remaining work: a **dedicated, self-healing Deployment** so a
pod restart auto-registers and starts the runner with zero touch, decoupled from the workspace.

### Phase 0 — Decisions (BLOCKING, user)
- 📋 **Credential:** GitHub App (recommended — `Administration: write` on `ROCm/rocm-cli` only,
  not tied to a user, revocable) vs fine-grained PAT (faster to prototype, user-scoped, expires).
- 📋 **Manifest home:** no runner manifests exist in rocm-cli today. Decide: separate infra repo
  vs apply-directly-and-store-in-progress-WIP. Confirm before creating (see Notes).

### Phase 1 — Credential as a Secret
- 📋 Create the chosen credential.
- 📋 Store in `app-dev` ns, `rocm-cli` namespace, as a k8s Secret
  (App: `app_id` + `installation_id` + private-key PEM; PAT: single token key).
- 📋 Least privilege — one repo only. Never bake the credential into the image.

### Phase 2 — Entrypoint script
- 📋 On start: mint a **registration token** from the credential (App → JWT → installation
  token → `POST .../actions/runners/registration-token`; PAT → same endpoint directly).
- 📋 `config.sh --unattended --replace --name app-dev-gpu --labels self-hosted,linux,amd-gpu`
  (`--replace` so it takes over the existing registration cleanly), then `run.sh`.
- 📋 **Trap SIGTERM → `config.sh remove`** for graceful deregister on pod delete/rollout, so
  GitHub doesn't accumulate offline ghosts.
- 📋 `RUNNER_ALLOW_RUNASROOT=1` (image runs as root). Reuse the runner install already on the
  PVC, or download v2.335.1 fresh in the entrypoint (decide — PVC reuse is faster cold-start).

### Phase 3 — Deployment manifest
- 📋 `replicas: 1`, spec from **Solution/Hardware** above (image
  `rocm/pytorch:rocm7.1.1_...`, `amd.com/gpu: 1`, cpu 1, mem 32Gi,
  `nodeSelector: kaiwo/worker: "true"`), `/dev/kfd` + `/dev/dri` access.
- 📋 Mount the credential Secret; mount the PVC if reusing the on-disk runner install.
- 📋 Deployment `restartPolicy: Always` (self-heal); entrypoint = Phase-2 script.

### Phase 4 — Cutover + verify
- 📋 Stop today's bare PVC runner (kill `bin/Runner.Listener run` by PID — NOT `pkill -f`,
  see migration gotcha) and `config.sh remove` it so the Deployment's `--replace` is clean.
- 📋 `kubectl apply` the Deployment; confirm it registers as `app-dev-gpu` with labels
  `[self-hosted, linux, amd-gpu]` and status **online**.
- 📋 Dispatch/queue a `@gpu` run and confirm the Deployment pod picks it up and passes.
- 📋 Kill-the-pod test: `kubectl delete pod` the runner → confirm it auto-restarts and
  re-registers with no manual step.
- 📋 THEN safe to shut down the vscode workspace pod (runner no longer depends on it).

## Notes
- Manifests are infra, likely NOT in the rocm-cli repo (no runner manifests exist there
  today) — probably a separate infra repo / applied directly. Confirm where such k8s
  manifests belong before creating.
- Relates to the manual-dispatch loop ([[ci-manual-e2e]]): dispatch is proven, but
  app-dev dispatches queue forever with no runner online — this WIP unblocks that.
