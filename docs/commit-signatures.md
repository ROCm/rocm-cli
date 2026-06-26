# Commit Signatures and Sign-off

Every commit that lands in this repository must be:

1. **Cryptographically signed** — so authorship is verifiable and history
   cannot be silently forged or impersonated.
2. **Signed-off** — a DCO `Signed-off-by:` trailer, certifying you have the
   right to submit the change under the project license.

Both requirements are **enforced**, not advisory: local git hooks reject
non-conforming commits before they leave your machine, and a blocking CI gate
rejects them on every pull request.

## Setting Up Signing

You can sign with SSH (simplest if you already have an SSH key) or GPG.

### SSH signing

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true
```

### GPG signing

```bash
git config --global gpg.format openpgp        # the default
git config --global user.signingkey <your-gpg-key-id>
git config --global commit.gpgsign true
```

With `commit.gpgsign true` set, commits are signed automatically. Add the
sign-off trailer with `-s`:

```bash
git commit -s -m "Your message"
```

To add sign-off to a range of existing commits:

```bash
git rebase --signoff <base>
```

## Registering Your Key on GitHub (required for CI)

CI uses **strict** verification: it asks GitHub whether each commit's signature
is `verified`. For GitHub to report a commit as **Verified** you must:

- Add your signing key under **Settings → SSH and GPG keys** (an SSH *signing*
  key, or your GPG public key).
- Ensure the email in your committer identity is a **verified email** on your
  GitHub account.

If either is missing, the signature exists but GitHub will not mark it
"Verified", and the CI gate will fail.

## How Enforcement Works

### Local hooks (prek)

The hooks are defined in `.pre-commit-config.yaml`. Install them once:

```bash
prek install                # pre-commit
prek install -t pre-push    # pre-push
```

- **pre-commit**: `cargo xtask verify-commits --check-config` — a fast check
  that commit signing is enabled (`commit.gpgsign`, in any of git's truthy
  spellings) and `user.signingkey` is set, with remediation if not.
- **pre-push**: `cargo xtask verify-commits` — verifies every outgoing commit
  in `origin/main..HEAD` is signed (signature present and not bad) and
  signed-off. The base is fixed at `origin/main`, so keep it fetched
  (`git fetch origin main`); on a branch targeting a different base, run the
  check manually with `--base <ref>` for an accurate range.

### CI gate

The `commit-signatures` job in `.github/workflows/ci.yml` runs on pull requests
and in the merge queue:

```bash
cargo xtask verify-commits --base origin/<base-branch> --require-verified
```

`--require-verified` switches on strict mode (GitHub "Verified" via the `gh`
CLI). The job checks out with `fetch-depth: 0` so the base ref and all PR
commits are available. On a `merge_group` event it verifies the queued commits
against the queue's base (`merge_group.base_sha..head_sha`) instead of a PR
base — so the check still runs in the queue and can safely be made *required*
without stalling it. Mark it as a required check in branch protection / your
repository ruleset so it blocks merges.

### The xtask check

`cargo xtask verify-commits` is the single source of truth for the logic:

| Flag                | Meaning                                                           |
| ------------------- | ---------------------------------------------------------------- |
| `--base <ref>`      | Base ref; checks `<base>..HEAD`. Defaults to `GITHUB_BASE_REF` (as `origin/<branch>`) in CI, else `origin/main`. |
| `--require-verified`| Strict mode: require GitHub's `verification.verified` (needs `gh`). |
| `--check-config`    | Instead of checking commits, assert local signing is configured. |

An empty commit range is treated as success.

## Native GitHub Ruleset (Complementary)

GitHub can natively **require signed commits** via a repository ruleset. Enable
it as defense-in-depth alongside the in-repo check:

```bash
gh api -X POST /repos/<owner>/<repo>/rulesets \
  --input - <<'JSON'
{
  "name": "Require signed commits",
  "target": "branch",
  "enforcement": "active",
  "conditions": {
    "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] }
  },
  "rules": [
    { "type": "required_signatures" }
  ]
}
JSON
```

The native ruleset is useful but has **two gaps** that the `verify-commits`
xtask check exists to close:

1. **It does not apply to pull requests from forks.** Fork PRs can carry
   unsigned commits that the native rule will not block; the CI gate runs on
   every PR regardless of source.
2. **It cannot enforce DCO sign-off.** The native rule only checks signatures,
   not the `Signed-off-by` trailer.

Run both: the native ruleset for the merge-to-protected-branch path, and the
xtask check (local hooks + CI) for signatures *and* sign-off on every PR.
