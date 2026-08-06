<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Release Trust

rocm-cli release packages are protected in two layers:

1. Every bundle has a `.sha256` checksum, and installers verify it before
   activation.
2. Release packaging can emit detached RSA/SHA-256 `.sig` files, and release
   and nightly CI now require those signatures for published assets.

## Signing Inputs

Release packaging is produced by `cargo xtask package <dist-name>`, a
cross-platform command that builds the bundle (`.tar.gz` on Unix, `.zip` on
Windows), writes the `.sha256` sidecar, and emits the detached `.sig` when a
signing key is configured.

Set one of these secrets or environment variables before release packaging:

```text
ROCM_CLI_SIGNING_PRIVATE_KEY_PEM
ROCM_CLI_SIGNING_PRIVATE_KEY_PATH
```

Stable release and nightly CI set `ROCM_CLI_REQUIRE_SIGNATURE=1`, so packaging
fails if no signing key is configured or if the `.sig` file is not produced.
Upload steps also fail if signature assets are missing instead of publishing an
unsigned release or nightly.

Release CI also runs the cross-platform dist verifier before upload. Release
jobs name both the versioned archive and the generic installer-facing archive,
so CI fails if an installer asset alias is missing:

```bash
python scripts/release_readiness.py \
  --dist dist \
  --require-signatures \
  --require-rocm-asset-names \
  --require-exact-assets \
  --asset rocm-cli-v0.1.0-linux-amd64.tar.gz \
  --asset rocm-cli-linux-amd64.tar.gz
```

The verifier confirms each named archive has the required rocm-cli bundle
files, confirms the `.sha256` sidecar matches the archive bytes and names the
archive being published, and confirms a non-empty `.sig` sidecar exists when
signatures are required. If a public key is provided with `--public-key`, it
also verifies the detached signature with `openssl`. Release CI also enables
`--require-rocm-asset-names`, which rejects branch-like, path-like, or
otherwise unsupported archive names before upload.

Release CI also enables `--require-exact-assets`. That rejects stale
publishable files in `dist`, including old `.tar.gz`/`.zip` archives and orphan
`.sha256` or `.sig` sidecars, unless they are explicitly named by `--asset` and
therefore verified before upload.

Release workflow uploads Linux and Windows assets into a draft GitHub release.
The draft is published only after both platform jobs pass their readiness
checks, so a failed Windows package does not publish a Linux-only stable
release by accident.

Nightly builds use the same readiness checks for both the timestamped nightly
archives and the generic installer-facing nightly aliases. Linux creates a
temporary staging release, Windows uploads to that staging release, and the
publish job downloads and re-verifies the combined Linux plus Windows asset set
before replacing the public `nightly` release. If Windows fails, the previous
public nightly remains available and the temporary staging release is removed.

The opt-in `@lifecycle` install-lifecycle E2E scenarios cover both path and PEM
signing inputs with generated local keys. Those keys are test material only;
they are not trust roots.

## Installer Verification

Installers always verify checksums. Signature verification runs when a public
key is supplied or `ROCM_CLI_REQUIRE_SIGNATURE=1` is set.

The detached `.sig` sidecar is the raw RSASSA-PKCS#1 v1.5 signature over the
archive's SHA-256 digest, verified against the SubjectPublicKeyInfo
(`-----BEGIN PUBLIC KEY-----`) public key. The Linux installer verifies it with
`openssl`. The Windows installer (`install.ps1`) verifies it with the built-in
.NET RSA APIs and has no runtime dependency on `openssl`: it parses the public
key and calls `RSA.VerifyData` with SHA-256 and PKCS#1 v1.5 padding. This works
on both Windows PowerShell 5.1 (.NET Framework 4.6+, which the
`RSA.VerifyData` overload requires) and PowerShell 7+ (.NET).
`certutil`/`certreq` are not used, because they only verify certificate-backed
CMS/Authenticode blobs, not a raw detached signature checked against a bare
public key.

Linux:

```bash
ROCM_CLI_REQUIRE_SIGNATURE=1 \
ROCM_CLI_SIGNING_PUBLIC_KEY_PATH=/path/to/rocm-cli-signing-public-key.pem \
sh ./install.sh release
```

Windows:

```powershell
$env:ROCM_CLI_REQUIRE_SIGNATURE = "1"
.\install.ps1 release -SigningPublicKeyPath C:\path\to\rocm-cli-signing-public-key.pem
```

The `@lifecycle` install-lifecycle E2E scenarios cover public-key path and PEM
installer inputs, bad checksums, bad detached signatures, missing required
`.sig` sidecars, and required-signature mode without a public key. The Windows
scenarios additionally run a full signed install over a loopback HTTP download,
proving `install.ps1` verifies with native .NET crypto. All of those checks
happen before activation. Run the current host's set with
`E2E_INCLUDE_LIFECYCLE=1 E2E_ONLY_LIFECYCLE=1 cargo xtask e2e` (see
`docs/testing.md`).

## PyPI Wheel Channel

`pip install rocm-cli` distributes the same binaries through a second channel
with a *different trust root*. PyPI stores only the uploaded distribution
files, so it cannot carry the detached `.sig` sidecar described above: there is
nowhere for a wheel consumer to fetch `<wheel>.sig` from, and `pip` would not
check it. Do not describe the wheel channel as detached-signature verified.

What the wheel channel does provide:

- **Trusted Publishing (OIDC).** The `publish-pypi` job in
  `.github/workflows/release.yml` authenticates with a short-lived OpenID
  Connect token minted for this repository's release workflow and its `pypi`
  environment, so PyPI accepts `rocm-cli` uploads only from that identity.
  There is no long-lived upload token to leak or rotate.
- **PEP 740 attestations.** `pypa/gh-action-pypi-publish` publishes a signed
  attestation with each wheel, binding the file digest to the workflow that
  produced it. Attestations stay enabled; the publish steps do not opt out.
  Note what this is and is not: `pip` does not verify attestations at install
  time and offers no flag to require them, so for a plain `pip install` this is
  an auditable provenance record rather than an install-time gate. Fetch it
  from PyPI's integrity API,
  `https://pypi.org/integrity/rocm-cli/<version>/<filename>/provenance`.
- **In-pipeline verification.** Each wheel is built in the same job that
  packaged and signed the release archive, from the same `target/release`
  binaries, smoke-tested by installing it into a throwaway virtualenv, and
  re-checked against its `.sha256` sidecar in the publish job before upload.
  The sidecar is written next to the wheel on the builder runner and travels in
  the same artifact, so that check catches a file corrupted or truncated in
  artifact storage transit. It cannot detect a compromised builder that wrote a
  consistent wheel and sidecar together; the control against that is the
  attestation's binding to the workflow identity, not the checksum.

The GitHub release assets remain the signature-bearing artifacts. The `.tar.gz`
and `.zip` bundles and their detached `.sig` files are the only place the RSA
release signature is published, and they are what the installers verify — see
[Installer Verification](#installer-verification) for the verification inputs
and commands. `cargo xtask package` copies the release binaries verbatim, so
the `rocm` and `rocmd` payload in the wheel is byte-identical to the copy
inside the signed archive for the same tag; anyone who needs the detached
signature as their trust root should install from the release with `install.sh`
or `install.ps1` and verify there.

Publication is gated on the repository variable `ROCM_CLI_PUBLISH_PYPI`, which
must be set to `1` or `true`; every other value, including unset, leaves the
job dormant. While it is dormant, release CI still builds, smoke-tests, and
retains the wheels as workflow artifacts, but nothing is uploaded to any index.
The wheel build and its smoke test do run on every release, so a wheel
regression fails the release job even while publication is off — the uploaded
GitHub asset set is unchanged, the pipeline is not.

### Version mapping is one-way and unrepeatable

`scripts/build_wheel.py` maps the git tag onto a PEP 440 version: `vX.Y.Z` to
`X.Y.Z`, `-alpha.N` and `-experimental.N` to `X.Y.ZaN`, `-beta.N` to `X.Y.ZbN`,
`-rc.N` to `X.Y.ZrcN`. Any other tag shape, and any version carrying a local,
dev, post, or epoch segment, is refused rather than published. The mapped
release segment must also equal `[workspace.package] version` in the root
`Cargo.toml`, so a tag can never publish a wheel whose binaries report a
different version.

PyPI never lets a version be re-uploaded, even after deletion. Two consequences
bind the tagging policy: a botched release must be yanked and superseded by a
new version rather than replaced, and because `-alpha.N` and `-experimental.N`
both map to `X.Y.ZaN`, only one of those two spellings may be used for a given
serial. Tagging both `v1.2.3-experimental.1` and `v1.2.3-alpha.1` makes the
second release fail at upload.

## WSL ROCDXG Package Verification

`scripts/wsl_setup_rocdxg.sh` does not bake in a production checksum for the
ROCDXG `.deb`. Operators can require verification by providing the expected
digest:

```bash
ROCDXG_SHA256=<64-hex-sha256> bash scripts/wsl_setup_rocdxg.sh
```

When `ROCDXG_SHA256` is set, the script verifies the downloaded `.deb` before
`apt install` and fails on malformed or mismatched digests.

## Runtime Metadata Verification

The Rust metadata cache can verify detached metadata signatures when a metadata
public key is configured:

```text
ROCM_CLI_METADATA_PUBLIC_KEY_PATH
ROCM_CLI_METADATA_PUBLIC_KEY_PEM
ROCM_CLI_REQUIRE_METADATA_SIGNATURE
```

When verification is active, rocm-cli fetches `<metadata-url>.sig`, verifies the
cached body with `openssl dgst -sha256 -verify`, and records verification
details in the cache metadata. If `ROCM_CLI_REQUIRE_METADATA_SIGNATURE=1` is set
without a public key, or if the sidecar signature is missing or invalid, the
metadata fetch fails and does not use unsigned data.

Developer tests cover this path with generated local RSA keys and tampered
payload rejection; see `docs/testing.md`.

## Model Recipe Index Verification

rocm-cli can read an externally configured model recipe index only when it is
signed with a detached signature and a local public key is provided:

```text
ROCM_CLI_MODEL_RECIPE_INDEX_PATH
ROCM_CLI_MODEL_RECIPE_INDEX_SIGNATURE_PATH
ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH
ROCM_CLI_REQUIRE_MODEL_RECIPE_SIGNATURE
```

If `ROCM_CLI_MODEL_RECIPE_INDEX_PATH` is set, rocm-cli verifies the index with
`openssl dgst -sha256 -verify` before using it. The default signature path is
`<index>.sig`. If verification fails, model recipe loading fails loudly instead
of using the built-in registry for a configured-but-invalid external index.
Without an external index, rocm-cli uses the built-in offline recipe registry.

Developer tests cover signed-index loading with generated local RSA keys and
tampered payload rejection; see `docs/testing.md`.

Signed recipe artifact records may also declare an optional `source_policy`
object. Supported policy names are:

- `direct_https_sha256`
- `huggingface_public`
- `huggingface_authenticated`
- `manual_only`

When present, rocm-cli validates the policy while loading the signed index,
shows it in `/model` as a download rule, and enforces it during restricted
artifact prefetch. A declared policy can require HTTPS, required source hosts,
sha256 and size metadata, Hugging Face host scoping, Hugging Face token
approval, or manual-only blocking. Existing signed indexes without
`source_policy` keep the older explicit-review behavior.

## Remaining Owner Step

The repo still needs a real project-owned public signing key and matching
private release secret, plus the public metadata and model recipe index keys,
signed sidecar publication, and hosted production recipe/source-policy
documents. Do not generate those production keys inside CI. Once published, the
installers, metadata cache, and model recipe index can make release-channel
signature verification mandatory by default with those keys as trust roots.

When those owner-controlled inputs exist, release operators can add
`--require-production-trust` or set `ROCM_CLI_REQUIRE_PRODUCTION_TRUST=1` so the
dist verifier fails unless the release signing public key, runtime metadata
public key, and signed model recipe index inputs are explicitly configured.
