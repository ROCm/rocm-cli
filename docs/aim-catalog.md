<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM catalog: freshness and updates

AIM entries live in [model_catalog.json](../crates/rocm-core/src/model_catalog.json)
alongside every other model recipe (`"loader": "aim"`). They are hand-maintained
and ship on the normal CLI release cadence, through the same PR review as any
other recipe. There is no network fetch of the catalog: a hosted, signed
recipe index is blocked on production signing keys (see
[release-trust.md](release-trust.md)) and is out of scope here.

## Freshness

Each AIM `engine_recipes` entry pins an image by `image_digest` and records
`generated_at_unix_ms`, the time a maintainer last confirmed the image and
digest still exist in the registry. `rocm model` prints a warning for any
engine recipe whose timestamp is older than 180 days
(`MODEL_RECIPE_STALE_AFTER_MS` in
[model_catalog.rs](../crates/rocm-core/src/model_catalog.rs)), because the
pinned image may have been withdrawn. Entries with no timestamp are never
flagged.

## Refreshing an entry

The owner of the AIM relationship (likely the AMD EAI team) does this; it does
not need a rocm-cli maintainer.

1. Watch for AMD Enterprise AI Suite releases. A version bump is the practical
   signal that AIM images were added or changed.
2. For each AIM entry, find the newest stable tag
   (`oras repo tags docker.io/amdenterpriseai/<image>`) and copy its digest
   (`oras resolve docker.io/amdenterpriseai/<image>:<tag>`) into `image_digest` (and `model_id_override` if the tag moved).
   Check `AIM_PROFILE_ID` in `extra_env` is still a valid profile.
3. Set `generated_at_unix_ms` to now (`date +%s%3N`).
4. Add entries for new AIM images the same way as the existing ones. The list
   of images is the AMD Enterprise AI AIM catalog
   (<https://enterprise-ai.docs.amd.com/en/latest/aims/catalog/models.html>).
   CPU-only (`aim-epyc-*`) images are left out, since rocm-cli never falls back
   to CPU, as are non-chat images (speech, video, bio) until the serve path
   supports their tasks.
5. Open a PR; `cargo test -p rocm-core model_recipe` validates the catalog.
6. Check each image's model page in the AIM catalog for the line "requires a
   Hugging Face authentication". For those, add the `HF_TOKEN` note to the
   entry's `notes` (the pattern used by the existing gated entries).
