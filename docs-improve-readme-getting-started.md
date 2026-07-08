# WIP: Improve README Getting-Started Flow

**Stage:** 6-implementing
**Pipeline:** lightweight
**Branch:** docs/improve-readme-getting-started
**Last Updated:** 2026-06-30

**Token Usage:** in=0 out=0 cache_create=0 cache_read=0 calls=0

---

## Problem

The README's getting-started section was too sparse — it didn't explain PATH
setup, skipped the `rocm install sdk` prerequisite, and lacked guidance on
model IDs, gated models, and the `adopt` command's scope. New users hit
avoidable friction on first launch.

## Solution

Expand the README with:
- PATH setup instructions after binary download
- A "Getting started" section covering `rocm install sdk` and `rocm serve`
- Clarification that TUI behavior depends on whether ROCm is already installed
- Notes on full HuggingFace model IDs vs short aliases
- Gated model authentication guidance (HF_TOKEN / huggingface-cli login)
- Clarification that `adopt` only works with TheRock-based environments

## Implementation Steps

### Completed ✅
- ✅ Add `mkdir -p ~/.local/bin` and PATH setup instructions
- ✅ Rewrite TUI first-launch description (conditional on ROCm detection)
- ✅ Add "Getting started" section with `rocm install sdk` + `rocm serve`
- ✅ Document `adopt` scope limitation
- ✅ Add model ID and gated model notes to inference section

### In Progress ⏳

### Todo 📋
- 📋 Open PR and get review

## Next Steps

Open a PR for review.

## Worktree Context

**Worktree directory**: `~/Developer/rocm-cli-wt/docs/improve-readme-getting-started`
- Recreate with: `create_worktree.sh docs/improve-readme-getting-started`

## Work Log

### 2026-06-30

- Created WIP file from existing commit
- All README changes already committed (caaa3c7)
- Next: open PR
