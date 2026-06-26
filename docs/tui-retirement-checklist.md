# tui.rs retirement checklist — human deletion gate

The legacy chat-first assistant (`apps/rocm/src/tui.rs`) is **RETAINED, compiling
but unreferenced** by the interactive entrypoints. As of Phase 9, bare `rocm` and
interactive `rocm chat` route to the dash chat (`dash::run_chat` →
`ActiveTab::Chat`). The accept/retire gate is GO — see
[dash-parity-checklist.md](dash-parity-checklist.md) (all six capability buckets
ACCEPTED) and [dash-parity-map.md](dash-parity-map.md) (30/30 covered).

> **Do not execute these steps automatically.** This is the human go/no-go
> deletion procedure. Run it only after an explicit human decision to delete.

Until then, the module is kept reachable under `-D warnings` by a single
retention anchor in `apps/rocm/src/main.rs` (just below `mod tui;`):

```rust
#[allow(dead_code)]
const _RETAINED_TUI_ENTRY: fn(Option<String>) -> anyhow::Result<()> = tui::run;
```

This anchor references `tui::run`, which keeps its whole call graph reachable, so
no `dead_code` cascade fires into the `tui.rs`-only helpers in `main.rs`.

## Deletion steps (human, on go)

1. **Remove the module file.**
   ```bash
   git rm apps/rocm/src/tui.rs
   ```

2. **Remove the module declaration + retention anchor.** In
   `apps/rocm/src/main.rs`:
   - delete `mod tui;` (the `mod tui;` line near the top of the file);
   - delete the retention block: the `// tui.rs is RETAINED ...` comment, the
     `#[allow(dead_code)]` attribute, and the
     `const _RETAINED_TUI_ENTRY: fn(Option<String>) -> anyhow::Result<()> = tui::run;`
     line;
   - delete any other `#[allow(dead_code)]` attributes that were added solely for
     tui retention (none beyond the anchor were needed at Phase 9 — grep for
     `tui-retirement-checklist` comments to confirm: `grep -rn "tui-retirement-checklist" apps/rocm/src`).

3. **Prune now-dead helpers reachable ONLY from `tui.rs`.** Once the anchor is
   gone, `tui.rs`-only items in `main.rs` (and the `comfyui` `render_tui_*`
   helpers, etc.) become unreferenced and rustc will flag them. Do NOT guess the
   list by hand — let the compiler drive it:
   ```bash
   # Discover what tui.rs depended on from the crate root (pre-deletion survey):
   grep -oE "crate::[A-Za-z_][A-Za-z0-9_:]*|super::[A-Za-z_][A-Za-z0-9_]*" apps/rocm/src/tui.rs | sort -u

   # After steps 1-2, the build surfaces the exact dead items:
   cargo build --workspace --all-targets
   cargo clippy --workspace --all-targets -- -D warnings
   ```
   Prune each `dead_code`-flagged item that is reachable only from the deleted
   module. Known candidate buckets (verify each is truly unreferenced before
   removing — some are shared with the non-interactive chat render path and MUST
   stay):
   - `main.rs` chat-approval / sandbox-tool helpers used only by the TUI
     (e.g. `install_sdk_chat_approval_for_prompt`,
     `install_sdk_without_prefix_chat_approval`,
     `chat_install_folder_from_prompt`, `format_structured_tool_call`,
     `run_internal_sandbox_tool`, `render_chat_prompt_result_with_progress`,
     `tui_help_text`, `SandboxToolArg`) — **confirm** each has no remaining
     caller after deletion; keep any still used by `render_chat_prompt_text` /
     `render_chat_text` (the scriptable passthrough that stays).
   - `comfyui::render_tui_status`, `comfyui::render_tui_logs`,
     `comfyui::render_models_path` — TUI-only renderers.
   - The many `super::*` types/consts in `tui.rs` (managers, screens, menu
     choices, running-job machinery) defined in `main.rs` — remove those with no
     other caller.

   Iterate: delete a batch, rebuild, repeat until `cargo clippy -D warnings` is
   clean. Each pass narrows the set; the compiler is the source of truth.

4. **Move or drop `tui.rs`-specific tests.** `tui.rs` carries its own
   `#[cfg(test)] mod tests` (these are the known parallel env-race tests). They
   are deleted with the file in step 1. If any assertion covers behavior that
   now lives in the dash, port it to the corresponding dash test
   (`crates/rocm-dash-tui/src/app.rs` tests); otherwise drop it.

5. **Post-deletion verification (must all be green).**
   ```bash
   cargo build --workspace --all-targets \
     && cargo clippy --workspace --all-targets -- -D warnings \
     && cargo test --workspace --all-targets
   ```
   Run `cargo fmt --all` before committing. If `cargo test` is run in parallel,
   the historical `tui::tests` env-races disappear with the file; remaining
   flakes (ETXTBSY on freshly built binaries) clear under
   `cargo test --workspace --all-targets -- --test-threads=1`.

## Rollback

If any step cannot be made green, restore the retained state:
```bash
git checkout -- apps/rocm/src/tui.rs apps/rocm/src/main.rs
```
The retention anchor keeps the workspace compiling indefinitely, so there is no
time pressure to complete the deletion in one sitting.
