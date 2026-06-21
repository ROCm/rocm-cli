# Dash parity checklist — accept/retire gate

ROADMAP §1, step 1: the accept/retire gate for retiring the legacy `tui.rs`
assistant. One row per legacy **capability bucket** (not per command — the
per-command map is [dash-parity-map.md](dash-parity-map.md)). Every bucket must
be **ACCEPTED** with a concrete test/artifact pointer before the human deletion
gate ([tui-retirement-checklist.md](tui-retirement-checklist.md)) is opened.

**Zero unaccepted rows.** As of Phase 9, bare `rocm` and interactive `rocm chat`
route to the dash chat (`dash::run_chat` → `ActiveTab::Chat`).

| # | Capability bucket | Legacy surface | Dash mechanism | Status | Evidence |
|---|-------------------|----------------|----------------|--------|----------|
| 1 | Read-only inspection (doctor, engines, services, gpu, model, config, logs, daemon, runtimes, paths) | `tui.rs` read-only tool-calls + slash commands | Phase 3 read-only `rocm_command` tools via the bin-side seam (`dash_seam::BinToolExecutor`) + nav/overlay slash commands | ACCEPTED | `read_only_tool_round_trips_to_json` (agent); `slash_doctor_opens_overlay`, `slash_runtimes_opens_overlay`, `slash_gpu_switches_to_hardware`, `slash_daemon_raises_executor_request` (app); map rows B |
| 2 | Mutating ops + in-chat approval (install, engine, serve, services, update, comfyui, uninstall, setup) | `tui.rs` `ChatToolApprovalRequest` + `execute_approved` | Phase 4/5 mutating tools → approval modal → `execute_approved` (validator-gated, captured subprocess) | ACCEPTED | `approve_path_runs_execute_approved`, `slash_install_raises_install_sdk_request`, `slash_serve_raises_launch_server_request`, `slash_update_apply_is_mutating`, `slash_uninstall_defaults_to_dry_run` (app); `seam_execute_approved_rejects_unsafe_call_via_validator` (seam); map rows D |
| 3 | Automations / reviews / permissions | `tui.rs` automations + proposal + permissions flows | Phase 6 `watcher_enable/disable`, `proposal_action {show,approve,reject}`, `config set-permissions` → approval modal (escalation gated) | ACCEPTED | `slash_automations_enable_raises_watcher_enable`, `slash_approve_id_raises_proposal_approve`, `proposal_action_approve_updates_status`, `slash_permissions_full_access_is_mutating`, `config_set_permissions_classifies_as_approval` (app); map rows C, E |
| 4 | Natural-language `/plan` (Ask → Plan → Review → Run) | `tui.rs` `FreeformPlanAction` planner | Phase 7 `/plan` edge → read-only `natural_language_plan` tool → `PlanReady` → `on_plan_ready` (complete mutating action hands off to the Phase-4 approval modal; placeholder/non-mutating stays plan-only) | ACCEPTED | `plan_request_set_by_slash`, `on_plan_ready_complete_mutating_hands_off_to_approval`, `on_plan_ready_placeholder_stays_plan_only` (app); `natural_language_plan_returns_structured_mutating_action` (bin) |
| 5 | 30 slash commands | `tui.rs` slash command set | Phases 3-8 dash slash router (nav / overlay / slash-tool / approval) | ACCEPTED | [dash-parity-map.md](dash-parity-map.md) — 30 rows, all `covered`, 0 pending-status |
| 6 | Providers: local + openai + anthropic | `tui.rs` provider switching | Phase 8 `/provider` edge → `build_chat_agent` rebuilds the live agent (local auto-detect / `RigAgentClient` / `AnthropicAgentClient`); keys env-first then secure store, never argv; same ROCm read+mutating tools on every backend | ACCEPTED | `slash_provider_switches_backend`, `build_chat_agent_anthropic_with_key`, `build_chat_agent_anthropic_without_key_is_none` (app); `anthropic_backend_constructs_and_is_agentclient`, `anthropic_requires_key`, `all_backends_register_same_rocm_tools` (agent) |

All six capability buckets are ACCEPTED. The accept/retire gate is **GO** for the
human deletion steps in [tui-retirement-checklist.md](tui-retirement-checklist.md).
