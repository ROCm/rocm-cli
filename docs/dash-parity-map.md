# Dash parity map

Tracks the 30 legacy `rocm chat` slash commands as they are re-homed into the
dash TUI, so `tui.rs` can be retired without regression. One row per legacy
command. `Status` is per phase; `Dash mechanism` is the concrete surface (tab /
overlay / modal / tool + slash command); `Evidence` points at the test that
locks the behavior.

Groups: **A** nav/session · **B** read-only · **C** approvals/automations ·
**D** mutating ops · **E** permissions · **F** planning · **G** provider/chat.

| Command | Group | Status | Dash mechanism | Evidence |
|---------|-------|--------|----------------|----------|
| home | A | covered (Phase 3) | `/home` slash → `ActiveTab::Overview` | `slash_home_switches_to_overview` |
| help | A | covered (Phase 3) | `/help` slash → `Modal::Help` | `slash_help_opens_help_modal` |
| ? | A | covered (Phase 3) | `/?` slash → `Modal::Help` | `slash_question_mark_opens_help_modal` |
| clear | A | covered (Phase 3) | `/clear` slash → empties `chat` transcript | `slash_clear_empties_transcript` |
| quit | A | covered (Phase 3) | `/quit` slash → `should_quit` (event loop breaks) | `slash_quit_sets_should_quit` |
| exit | A | covered (Phase 3) | `/exit` slash → `should_quit` (event loop breaks) | `slash_exit_sets_should_quit` |
| doctor | B | covered (Phase 3) | `/doctor` slash → opens the doctor overlay (`doctor_manager`); the read-only `doctor` tool is covered separately via the LLM tool-call seam | `slash_doctor_opens_overlay` / `read_only_tool_round_trips_to_json` |
| runtimes | B | covered (Phase 3) | `/runtimes` slash → runtime-manager overlay | `slash_runtimes_opens_overlay` |
| model | B | covered (Phase 3) | `/model` slash → `slash_tool` `rocm_command ["model"]` (off-thread) | `slash_model_raises_executor_request` |
| config | B | covered (Phase 3) | `/config` slash → config-manager overlay | `slash_config_opens_overlay` |
| logs | B | covered (Phase 3) | `/logs` slash → opens the logs overlay (`logs_view`); the read-only `service_logs` tool is covered separately via the LLM tool-call seam | `slash_logs_opens_overlay` / `read_only_tool_round_trips_to_json` |
| gpu | B | covered (Phase 3) | `/gpu` slash → `ActiveTab::Hardware`; tool `gpu_snapshot` via seam | `slash_gpu_switches_to_hardware` |
| daemon | B | covered (Phase 3) | `/daemon` slash → `slash_tool` `rocm_command ["daemon","status"]` (off-thread) | `slash_daemon_raises_executor_request` |
| install | D | covered (Phase 4) | `/install` slash → `install_sdk` mutating tool → approval modal → `execute_approved` (captured subprocess); also LLM tool-call seam | `slash_install_raises_install_sdk_request` / `approve_path_runs_execute_approved` |
| engine | D | covered (Phase 4) | `/engine <name>` slash → `install_engine` mutating tool → approval modal → `execute_approved`; also LLM tool-call seam | `slash_engine_raises_install_engine_request` |
| serve | D | covered (Phase 4) | `/serve <model>` slash (loopback host) → `launch_server` mutating tool → approval modal → `execute_approved`; also LLM tool-call seam | `slash_serve_raises_launch_server_request` / `seam_execute_approved_rejects_unsafe_call_via_validator` |
| services | D | covered (Phase 4) | `/services stop <id>` slash → `stop_server` mutating tool → approval modal → `execute_approved`; `restart` is guided (not yet wired through the chat seam — points to stop + `/serve`); bare `/services` is read-only | `slash_services_stop_raises_stop_server_request` / `slash_services_restart_is_guided_not_stop` |
| update | D | covered (Phase 5) | `/update` slash → `slash_tool` `rocm_command ["update"]` (read-only report); `/update --apply` → `["update","--apply"]` → approval modal | `slash_update_is_read_only_report` / `slash_update_apply_is_mutating` / `lifecycle_read_mutate_split_is_honest` |
| comfyui | D | covered (Phase 5) | `/comfyui` slash → `rocm_command ["comfyui","status"]` (read; status/logs); `install`/`start`/`stop` → approval modal | `slash_comfyui_bare_is_status` / `slash_comfyui_start_is_mutating` / `slash_comfyui_logs_is_read_only` |
| uninstall | D | covered (Phase 5) | `/uninstall` slash → `rocm_command ["uninstall","--dry-run"]` (SAFE read-only default); `/uninstall --apply` → `["uninstall"]` → approval modal (bin auto-adds `--yes`) | `slash_uninstall_defaults_to_dry_run` / `slash_uninstall_apply_is_real` |
| setup | D | covered (Phase 5) | `/setup` slash → `rocm_command ["setup","status"]` (read); `/setup reset` → `["setup","reset"]` → approval modal; unsupported subs guided | `slash_setup_bare_is_status` / `slash_setup_reset_is_mutating` / `setup_status_is_read_only` / `setup_reset_requires_approval` |
| automations | C | covered (Phase 6) | `/automations` (or `list`) slash → `rocm_command ["automations","list"]` (read); `/automations enable <watcher> [--mode <m>]` → `watcher_enable` mutating tool → approval modal; `/automations disable <watcher>` → `watcher_disable` → approval modal; also LLM tool-call seam | `slash_automations_bare_lists` / `slash_automations_enable_raises_watcher_enable` / `slash_automations_disable_raises_watcher_disable` / `watcher_validator_rejects_unknown_and_invalid_mode` |
| reviews | C | covered (Phase 6) | `/reviews` slash → `rocm_command ["automations","list"]` (read, pending reviews); `/reviews <id>` → `proposal_action {action:show}` (read detail) | `slash_reviews_bare_lists` / `slash_reviews_id_shows_proposal` / `proposal_action_show_is_read_only` |
| approve | C | covered (Phase 6) | `/approve <id>` slash → `proposal_action {action:approve}` → approval modal → `execute_approved` → `update_automation_proposal_status(..,"approved")`; bare `/approve` hints | `slash_approve_id_raises_proposal_approve` / `proposal_action_approve_updates_status` / `proposal_action_approve_requires_approval` |
| reject | C | covered (Phase 6) | `/reject <id>` slash → `proposal_action {action:reject}` → approval modal → `execute_approved` → `update_automation_proposal_status(..,"rejected")`; bare `/reject` hints | `slash_reject_id_raises_proposal_reject` / `proposal_action_reject_updates_status` |
| edit | C | covered (Phase 6) | `/edit <id>` slash → `proposal_action {action:show}` (review detail; content editing unsupported by the bin) + a one-line note directing to /approve or /reject; bare `/edit` hints | `slash_edit_id_shows_proposal_with_note` |
| permissions | E | covered (Phase 6) | `/permissions` (or `status`) slash → `rocm_command ["config","show"]` (read current mode); `/permissions full-access` → `["config","set-permissions","full_access"]` → approval modal (escalation gated); `/permissions ask` → `["config","set-permissions","ask"]` → approval modal | `slash_permissions_bare_is_config_show` / `slash_permissions_full_access_is_mutating` / `slash_permissions_ask_is_mutating` / `config_set_permissions_classifies_as_approval` / `config_set_permissions_sets_mode` |
| plan | F | pending (Phase 7) | `natural_language_plan` tool + plan review surface | — |
| provider | G | pending (Phase 8) | config & provider manager (provider switch) | — |
| chat | G | pending (Phase 8) | Chat tab + provider/Anthropic backend selection | — |
