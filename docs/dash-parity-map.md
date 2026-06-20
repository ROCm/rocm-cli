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
| services | D | covered (Phase 4) | `/services [stop\|restart] <id>` slash → `stop_server` mutating tool → approval modal → `execute_approved`; bare `/services` is read-only | `slash_services_stop_raises_stop_server_request` |
| update | D | pending (Phase 5) | update overlay + mutating apply (approval-gated) | — |
| comfyui | D | pending (Phase 5) | ComfyUI serve/launch flow (approval-gated) | — |
| uninstall | D | pending (Phase 5) | uninstall flow (approval-gated) | — |
| setup | D | pending (Phase 5) | onboarding wizard (deterministic first-run) | — |
| automations | C | pending (Phase 6) | automations-manager overlay + run/approve actions | — |
| reviews | C | pending (Phase 6) | approval/review queue surface | — |
| approve | C | pending (Phase 6) | approval-queue accept action | — |
| reject | C | pending (Phase 6) | approval-queue reject action | — |
| edit | C | pending (Phase 6) | approval-queue edit action | — |
| permissions | E | pending (Phase 6) | permissions/full-access toggle surface | — |
| plan | F | pending (Phase 7) | `natural_language_plan` tool + plan review surface | — |
| provider | G | pending (Phase 8) | config & provider manager (provider switch) | — |
| chat | G | pending (Phase 8) | Chat tab + provider/Anthropic backend selection | — |
