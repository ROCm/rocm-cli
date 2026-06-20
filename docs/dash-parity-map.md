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
| doctor | B | covered (Phase 3) | `/doctor` slash → doctor overlay; tool `doctor` via seam | `slash_doctor_opens_overlay` / `read_only_tool_round_trips_to_json` |
| runtimes | B | covered (Phase 3) | `/runtimes` slash → runtime-manager overlay | `slash_runtimes_opens_overlay` |
| model | B | covered (Phase 3) | `/model` slash → `slash_tool` `rocm_command ["model"]` (off-thread) | `slash_model_raises_executor_request` |
| config | B | covered (Phase 3) | `/config` slash → config-manager overlay | `slash_config_opens_overlay` |
| logs | B | covered (Phase 3) | `/logs` slash → logs overlay; tool `service_logs` via seam | `slash_logs_opens_overlay` |
| gpu | B | covered (Phase 3) | `/gpu` slash → `ActiveTab::Hardware`; tool `gpu_snapshot` via seam | `slash_gpu_switches_to_hardware` |
| daemon | B | covered (Phase 3) | `/daemon` slash → `slash_tool` `rocm_command ["daemon","status"]` (off-thread) | `slash_daemon_raises_executor_request` |
| install | D | pending (Phase 4) | install overlay + mutating tool (approval-gated) | — |
| engine | D | pending (Phase 4) | engine-manager overlay + mutating tool (approval-gated) | — |
| serve | D | pending (Phase 4) | serve wizard + mutating tool (approval-gated) | — |
| services | D | pending (Phase 4) | services-manager overlay + mutating actions (approval-gated) | — |
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
