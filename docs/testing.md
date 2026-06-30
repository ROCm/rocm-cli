<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Testing

This page collects developer verification commands. The README stays focused on quick-start usage.

## Local Verification

Run the Rust test suite:

```bash
cargo test --workspace --all-targets
```

Run clippy with warnings as errors:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Run the cross-platform smoke test:

```bash
python scripts/smoke_local.py
```

If the workspace is already built:

```bash
python scripts/smoke_local.py --skip-build
```

The smoke path is the cross-platform local no-fallback acceptance surface. It
uses an isolated config/data/cache root, does not install TheRock wheels, does
not require a managed runtime, and verifies:

- first-run examine and engine inventory state
- telemetry-off config behavior
- GPU-required recipe planning for `tiny-gpt2`
- exact-runtime rejection for `rocm engines install` before any runtime exists
- direct llama.cpp external-runtime probing with a fake local `llama-server`
- explicit PyTorch `tiny-gpt2` GPU-required recipe resolution
- direct adapter and `rocm serve` CPU policy rejection
- GPU-required paths fail loudly instead of silently falling back
- no first-run pip cache or runtime registry is created

Build the platform-native release artifact:

```bash
python scripts/build_single_exe_release.py standalone
```

On Windows this writes `.rocm-work/standalone-release/rocm.exe`; on Linux it
writes `.rocm-work/standalone-release/rocm`. The artifact is the rocm-cli binary
itself, not a self-extracting launcher and not a model bundle. Running it with
no arguments opens normal rocm-cli; if setup is not complete, the first-time
setup wizard appears automatically.

rocm-cli ships native per-OS binaries; there is no cross-OS universal binary.
Build and test the binary natively on each supported target (native Windows,
Ubuntu/Fedora on WSL, and Linux on server/bare-metal):

```bash
cargo build --workspace --release
cargo test --workspace
```

On WSL/Linux, build and run the native Linux binary directly (no APE launcher
prefix is required). The examine output on WSL must include `os: linux` and
`wsl: true`, and must not include `os: windows`.

Use isolated `ROCM_CLI_CONFIG_DIR`, `ROCM_CLI_DATA_DIR`, and
`ROCM_CLI_CACHE_DIR` roots for smoke tests, then delete those roots after the
test so the user's real `.rocm` state stays clean.

Focused examine guidance coverage:

```bash
cargo test -p rocm-core examine_render
cargo test -p rocm-core managed_sdk_probe
cargo test -p rocm --bin rocm examine_runtime_state_reports_ambiguous_default_runtime_id
cargo test -p rocm --bin rocm engine_runtime_selection_rejects_ambiguous_default_runtime_id
cargo test -p rocm --bin rocm first_run_shows_dedicated_setup_screen_when_rocm_is_not_installed
cargo test -p rocm --bin rocm onboarding_enter_requests_rocm_install_with_folder_prefix
cargo test -p rocm --bin rocm setup_subcommands_do_not_expose_old_runtime_engine_wizard
cargo test -p rocm --bin rocm permissions_
```

Focused TUI command navigability coverage:

```bash
cargo test -p rocm --bin rocm tui::tests:: -- --nocapture
cargo test -p rocm tui::tests::advertised_slash_commands_ -- --nocapture
cargo test -p rocm tui::tests::slash_subcommands_stay_on_navigable_surfaces_without_transcript_dumps -- --nocapture
cargo test -p rocm tui::tests::slash_command_variants_stay_navigable_without_transcript_dumps -- --nocapture
cargo test -p rocm tui::tests::slash_actions_stream_inside_active_surface_without_transcript_dump -- --nocapture
cargo test -p rocm tui::tests::slash_first_views_hide_backend_jargon -- --nocapture
cargo test -p rocm tui::tests::install_sdk_bad_format_uses_friendly_labels_without_backend_jargon -- --nocapture
cargo test -p rocm tui::tests::onboarding_ctrl_c_during_install_confirms_before_cancelling -- --nocapture
cargo test -p rocm tui::tests::automations_review_actions_without_review_show_body_guidance -- --nocapture
cargo test -p rocm tui::tests::examine_background_completion_updates_report_without_transcript -- --nocapture
cargo test -p rocm tui::tests::hidden_examine_refresh_does_not_block_next_prompt_commands -- --nocapture
cargo test -p rocm tui::tests::examine_command_does_not_steal_running_workflow_screens -- --nocapture
cargo test -p rocm tui::tests::examine_command_does_not_hide_pending_approval -- --nocapture
cargo test -p rocm tui::tests::daemon_status_uses_background_helper_language -- --nocapture
cargo test -p rocm tui::tests::automations_bad_watcher_and_mode_stay_screen_local -- --nocapture
cargo test -p rocm tui::tests::services_manager_stop_requests_rocmd_approval_without_transcript -- --nocapture
cargo test -p rocm tui::tests::services_command_restart_requests_rocmd_approval_without_transcript -- --nocapture
cargo test -p rocm --bin rocm services_is_structured_not_freeform -- --nocapture
cargo test -p rocm --bin rocm top_level_cli_commands_are_not_treated_as_freeform -- --nocapture
cargo test -p rocm --bin rocm render_model_registry_text_lists_builtin_recipes -- --nocapture
cargo test -p rocm --bin rocm plain_command_routing_distinguishes_commands_from_requests -- --nocapture
cargo test -p rocm --bin rocm plain_structured_commands_route_to_navigable_tui_before_plan -- --nocapture
cargo test -p rocm --bin rocm render_services_text_separates_ready_starting_and_past_attempts -- --nocapture
cargo test -p rocm --bin rocm service_actions_require_yes_and_render_sandbox_result -- --nocapture
cargo test -p rocm tui::tests::runtimes_import_without_args_opens_guided_form -- --nocapture
cargo test -p rocm tui::tests::approval_card_is_arrow_key_navigable_and_scrollable -- --nocapture
cargo test -p rocm tui::tests::typed_serve_prefills_wizard_and_start_requests_approval_without_prompt_editing -- --nocapture
cargo test -p rocm tui::tests::explicit_foreground_serve_prefills_wizard_without_launching -- --nocapture
cargo test -p rocm tui::tests::tui_help_teaches_engine_not_engines -- --nocapture
```

Self-hosted GPU CI smoke is intentionally non-mutating. The MI300X job builds
the workspace, runs `rocm examine`, then runs `detect` and `capabilities` for
all first-party engine adapters: PyTorch, llama.cpp, ATOM, vLLM, and SGLang.
Live serving acceptance remains separate because it needs engine-specific
runtime installs, model artifacts, and supported upstream GPU targets.

WSL live examine sanity after building with a Linux target directory:

```bash
export CARGO_TARGET_DIR=/home/user/.cache/rocm-cli-target
cargo build --workspace
rocm examine
```

Expected WSL fields when a managed TheRock runtime is registered and ROCDXG is
ready:

- `detected_gfx_target: gfxNNNN` from sysfs, PATH ROCm tools, or the managed
  TheRock SDK tool path recorded in the runtime registry
- `compatible_therock_family: ...`
- `active_runtime_status: ready`, `unset`, `missing_manifest`, or
  `ambiguous_runtime_id`
- if ambiguous, `active_runtime_matches` plus the exact
  `rocm runtimes activate <runtime_key>` action

Ambiguous runtime selectors should also fail before engine install or serve
launches:

```bash
rocm engines install pytorch
rocm serve target/models/stories260K.gguf --engine llama.cpp --managed
```

On a host where `default_runtime_id` matches multiple registered runtime keys,
both commands should report the matching runtime keys and ask for
`rocm runtimes activate <runtime_key>` or an explicit runtime key.

Focused TheRock metadata cache coverage:

```bash
cargo test -p rocm --bin rocm metadata_cache
cargo test -p rocm --bin rocm metadata_signature_verification_accepts_generated_key_and_rejects_tamper
cargo test -p rocm --bin rocm http_header_value
rocm install sdk --channel release --format wheel --dry-run
```

## TheRock SDK Install Test

The live SDK acceptance test creates an isolated test root under `target/`, creates a local bootstrap Python venv, runs:

```bash
rocm install sdk --channel release --format wheel
```

Then it verifies:

- runtime manifest metadata
- localized pip cache inside the selected ROCm runtime folder at
  `<install-root>/pip-cache`, including both generated managed folders and
  explicit `--prefix` folders
- the installer does not pre-create that pip cache during dry-run or setup;
  pip creates it inside the ROCm folder when packages are downloaded
- a single TheRock-index pip install plan for pinned `rocm[libraries,devel]`,
  `torch`, `torchvision`, and `torchaudio` versions
- package selection uses the newest exact ROCm build suffix common to the SDK
  package and the PyTorch stack for the current Python/platform wheel tags
- `python -m rocm_sdk version`
- `python -m rocm_sdk targets`
- runtime-only TheRock wheel discovery through
  `rocm_sdk.find_libraries("amdhip64", "hipblas")`
- llama.cpp adapter discovery of the managed TheRock HIP runtime environment

Run the full live test with auto-detected GPU family:

```bash
python scripts/therock_sdk_install_test.py --root .rocm-work/tests/therock-sdk-install --fresh
```

Run only the resolver/harness path without downloading SDK wheels:

```bash
python scripts/therock_sdk_install_test.py --root .rocm-work/tests/therock-sdk-install --dry-run --check-windows-tools
```

Verify the first-time-setup style explicit folder path without downloading SDK
wheels:

```bash
python scripts/therock_sdk_install_test.py --root .rocm-work/tests/therock-sdk-install --dry-run --prefix .rocm-work/tests/rocm-folder
```

Use a deterministic family only for developer tests that need fixed package
selection:

```bash
python scripts/therock_sdk_install_test.py --root .rocm-work/tests/therock-sdk-install --fresh --family gfx120X-all
```

## PyTorch TheRock GPU Acceptance

This opt-in test verifies that PyTorch can run a tiny model on your AMD GPU
using the TheRock ROCm wheels managed by rocm-cli. It does not use an external
Python environment and it does not fall back to CPU.

Simple path:

```bash
rocm
```

In the first-time setup screen, choose the install folder and choose
`Install ROCm`. After setup finishes, install the managed PyTorch engine:

```bash
rocm engines install pytorch
```

The engine installer should use the Python version from the selected managed
TheRock runtime, pin `torch`, `torchvision`, and `torchaudio` to the exact
versions already installed in that runtime, and keep its pip cache under
rocm-cli's cache directory. It must not ask pip to solve an unbounded TheRock
torch stack.

Then run the GPU acceptance script:

```bash
python scripts/pytorch_therock_gpu_test.py
```

The script uses the active rocm-cli runtime key. If no runtime is active, run
`rocm runtimes activate <runtime_key>` first or pass `--runtime-id
<runtime_key>`. A broad runtime id is only accepted when it matches one saved
runtime unambiguously.

Developer direct path:

```bash
rocm install sdk --channel release --format wheel
rocm runtimes activate <runtime_key from install output>
rocm engines install pytorch
python scripts/pytorch_therock_gpu_test.py
```

The script starts a local PyTorch test server in AMD GPU mode, loads
`hf-internal-testing/tiny-random-gpt2`, sends a tiny prompt, verifies the
service reports AMD GPU execution, and checks the worker process for loaded HIP
modules from the managed PyTorch folder. PyTorch reports ROCm GPUs through its
`cuda` device API; seeing `device: cuda` in the test output is expected on AMD
ROCm. Model downloads are localized under `target/test-cache/huggingface`
unless Hugging Face cache variables are already set. When
`ROCM_CLI_CACHE_DIR` or `ROCM_CLI_DATA_DIR` are set for isolated acceptance
tests, the script keeps its Hugging Face cache, state file, and logs under
those rocm-cli directories. If `CARGO_TARGET_DIR` is set, the default engine
binary is resolved from that target directory. If the managed PyTorch folder is
missing, the script stops with the exact
`rocm engines install pytorch` prerequisite instead of creating a surprise
install or using CPU.

Offline selector sanity check:

```bash
python scripts/pytorch_therock_gpu_test.py --self-test
```

## ComfyUI TheRock GPU Acceptance

This opt-in test verifies that rocm-cli can install or reuse its managed
ComfyUI app, start it through the active TheRock ROCm runtime, reach the local
ComfyUI HTTP endpoint, report status/logs, and stop the launched process
without CPU fallback.

Run the offline harness self-test:

```bash
python scripts/comfyui_therock_gpu_test.py --self-test
```

Run the live test after first-time setup has installed and selected a managed
ROCm runtime:

```bash
python scripts/comfyui_therock_gpu_test.py
```

Run the live test with isolated rocm-cli state while reusing only the existing
managed TheRock runtime records:

```bash
python scripts/comfyui_therock_gpu_test.py --temp-state --copy-runtime-state-from %USERPROFILE%\.rocm
```

That command copies `config.json` and `runtimes/` into a temporary
`ROCM_CLI_CONFIG_DIR`, `ROCM_CLI_DATA_DIR`, and `ROCM_CLI_CACHE_DIR`, installs
ComfyUI in that temporary app state, then removes the temporary state after it
stops the process it started.

To reuse an already installed ComfyUI app without reinstalling dependencies:

```bash
python scripts/comfyui_therock_gpu_test.py --skip-install
```

To run a real image-generation smoke test, add `--generate-cat`. The harness
places a safetensors checkpoint in ComfyUI's checkpoint folder when needed,
submits a cat text-to-image workflow through the ComfyUI HTTP API, waits for
history completion, downloads the PNG, and rejects CPU-only device reports:

```bash
python scripts/comfyui_therock_gpu_test.py --generate-cat --output-dir .rocm-work/tests/comfyui-cat
```

The live test uses `rocm comfyui install`, `rocm comfyui start`, `rocm comfyui
status`, `rocm comfyui logs`, and `rocm comfyui stop`. It starts ComfyUI with
`--no-open-browser`, waits for `/system_stats`, verifies that
`rocm comfyui status` says `status: running`, and stops the process it started
unless `--keep-running` is set.

Manual non-TUI launch:

```bash
rocm comfyui install
rocm comfyui start
```

`rocm comfyui start` prints the local browser URL, normally
`http://127.0.0.1:8188`. It must also print `AMD GPU check: ready`; if that
check fails, rocm-cli stops instead of launching ComfyUI in CPU mode.

For WSL, run from the WSL filesystem instead of `/mnt`:

```bash
cd /home/$USER/rocm-cli-work/rocm-cli
CARGO_TARGET_DIR=/home/$USER/rocm-cli-work/target-linux cargo build -p rocm
python3 scripts/comfyui_therock_gpu_test.py \
  --rocm /home/$USER/rocm-cli-work/target-linux/debug/rocm \
  --temp-state \
  --copy-runtime-state-from /home/$USER/.rocm \
  --generate-cat
```

## Runtime Selection And Activation

List registered managed or read-only runtimes and their exact side-by-side keys:

```bash
rocm runtimes list
```

Activate one validated runtime:

```bash
rocm runtimes activate <runtime_key>
```

Engine installs and managed serving inherit this active runtime by default.
If no runtime is active, pass `--runtime-id` explicitly or activate one first;
the CLI does not fall back to a built-in TheRock selector.

Normal user testing should switch versions by activating the exact runtime key
printed by `rocm runtimes list`. The TUI equivalent is `/runtimes`, then arrow
to an installed ROCm entry and press Enter.

Developer-only previous-runtime regression check:

```bash
rocm runtimes rollback
```

This command validates the saved previous-runtime marker, but it is not the
primary user-facing way to switch ROCm installs.

Import an existing rocm-cli/TheRock runtime manifest in read-only mode:

```bash
rocm runtimes import /path/to/.rocm-cli-runtime.json
```

Adopt an existing TheRock Python environment in read-only mode without writing
into that environment:

```bash
rocm runtimes adopt \
  --python /path/to/venv/bin/python \
  --root /path/to/venv \
  --runtime-id therock-release:gfx120X-all \
  --runtime-key adopted-release-pip-gfx120x-all-7-13-0
```

Focused unit coverage:

```bash
cargo test -p rocm --bin rocm runtime_activation
cargo test -p rocm --bin rocm runtime_import
cargo test -p rocm --bin rocm runtime_adopt
cargo test -p rocm --bin rocm runtime_key_includes_version_for_side_by_side_installs
cargo test -p rocm --bin rocm engine_plugin
cargo test -p rocm --bin rocm missing_packaged_engine_reason
cargo test -p rocm --bin rocm model_registry
cargo test -p rocm --bin rocm render_model_registry_text_reports_host_ram_fit
cargo test -p rocm --bin rocm model_recipe
cargo test -p rocm --bin rocm model_completion
cargo test -p rocm --bin rocm logs
cargo test -p rocm-core model_recipe
cargo test -p rocm-core load_model_recipe_index
cargo test -p rocm --bin rocm update_report_policy
cargo test -p rocm-engine-pytorch stdio_protocol_routes_all_methods_without_side_effects
cargo test -p rocm-engine-llama-cpp stdio_protocol_routes_all_methods_without_side_effects
cargo test -p rocm-engine-atom
cargo test -p rocm-engine-vllm
cargo test -p rocm-engine-sglang
cargo test -p rocmd event_collector
cargo test -p rocmd event_dispatcher
```

## Provider-Assisted Planning

The deterministic planner remains the default. Optional LLM/provider ambiguity
resolution is only used after a user configures a planner provider:

```bash
rocm config set-planner-provider local
rocm "start a local model"
```

For OpenAI or Anthropic, prompt sending must also be enabled and a key must be
available through the OS secure store or session environment:

```bash
rocm config set-planner-provider openai
rocm config enable-provider openai
rocm config set-provider-key openai
```

Provider-assisted output is reduced to a validated `rocm` tool call. It is
shown for review in the TUI, but non-interactive `rocm --yes ...` does not
execute provider-assisted plans automatically.

Focused coverage:

```bash
cargo test -p rocm --bin rocm provider_planner
cargo test -p rocm --bin rocm freeform_execution_validation_rejects_provider_assisted_plans
```

Log browser smoke:

```bash
rocm logs
rocm logs --search server
rocm logs --service <service-id>
```

The no-log first-run path should report `no CLI logs found yet` without
creating the configured `data/logs` directory.

TUI log pagination focused tests:

```bash
cargo test -p rocm --bin rocm logs_browser
cargo test -p rocm --bin rocm logs_next
cargo test -p rocm --bin rocm logs_completion_suggests_service_flag_before_service_ids
cargo test -p rocm --bin rocm logs_tui_hides_file_locations_until_user_reveals_them
cargo test -p rocm --bin rocm service_log_details_scroll_without_changing_selected_action
```

After opening `/logs` or `/logs --search <query>` in the TUI, use the arrow-key
menu instead of typing paging subcommands. `Left`/`Right` page the log browser,
`PageUp`/`PageDown` scroll the details, `R` refreshes, and the `Show file
locations` row reveals advanced paths only when needed.

Engine inventory smoke on native Windows:

```powershell
rocm engines list
```

The packaged Linux/WSL-only ATOM, vLLM, and SGLang adapters should render
`runtime: unsupported_native_windows`, not `runtime: not found`.
The vLLM and SGLang live GPU acceptance scripts should return a clean skip on
native Windows. They remain strict GPU-required tests on Linux/WSL.

Serve resolver focused tests:

```bash
cargo test -p rocm --bin rocm serve_engine_selection
```

These tests verify that explicit/configured engines still win and shared model
recipes only choose a preferred engine when no user/default engine is set.

Engine recipe adapter contract focused tests:

```bash
cargo test -p rocm-engine-protocol engine_recipe_hint_roundtrips_through_resolve_request
cargo test -p rocm --bin rocm engine_recipe
cargo test -p rocm-engine-pytorch engine_recipe
cargo test -p rocm-engine-llama-cpp engine_recipe
cargo test -p rocm-engine-atom engine_recipe
cargo test -p rocm-engine-vllm engine_recipe
cargo test -p rocm-engine-sglang engine_recipe
```

These tests verify that signed-index engine-specific recipe metadata is mapped
into the versioned engine protocol hint, and that each first-party adapter
accepts matching hints, rejects mismatched engine ids or contract versions, and
echoes accepted hints from `resolve_model`. First-party launch commands also
apply accepted engine recipe `required_flags`; mismatched hints are rejected
instead of being silently ignored.

Contained update-check automation tests:

```bash
cargo test -p rocmd therock_update_contained_mode_runs_read_only_check_without_queueing
cargo test -p rocmd therock_update_contained_mode_records_update_available_without_applying
cargo test -p rocmd therock_update_contained_mode_uses_restricted_check_updates_tool
cargo test -p rocmd therock_update_notify_if_newer_uses_restricted_notification_contract
cargo test -p rocmd sandbox_check_updates_value_is_read_only_and_preserves_output
cargo test -p rocmd sandbox_check_updates_value_marks_runtime_update_available
cargo test -p rocmd sandbox_driver_plan_value_is_read_only_and_preserves_output
cargo test -p rocmd watcher_policy_maps_modes_to_decisions
```

Manual Windows smoke:

```powershell
rocmd sandbox-run check_updates --allow-native-fallback
rocmd sandbox-run driver_plan --allow-native-fallback
```

The JSON output should include `mutating: false` and the captured
`rocm update` stdout for `check_updates`. Its `status` is `checked` when no
runtime update is reported, `update_available` when the read-only report
contains `status=update_available`, or `error` when the check fails.
Contained `therock-update` consumes the restricted `check_updates` tool result
and records a `notify_if_newer` notification/audit event only for the
`update_available` case, while still leaving proposal history empty.
`driver_plan` should include `status: planned`, `mutating: false`, and captured
`rocm install driver --dkms --dry-run` stdout. Neither tool must apply updates
or execute driver commands.

Restricted sandbox tool API tests:

```bash
cargo test -p rocmd sandbox_tool_cli_values_cover_restricted_plan_api
cargo test -p rocmd sandbox_tool_examine_snapshot_is_read_only
cargo test -p rocmd sandbox_tool_list_servers_returns_records
cargo test -p rocmd sandbox_tool_list_servers_first_run_returns_empty_list
cargo test -p rocmd sandbox_tool_requires_service_id_for_restart
cargo test -p rocmd sandbox_tool_restart_server_reports_missing_service
cargo test -p rocmd sandbox_tool_requires_service_id_for_stop
cargo test -p rocmd sandbox_tool_stop_server_reports_missing_service
cargo test -p rocmd sandbox_tool_stop_server_updates_manifest_and_skips_current_pid
cargo test -p rocmd sandbox_tool_notify_user_is_read_only
cargo test -p rocmd sandbox_runner_native_fallback_records_audit
cargo test -p rocm --bin rocm proposal_sandbox_args_support_update_check
```

These cover the plan-listed restricted internal tool API: `check_updates`,
`examine_snapshot`, `list_servers`, `restart_server`, `stop_server`,
`prefetch_artifact`, and `notify_user`, plus the plan-derived `driver_plan`
read-only extension. Read-only tools must report `mutating: false`;
`notify_user` must record a local notification audit event; server restart/stop
must require an explicit service id; sandbox-run native fallback or Linux
bubblewrap isolation must still execute only this internal tool API and record
a sandbox audit event.

Manual restricted-tool smoke:

```bash
rocmd sandbox-run examine_snapshot --allow-native-fallback
rocmd sandbox-run list_servers --allow-native-fallback
rocmd sandbox-run notify_user --message "ROCm check complete" --allow-native-fallback
```

On Linux/WSL with `bubblewrap` installed, these commands should report
`isolation: bubblewrap`; on native Windows they should report
`isolation: native_restricted`.

Direct MCP helper safety smoke:

```bash
cargo test -p rocmd direct_mcp_call -- --nocapture
cargo test -p rocmd rocm_mcp_tools -- --nocapture
rocmd mcp-call examine --arguments-json '{}'
rocmd mcp-call install_sdk --arguments-json '{}'
```

The read-only `examine` helper call should run. The `install_sdk` helper call
must fail unless `--allow-mutation` is supplied after an explicit user
approval. User-facing TUI/chat flows should still route mutating tool calls
through their normal approval cards rather than relying on this hidden helper
flag.

Artifact cache and prefetch policy tests:

```bash
cargo test -p rocm-core model_artifact_cache
cargo test -p rocm --bin rocm model_recipe_artifact_lines_surface_signed_index_metadata
cargo test -p rocmd sandbox_prefetch_downloads_direct_artifact_when_approved
cargo test -p rocmd sandbox_prefetch_blocks_approved_artifact_without_sha256
cargo test -p rocmd sandbox_prefetch_blocks_artifact_larger_than_approved_limit
cargo test -p rocmd sandbox_prefetch_cached_marker_skips_network
cargo test -p rocmd sandbox_prefetch_blocks_non_direct_non_huggingface_source
cargo test -p rocmd sandbox_prefetch_unknown_artifact_ref_errors
cargo test -p rocmd sandbox_prefetch_reports_policy_required_without_network
cargo test -p rocmd sandbox_prefetch_blocks_gated_huggingface_without_source_policy
cargo test -p rocmd sandbox_prefetch_blocks_huggingface_source_policy_without_token
cargo test -p rocmd sandbox_prefetch_respects_manual_only_source_policy
cargo test -p rocmd sandbox_prefetch_respects_declared_huggingface_auth_policy
cargo test -p rocmd sandbox_prefetch_never_sends_huggingface_token_to_non_huggingface_url
cargo test -p rocmd sandbox_prefetch_never_sends_huggingface_token_over_plain_http
cargo test -p rocmd sandbox_prefetch_requires_artifact_ref
```

`prefetch_artifact` is no-network by default. Without approval it should report
`source_policy_required`, `mutating: false`, and `network_used: false`.
Approved live prefetch uses:

```bash
rocmd sandbox-run prefetch_artifact \
  --artifact-ref <model-ref>#<artifact-id> \
  --allow-artifact-download \
  --artifact-max-bytes <bytes> \
  --allow-native-fallback
```

The approved path supports direct HTTP(S) artifact URIs that include
`size_bytes` and `sha256` recipe metadata. It downloads the bytes, enforces the
byte limit, verifies sha256, writes an atomic cache marker, and reports
`status: prefetched`. Gated artifacts, missing hashes, and non-direct sources
remain blocked instead of falling back to an unverified download unless a
source-specific policy explicitly applies.

Signed recipe artifacts can also declare `source_policy` metadata. The
signed-index loader validates policy names, host constraints, HTTPS
requirements, and required integrity metadata. `/model` shows the policy, and
`prefetch_artifact` enforces `manual_only` and authenticated Hugging Face
requirements even after a generic artifact download approval.

For gated Hugging Face artifacts, the approved path also requires
`--allow-huggingface-download` plus `ROCM_CLI_HUGGINGFACE_TOKEN`, `HF_TOKEN`,
or `HUGGING_FACE_HUB_TOKEN`. rocm-cli should still refuse to send that token to
anything other than an HTTPS Hugging Face URL, and tokens must not appear in
JSON output. On Windows, pass `--allow-native-fallback` when manually running
`rocmd sandbox-run` because bubblewrap isolation is Linux-only.

Driver reconciliation focused tests:

```bash
cargo test -p rocm --bin rocm driver_plan_ -- --nocapture
cargo test -p rocm --bin rocm driver_reconcile_updates_state_after_reboot
cargo test -p rocm --bin rocm driver_passive_check_summary_counts_non_present_as_missing
```

The driver-plan tests are non-privileged. They verify AMD-documented
package-manager plans for Ubuntu, Debian, RHEL, Oracle Linux, Rocky, and SLES,
while unsupported Linux IDs remain non-mutating instead of guessed. The
reconcile path is also non-privileged. It refreshes passive checks and persists
the `total`/`present`/`missing` check summary without running DKMS commands.

Provider audit focused test:

```bash
cargo test -p rocm --bin rocm local_provider_stream_chat_posts_sse_request
cargo test -p rocm --bin rocm remote_openai_stream_chat_uses_live_sse
cargo test -p rocm --bin rocm remote_anthropic_stream_chat_uses_live_sse
```

This verifies the streamed local provider path writes a `provider` audit event
with action `stream_chat` after the SSE stream completes. The remote provider
tests use local HTTP fixtures to prove OpenAI and Anthropic send `stream: true`
requests and deliver the first SSE delta to the callback before the server
closes the connection.

TUI focused approval tests:

```bash
cargo test -p rocm --bin rocm onboarding_pending_approval_shows_plain_install_details
cargo test -p rocm --bin rocm onboarding_custom_folder_approval_uses_selected_folder
cargo test -p rocm --bin rocm onboarding_folder_rejects_system_folder
cargo test -p rocm --bin rocm onboarding_folder_rejects_file_path
cargo test -p rocm --bin rocm onboarding_folder_rejects_missing_parent
cargo test -p rocm --bin rocm onboarding_install_current_step_covers_pre_pip_phases
cargo test -p rocm --bin rocm onboarding_running_install_empty_log_uses_general_starting_text
cargo test -p rocm --bin rocm onboarding_failed_install_shows_full_log_path_for_noisy_output
cargo test -p rocm --bin rocm render_command_output_keeps_untruncated_output_for_full_logs
cargo test -p rocm --bin rocm empty_enter_approves_focused_proposal
cargo test -p rocm --bin rocm restart_proposal_approval_uses_plain_review_language
cargo test -p rocm --bin rocm prefetch_proposal_approval_shows_plain_artifact_download_controls
cargo test -p rocm --bin rocm driver_plan_proposal_approval_says_no_driver_install
cargo test -p rocm --bin rocm render_automations_text_uses_plain_proposal_history
cargo test -p rocm --bin rocm tab_cycles_and_enter_accepts_slash_arguments
cargo test -p rocm --bin rocm approval_status_says_enter_or_y_approve
cargo test -p rocm --bin rocm esc_rejects_focused_proposal
```

The onboarding approval test verifies that first-run setup shows the selected
install folder, the downloads/cache folder, TheRock release wheel source,
package summary, and plain approve/cancel controls before installing ROCm.
The install-step test verifies that pre-pip phases render friendly labels for
checking ROCm packages, checking/downloading/installing Python, creating the
Python environment, installing pip, installing ROCm packages, and the final SDK
check.
The folder validation tests verify that setup rejects obvious system folders,
file paths, and folders whose parent does not exist before the installer starts.
The noisy-failure test verifies that long setup installs still save a complete
log under `data/logs/setup/` and show that full-log path on the failure screen.
The command-output test verifies that command rendering keeps the raw output
available for full logs while the transcript display remains bounded by the
separate truncation step.
The proposal approval tests verify that restart, artifact prefetch, and driver
plan reviews use plain-language sections and do not expose backend watcher,
action, tool, service id, sandbox-run, or raw internal tool names in the
default review card.
The plain proposal history test verifies that `/automations` and
`rocm automations` show recent review requests with friendly action, reason,
server/artifact details, and approve/reject controls instead of raw backend
action or restricted-tool names.
Enter accepts the highlighted completion first. If no completion is accepted
and a focused approval is visible with an empty input, Enter approves it.

Natural-language plan handoff tests:

```bash
cargo test -p rocm --bin rocm hybrid_planner_normalizes_model_alias_and_structured_serve_call
cargo test -p rocm --bin rocm freeform_plan_next_action_surfaces_approval_action
cargo test -p rocm --bin rocm freeform_invocation_supports_leading_yes_for_natural_language_only
cargo test -p rocm --bin rocm freeform_invocation_rejects_unquoted_structured_command_names_after_yes
cargo test -p rocm --bin rocm freeform_invocation_rejects_flag_shaped_yes_request
cargo test -p rocm --bin rocm freeform_execution_validation_rejects_placeholder_tool_calls
cargo test -p rocm --bin rocm freeform_execution_header_surfaces_explicit_approval_and_tool_call
cargo test -p rocm --bin rocm hybrid_planner_driver_action_includes_yes_for_approved_execution
cargo test -p rocm --bin rocm render_freeform_plan_exposes_structured_tool_calls
cargo test -p rocm --bin rocm natural_serve_command_renders_llama_plan
cargo test -p rocm --bin rocm natural_serve_with_missing_model_does_not_focus_approval
```

In the TUI, recognized mutating plain-language requests render a plan and then
focus the normal approval card. Missing placeholders remain plan-only.
Outside the TUI, `rocm --yes <natural language request>` renders the same plan
and executes only the final structured `rocm` tool call after placeholder
validation. Requests beginning with flags or unquoted structured command names
are rejected; quote the natural-language request or phrase it as natural
language, and run structured commands directly with their own approval flag
when they define one.

Cloud provider opt-in tests:

```bash
cargo test -p rocm-core provider_config_defaults_to_local_only
cargo test -p rocm --bin rocm provider
cargo test -p rocm --bin rocm render_config_text_includes_telemetry_policy
```

OpenAI and Anthropic prompt sending must fail before env-key lookup or network
access until the provider is explicitly enabled with `rocm config
enable-provider <provider>`.

Cloud API keys are stored outside `config.json`:

```bash
printf '%s' "$OPENAI_API_KEY" | rocm config set-provider-key openai
rocm config enable-provider openai
rocm config show
rocm config clear-provider-key openai
```

`rocm config show` should report whether the key is saved in the OS secure
store, missing, or coming from `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` for the
current session. It must never print the key. Remote provider HTTP requests are
sent in-process so API keys are not passed through `curl` command arguments.
In the TUI, clearing a saved API key defaults to `Cancel`; users must move the
highlight to `Clear API key` before Enter removes anything.

Focused TUI provider-key coverage:

```bash
cargo test -p rocm --bin rocm config_clear_provider_key_requires_arrow_key_confirmation
cargo test -p rocm --bin rocm config_provider_key_storage_failure_shows_retry_guidance
```

Update-surface report test:

```bash
cargo test -p rocm --bin rocm render_update_text_reports_all_update_surfaces
cargo test -p rocm --bin rocm plain_update_report_detects_status_update_available
```

`rocm update` must report runtime checks plus CLI, engine, and model-recipe
surfaces honestly. `rocm update --apply` remains runtime-only until production
metadata feeds exist for the other surfaces.

TUI mode-state checks:

```bash
cargo test -p rocm tui_mode
cargo test -p rocm natural_serve_command_renders_llama_plan
cargo test -p rocm logs_follow
cargo test -p rocm --bin rocm advertised_slash_commands
cargo test -p rocm --bin rocm manager_screens_page_keys_scroll_details_without_changing_selection
cargo test -p rocm --bin rocm approve_and_reject_commands_open_review_before_mutating
cargo test -p rocm --bin rocm slash_completion_advertises_guided_model_and_install_defaults
```

The sidebar must show the current mode (`ask`, `act`, `serve`, `logs`, or
`automations`), mode changes must not launch commands in tests, and `/logs
follow [query]` must refresh the current log browser without creating first-run
log directories. Advertised TUI commands should keep arrow-key focus on the
visible screen even after accidental prompt typing, manager PageUp/PageDown
should scroll details without changing the highlighted row, and automation
review approve/reject commands should open the visible review card before
mutating proposal status.

Model recipe engine metadata checks:

```bash
cargo test -p rocm-core model_recipe_index
cargo test -p rocm-core model_recipe_index_signature_accepts_generated_key_and_rejects_tamper
cargo test -p rocm model_recipe_artifact_lines_surface_signed_index_metadata
cargo test -p rocm render_freeform_plan_exposes_structured_tool_calls
```

Engine-specific recipe metadata uses a versioned adapter/protocol hint for
`resolve_model` and launch. `/model` and `rocm serve` should show selected
engine recipe metadata clearly. First-party adapters must apply accepted
launch-time `required_flags`, reject mismatched engine ids or contract
versions, and fail loudly for unsupported flags instead of silently ignoring
them.

Release trust checks:

```bash
cargo test -p rocm --bin rocm metadata_signature_verification_accepts_generated_key_and_rejects_tamper
cargo test -p rocm-core model_recipe_index_signature_accepts_generated_key_and_rejects_tamper
python scripts/release_readiness.py --self-test
ROCDXG_CHECKSUM_SELF_TEST=1 bash scripts/wsl_setup_rocdxg.sh
bash scripts/setup-wsl-portable-build-deps.sh --self-test
./scripts/acceptance-install-upgrade-tui-uninstall.sh
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\acceptance-install-upgrade-tui-uninstall.ps1
```

The release-readiness self-test is cross-platform and uses only workspace-local
temporary files under `.rocm-work/tests/release-readiness`. It also checks exact
release asset sets, so stale archives and orphan checksum/signature sidecars in
`dist` fail before upload, and it validates both missing and explicitly
configured production trust inputs. Normal Linux and Windows CI run this
self-test before the install-lifecycle acceptance scripts. The Linux and
Windows install-lifecycle acceptance scripts also reject bad checksums, bad
detached signatures, missing required signature sidecars, and required-signature
mode without a public key before activation. They also cover generated
private-key PEM packaging and generated public-key PEM installer verification.
They also verify first-install PATH setup: Windows persists the install folder
to the user PATH, and Linux/WSL writes the shell profile while running the
installer itself with a minimal PATH that excludes developer Cargo paths.
These checks use generated local keys only; project-owned production signing
keys remain an owner-controlled release step.

By default, the Windows script keeps temporary state under
`target/acceptance-windows`, and the Linux/WSL script keeps temporary state
under `.rocm-work/acceptance-linux`. Both roots are cleaned up unless the script
fails or `ROCM_CLI_KEEP_ACCEPTANCE_ROOT=1` is set for debugging. Installed
binary smoke checks set isolated config/data/cache directories inside those
roots and fail if `rocm examine` reads the real user `.rocm` state.

For historical platform-bundle acceptance, the Linux bundle still verifies the
vendored `rocm-codex` binary. The native rocm-cli binary does not include a
vendored Codex binary or require it as a sidecar. If the host
does not have `libcap-dev` or `libssl-dev`, the Linux acceptance script
downloads the Ubuntu development packages into `.rocm-work/tools/wsl-build-deps`,
extracts only the headers, libraries, and pkg-config metadata there, and points
`PKG_CONFIG_PATH`/`PKG_CONFIG_SYSROOT_DIR` at that local copy. No sudo install
is required.
Run `bash scripts/setup-wsl-portable-build-deps.sh --self-test` to verify the
portable sysroot normalization path without apt, network access, or a real WSL
package download.

Automation endpoint-health event tests:

```bash
cargo test -p rocmd event_collector
cargo test -p rocmd event_collector_emits_endpoint_recoverable_service_event
cargo test -p rocmd server_recover_local_webhook_does_not_restart_healthy_service
cargo test -p rocmd recovery_reason_display_avoids_raw_status_tokens
```

Automation GPU-metrics event tests:

```bash
cargo test -p rocmd gpu_metrics
cargo test -p rocmd event_collector_emits_gpu_metrics_event_when_enabled
```

The `gpu-metrics` watcher is read-only in all modes. It records bounded local
`amd-smi` telemetry availability or unavailability and must not queue proposals
or take thermal/serving actions until a policy is explicitly defined.

Automation cache-warm event tests:

```bash
cargo test -p rocmd local_webhook_cache_warm_builds_prefetch_event
cargo test -p rocmd local_webhook_cache_warm_rejects_other_cache_events
cargo test -p rocmd cache_warm_propose_mode_queues_prefetch_proposal
cargo test -p rocmd cache_warm_unknown_artifact_does_not_queue_proposal
cargo test -p rocmd cache_warm_contained_mode_still_requires_reviewed_source_policy
cargo test -p rocm --bin rocm proposal_sandbox_args_support_cache_warm_prefetch
cargo test -p rocm --bin rocm proposal_sandbox_args_support_cache_warm_prefetch_source_policy
cargo test -p rocm --bin rocm proposal_sandbox_args_reject_cache_warm_download_without_max_bytes
cargo test -p rocm --bin rocm proposal_edit_updates_prefetch_source_policy
```

The `cache-warm` watcher accepts exactly `cache.warm`, requires
`payload.artifact_ref`, verifies that the ref exists in the model recipe
registry before queueing, queues a reviewed `prefetch_artifact` proposal, and
does not grant webhook payloads arbitrary tool choice or automatic download
permission. The TUI can add reviewed source-policy fields with
`/automations edit <id> allow-download yes` and
`/automations edit <id> artifact-max-bytes <bytes>`; approved downloads require
that byte limit.

Automation driver-upgrade event tests:

```bash
cargo test -p rocmd local_webhook_driver_upgrade
cargo test -p rocmd driver_upgrade
cargo test -p rocmd driver_upgrade_contained_mode_runs_restricted_driver_plan
cargo test -p rocmd driver_upgrade_contained_mode_requires_restricted_driver_plan_tool
cargo test -p rocmd sandbox_driver_plan_value_is_read_only_and_preserves_output
cargo test -p rocm-core builtin_watchers_include_reviewed_driver_upgrade
cargo test -p rocm --bin rocm proposal_sandbox_args_support_driver_upgrade_plan
```

The `driver-upgrade` watcher accepts exactly `update.available` with
`payload.component=driver`. In `propose` mode it queues a reviewed
`driver_plan` proposal. In `contained` mode it consumes the restricted
`driver_plan` tool result directly. Both paths stay read-only; the user-facing
approval should say it shows the driver plan only and that no driver will be
installed.

Automation local-webhook source tests:

```bash
cargo test -p rocmd local_webhook
cargo test -p rocmd local_webhook_therock_update_accepts_exact_schedule_tick_only
cargo test -p rocmd local_webhook_server_recover_rejects_nonrecoverable_service_kind
cargo test -p rocmd local_webhook_gpu_metrics_rejects_thermal_action_kinds
cargo test -p rocm render_automations_text_surfaces_local_webhook_endpoint
```

The local webhook source is loopback-only. It is enabled only with
`rocmd run --automations-enabled --local-webhook-port <port>`, accepts JSON
`POST /automation-events`, validates `watcher_hint` and `kind` against the
exact event kinds accepted by that watcher, and then dispatches through the
same watcher policy paths as scheduler, managed-service recovery, GPU
telemetry, and cache-warm proposal events. See `docs/automations.md` for the
local curl smoke and accepted fields.

ATOM TheRock GPU acceptance:

```bash
cargo test -p rocm-engine-atom managed_env_reflects_managed_runtime_manifest_source
cargo test -p rocm-engine-atom running_state_records_managed_therock_env_for_gpu_verification
python -m py_compile scripts/atom_therock_gpu_test.py
python scripts/atom_therock_gpu_test.py --self-test
```

Live ATOM ROCm serving is only valid on a GPU/runtime/model combination
supported by upstream ATOM. The current WSL validation host is `gfx1201`, while
upstream ATOM documentation currently lists Instinct `gfx950`, `gfx942`, and
`gfx90a` targets. Do not force a different target and do not use CPU fallback.
On a supported ATOM ROCm target, install/build ATOM inside a rocm-cli managed
TheRock runtime, then verify `rocm-engine-atom detect` reports
`managed_env: true`, launch with `gpu_required`, and check loaded HIP libraries
from the managed TheRock SDK wheel directories:

```bash
python3 scripts/atom_therock_gpu_test.py \
  --engine /home/user/.cache/rocm-cli-target/debug/rocm-engine-atom \
  --model Qwen/Qwen3-0.6B
```

The script defaults to the active exact runtime key. If `--runtime-id` is used,
it must be an exact runtime key or an unambiguous runtime id. It rejects
external ATOM command/Python overrides and does not allow CPU fallback.

vLLM TheRock GPU acceptance:

```bash
cargo test -p rocm-engine-vllm running_state_records_managed_therock_env_for_gpu_verification
cargo test -p rocm-engine-vllm managed_env_reflects_managed_runtime_manifest_source
cargo test -p rocm-engine-vllm resolve_model_surfaces_conservative_vram_default
python -m py_compile scripts/vllm_therock_gpu_test.py
python scripts/vllm_therock_gpu_test.py --self-test

# On Linux/WSL with vLLM installed in the active rocm-cli managed TheRock venv:
python3 scripts/vllm_therock_gpu_test.py \
  --engine /home/user/.cache/rocm-cli-target/debug/rocm-engine-vllm \
  --model facebook/opt-125m
```

The script defaults to the active exact runtime key. If `--runtime-id` is used,
it must be an exact runtime key or an unambiguous runtime id. It requires
`gpu_required`, rejects external vLLM command overrides, checks `/health` and
`/v1/completions`, and verifies loaded ROCm libraries came from the managed
TheRock SDK wheel directories.
For TheRock 7.13, patch vLLM's GPTQ ROCm compatibility guard to include HIP
7.13 before building from source; otherwise `q_gemm.hip` can fail on missing
`half`/`half2` `atomicAdd` overloads.
On native Windows this script prints a JSON skip result; run it from WSL/Linux
for live ROCm GPU acceptance.

SGLang TheRock GPU acceptance:

```bash
cargo test -p rocm-engine-sglang managed_env_reflects_managed_runtime_manifest_source
cargo test -p rocm-engine-sglang running_state_records_managed_therock_env_for_gpu_verification
python -m py_compile scripts/sglang_therock_gpu_test.py
python scripts/sglang_therock_gpu_test.py --self-test
```

Live SGLang ROCm serving is only valid on a GPU target supported by upstream
SGLang ROCm kernels. The current WSL validation host is `gfx1201`; SGLang
`v0.5.12` and current `origin/main` reject `sgl-kernel/setup_rocm.py` for that
target and accept only `gfx942`/`gfx950`. Do not force a different target and
do not use CPU fallback. On a supported SGLang ROCm target, install/build
SGLang inside a rocm-cli managed TheRock runtime, then verify
`rocm-engine-sglang detect` reports `managed_env: true`, launch with
`gpu_required`, and check loaded HIP libraries from the managed TheRock SDK
wheel directories:

```bash
python3 scripts/sglang_therock_gpu_test.py \
  --engine /home/user/.cache/rocm-cli-target/debug/rocm-engine-sglang \
  --model Qwen/Qwen2.5-1.5B-Instruct
```

The script defaults to the active exact runtime key. If `--runtime-id` is used,
it must be an exact runtime key or an unambiguous runtime id. It rejects
external SGLang command/Python overrides and does not allow CPU fallback.
For MI300X/gfx942 TheRock 7.13, install SGLang from source with
`python/pyproject_other.toml` and the ROCm `sgl-kernel` wheel. Use the
rocm-cli adapter's Triton attention default unless AITER has been built and
verified for the runtime.
On native Windows this script prints a JSON skip result; run it from WSL/Linux
for live ROCm GPU acceptance.

## Windows Tool Notes

The TheRock SDK wheel install path should not require users to install global
Python or curl. rocm-cli uses Rust-native HTTP downloads and can bootstrap a
managed Python under its data directory when no usable Python is available.

TheRock's Windows source-build guide documents `winget` and manual installs for
MSVC, Git, CMake, Ninja, Python, Strawberry Perl, DVC, and related tools. The
SDK install test reports these when `--check-windows-tools` is set, but normal
SDK wheel setup should avoid global source-build tools.

Reference: [TheRock Windows install tools](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md#install-tools)

## llama.cpp TheRock GPU Test

This opt-in test requires a HIP-enabled `llama-server`, a rocm-cli managed
TheRock runtime, and an AMD GPU. It does not allow CPU fallback.

```bash
python scripts/llama_cpp_therock_gpu_test.py --llama-server target/llama.cpp-build-hip/bin/llama-server.exe
```

To verify the background `launch` command also returns promptly when its JSON
output is captured by a shell or test harness:

```bash
python scripts/llama_cpp_therock_gpu_test.py --launch-mode launch --llama-server target/llama.cpp-build-hip/bin/llama-server.exe
```

WSL command shape, using the WSL-built adapter and server:

```bash
python3 scripts/llama_cpp_therock_gpu_test.py \
  --engine /home/user/.cache/rocm-cli-target/debug/rocm-engine-llama-cpp \
  --llama-server /home/user/.cache/rocm-cli-llama.cpp-build-hip/bin/llama-server \
  --model-path /mnt/d/path/to/rocm-cli/target/models/stories260K.gguf \
  --timeout 120
```

The script uses the active rocm-cli runtime key. If no runtime is active, run
`rocm runtimes activate <runtime_key>` first or pass `--runtime-id
<runtime_key>`. A broad runtime id is only accepted when it matches one saved
runtime unambiguously; the test no longer guesses by picking the newest
manifest. When testing isolated state, set `ROCM_CLI_CONFIG_DIR` and
`ROCM_CLI_DATA_DIR`; the script should honor those directories instead of
looking in the default `~/.rocm` state. If `CARGO_TARGET_DIR` is set, the
default adapter binary is resolved from that target directory. It downloads or
reuses the tiny `stories260K.gguf` model, launches the llama.cpp adapter with
`gpu_required`, checks `/health` and `/v1/completions`, and on Windows verifies
that HIP DLLs load from the staged rocm-cli TheRock runtime rather than
`System32`. On Linux/WSL it verifies HIP and BLAS shared objects are loaded
from the managed TheRock SDK wheel roots via `/proc/<pid>/maps`, including
split `_rocm_sdk_core` and `_rocm_sdk_libraries_*` wheel directories.

Offline selector sanity check:

```bash
python scripts/llama_cpp_therock_gpu_test.py --self-test
```

Top-level managed serving smoke:

```bash
# Set ROCM_CLI_LLAMA_CPP_SERVER if llama-server is not on PATH.
rocm serve target/models/stories260K.gguf --engine llama.cpp --managed --port 11450
rocmd sandbox-run stop_server --service-id <printed-service-id> --allow-native-fallback
```

## WSL Preflight

Read-only WSL/ROCDXG preflight:

```bash
python scripts/wsl_preflight.py --json
python scripts/wsl_preflight.py --require-ready
```

`--require-ready` checks WSL, `/dev/dxg`, DXCore, ROCDXG, `python3 -m venv`,
and library registration. Source-build tools such as Windows SDK headers,
CMake, and compilers are optional for runtime acceptance; add
`--require-build-tools` only when validating a WSL source-build environment.

Interactive ROCDXG install inside WSL:

```bash
bash scripts/wsl_setup_rocdxg.sh
python scripts/wsl_preflight.py --require-ready
```

To require checksum verification for the downloaded ROCDXG `.deb`, provide the
expected package digest from a trusted release source:

```bash
ROCDXG_SHA256=<64-hex-sha256> bash scripts/wsl_setup_rocdxg.sh
```
