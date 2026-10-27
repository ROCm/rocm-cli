// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for the interactive dashboard/chat TUI, driven black-box through a
//! pseudo-terminal (see `tui_driver`). These are the only steps that exercise
//! the real crossterm event loop end to end — launch, key input, rendered
//! screen, and clean exit — which the piped-`Command` steps structurally cannot.

use cucumber::{given, then, when};
use e2e_cucumber::mock_server::{MetricsMode, MockServer, ServiceRecordOptions};
use std::time::{Duration, Instant};

use crate::E2eWorld;
use crate::e2e::tui_driver::{QUIT_CONFIRM_MARKER, TermSignal, TuiSession, default_timeout};
/// The exact prompt `send_managed_model_message` types, and the string the
/// corresponding `Then` step (`managed_chat_request_carried_prompt`) asserts
/// the mock actually received — so the two can never silently drift apart.
const MANAGED_MODEL_PROMPT: &str = "hello from the terminal";

/// The services overlay's own panel title, drawn by `draw_services_manager` on
/// the overlay's border row. It is on screen exactly while the overlay is, so
/// it proves both that the overlay opened and - as an absence - that it closed.
const SERVICES_OVERLAY_TITLE: &str = "Services — managed inference servers";

/// The Observe instances table's TTFT cell while the scripted mock is serving:
/// its histogram pins time-to-first-token at exactly 50 ms
/// (`ttft_sum_s = ticks × 0.050` over `ttft_count = ticks`), and the cell is
/// rendered `"{v:.0}ms"`. A *failed* scrape clears `ttft_ms`/`tpot_ms`
/// (`runner.rs`), so this cell changing is the screen's own proof that the
/// frame on display was assembled after the failure — the only frame the
/// held-throughput assertion is about.
const SCRIPTED_TTFT_CELL: &str = "50ms";

/// Zero-based index of the TTFT cell within an Observe instances row, counting
/// from the model id: `MODEL TOK/S TOK/W TTFT TPOT POWER QUEUE KV%`
/// (`instances.rs`).
///
/// The cell is read by position on the scripted instance's own row rather than
/// matched as a substring of the whole screen. A bare substring cannot tell a
/// cleared cell from a surviving one: any future ms-suffixed value that merely
/// *contains* the scripted one (`150ms`, `250ms`), or a second row whose TTFT
/// is also 50 ms, would keep the marker on screen and time this step out for a
/// reason that has nothing to do with the scrape it synchronises on.
const TTFT_COLUMN: usize = 3;

/// Borrow the scenario's active TUI session, or fail clearly if none was opened.
const fn session(world: &mut E2eWorld) -> &mut TuiSession {
    world
        .tui
        .as_mut()
        .expect("no interactive TUI session is open for this scenario")
}

// This is a protocol contract journey: CLI, embedded daemon and TUI are real;
// the HTTP service is a named protocol fixture, not GPU inference.
const PROTOCOL_MODELS: [&str; 2] = ["Protocol-Qwen-4B", "Protocol-Phi-3B"];

#[given("an external Lemonade server has two models loaded")]
async fn external_lemonade_protocol(world: &mut E2eWorld) {
    let server = e2e_cucumber::lemonade_protocol::LemonadeProtocolServer::start(&PROTOCOL_MODELS);
    configure_lemonade_discovery(world, server.port());
    let root = world.isolated_root.as_ref().expect("isolated root").path();
    assert!(
        !root.join("data/services").exists(),
        "no managed service may be planted"
    );
    world.lemonade_protocol = Some(server);
}

fn configure_lemonade_discovery(world: &E2eWorld, port: u16) {
    // Discovery is off by default in the isolated config (see
    // `disable_lemonade_discovery_by_default`), so these scenarios turn it on
    // explicitly. `lemonade_host` is omitted to exercise its serde default.
    // Socket isolation comes from the PTY driver's HOME/XDG environment.
    write_dashboard_config(
        world,
        serde_json::json!({
            "dashboard": { "daemon": { "enable_lemonade": true, "lemonade_port": port,
                "discovery_tick_secs": 0.1, "instance_tick_secs": 0.1 } }
        }),
    );
}

#[given("the dashboard watches only managed services")]
async fn dashboard_managed_only(world: &mut E2eWorld) {
    // An unrelated live endpoint on the default port must not supply the
    // model or external-running status in this real managed-service journey.
    write_dashboard_config(
        world,
        serde_json::json!({"dashboard": {"daemon": {"enable_lemonade": false}}}),
    );
}

/// Keep a dashboard that is not testing Lemonade discovery away from any real
/// Lemonade server on the default port. Call before every dashboard spawn.
fn disable_lemonade_discovery_by_default(world: &E2eWorld) {
    let root = world.isolated_root.as_ref().expect("isolated root").path();
    let dir = root.join("config");
    std::fs::create_dir_all(&dir).expect("create config dir");
    let path = dir.join("config.json");
    let existing = std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());
    let config = e2e_cucumber::dash_screen::with_lemonade_discovery_default_off(existing);
    std::fs::write(
        &path,
        serde_json::to_vec(&config).expect("serialize config"),
    )
    .expect("write config");
}

fn write_dashboard_config(world: &E2eWorld, config: serde_json::Value) {
    let root = world.isolated_root.as_ref().expect("isolated root").path();
    let dir = root.join("config");
    std::fs::create_dir_all(&dir).expect("create config dir");
    let path = dir.join("config.json");
    // An earlier step may have written the active runtime here; keep it.
    let existing = std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());
    let merged = e2e_cucumber::dash_screen::merge_dashboard_daemon_config(existing, &config);
    std::fs::write(
        &path,
        serde_json::to_vec(&merged).expect("serialize config"),
    )
    .expect("write config");
}

#[then("both loaded models are displayed with throughput shown as unknown")]
async fn external_models_visible(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen_where(
            "both protocol models, with unknown unattributable throughput",
            |screen| {
                PROTOCOL_MODELS.iter().all(|model| {
                    screen
                        .lines()
                        .any(|line| protocol_tps_cell(line, model) == Some("—"))
                })
            },
            default_timeout(),
        )
        .await
        .expect("external models should be discovered");
}

fn protocol_tps_cell<'a>(line: &'a str, model: &str) -> Option<&'a str> {
    let mut cells = line.split_whitespace().skip_while(|cell| *cell != model);
    cells.next()?;
    cells.next()
}

#[when("the external Lemonade server unloads its second model")]
async fn external_single_model(world: &mut E2eWorld) {
    world
        .lemonade_protocol
        .as_ref()
        .expect("fixture")
        .set_loaded(&PROTOCOL_MODELS[..1]);
}

#[then("only the first model is displayed, with its throughput")]
async fn external_single_model_rate(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen_where(
            "single protocol model with unambiguous last-request throughput",
            |screen| {
                !screen.contains(PROTOCOL_MODELS[1])
                    && screen
                        .lines()
                        .any(|line| protocol_tps_cell(line, PROTOCOL_MODELS[0]) == Some("38"))
                    // The rate is the last completed request's, and the screen says so.
                    && screen.contains("last req tok/s")
            },
            default_timeout(),
        )
        .await
        .expect("single loaded model owns endpoint stats");
}

#[when("the external Lemonade server unloads every model")]
async fn external_unload(world: &mut E2eWorld) {
    world
        .lemonade_protocol
        .as_ref()
        .expect("fixture")
        .set_loaded(&[]);
}

#[then("no loaded model is displayed and none are reported as discovered")]
async fn external_models_absent(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen_where(
            "unloaded external models disappear",
            |screen| {
                PROTOCOL_MODELS.iter().all(|m| !screen.contains(m))
                    && screen.contains("no serving models discovered")
            },
            default_timeout(),
        )
        .await
        .expect("stale external models must disappear");
}

#[when("the external Lemonade server loads both models again")]
async fn external_reload(world: &mut E2eWorld) {
    world
        .lemonade_protocol
        .as_ref()
        .expect("fixture")
        .set_loaded(&PROTOCOL_MODELS);
}

#[when("the external Lemonade server stops responding")]
async fn external_unreachable(world: &mut E2eWorld) {
    drop(world.lemonade_protocol.take());
}

#[then("the external server received no lifecycle request and no managed record exists")]
async fn external_server_untouched(world: &mut E2eWorld) {
    // What this proves: the fixture recorded only observational requests (the
    // stray-request check below) and the registry holds no record. The health
    // read is a sanity check that the fixture still answers; the fixture is an
    // in-process thread the dashboard cannot stop, so it is not evidence of
    // anything on its own.
    let fixture = world.lemonade_protocol.as_ref().expect("fixture");
    let response = reqwest::get(format!("{}/api/v1/health", fixture.base_url()))
        .await
        .expect("dashboard quit must leave external service alive");
    assert!(response.status().is_success());
    let body: serde_json::Value = response.json().await.expect("health JSON");
    assert_eq!(body["status"], "ok");
    // The dashboard may only observe: any load, unload, stop or other request
    // reaching the external server would be a lifecycle action on a service it
    // does not own.
    assert_eq!(
        fixture.stray_requests(),
        Vec::<String>::new(),
        "the dashboard sent a non-observational request to the external server"
    );
    assert_eq!(
        body["all_models_loaded"].as_array().expect("models").len(),
        2
    );
    let services = world
        .isolated_root
        .as_ref()
        .expect("root")
        .path()
        .join("data/services");
    assert!(
        !services.exists()
            || std::fs::read_dir(services)
                .expect("services")
                .next()
                .is_none(),
        "discovery must not adopt an external fixture as a managed service"
    );
}

#[when("the user selects engine status in the Serving view")]
async fn engine_status_pane(world: &mut E2eWorld) {
    let tui = session(world);
    tui.use_detail_size().expect("enlarge dashboard");
    tui.send_until("3", "● Serving", default_timeout())
        .await
        .expect("Serving view");
    tui.send("gj").expect("select Engines action");
}

#[then("Lemonade is shown as externally running without a managed installation")]
async fn external_installation_truth(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen_where(
            "external versus managed engine state",
            |screen| {
                screen.contains("Lemonade — externally running")
                    && screen.contains("Managed runtime — not installed")
            },
            default_timeout(),
        )
        .await
        .expect("engine status must distinguish external service and managed install");
}

#[then("Lemonade returns to the managed installation status without external availability")]
async fn external_availability_cleared(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen_where(
            "unreachable endpoint no longer counts as external availability",
            |screen| {
                screen.contains("Lemonade — managed runtime not installed")
                    && !screen.contains("Lemonade — externally running")
            },
            default_timeout(),
        )
        .await
        .expect("engine status must return to managed inventory after endpoint loss");
}

#[then("the real managed Lemonade model is displayed")]
async fn real_lemonade_visible(world: &mut E2eWorld) {
    // The preceding real inference assertion identifies this scenario's model.
    // Direct llama-server need not expose Lemonade stats: missing throughput is
    // truthful and must not cause us to invent a rate to satisfy this journey.
    //
    // Anchored to a full Observe instances row (the model cell followed by the
    // TOK/S .. KV% cells), not to the name appearing anywhere on screen, so a
    // stray mention elsewhere cannot satisfy it. The dashboard watches only
    // managed services in this scenario, so that row is the registry-owned one.
    // The row carries neither status nor port; the next steps read those from
    // the instance detail.
    session(world)
        .wait_for_screen_where(
            "an Observe row for the managed Lemonade model",
            |screen| {
                screen.lines().any(|line| {
                    e2e_cucumber::dash_screen::observe_row_cells(line, "Qwen3-0.6B").is_some()
                })
            },
            default_timeout(),
        )
        .await
        .expect("dashboard must display this scenario's real served model");
}

#[then("the managed model's instance detail shows it ready on the port it was served on")]
async fn real_lemonade_detail_ready(world: &mut E2eWorld) {
    let endpoint = world
        .endpoint
        .clone()
        .expect("the serve step sets an endpoint");
    let port = e2e_cucumber::dash_screen::endpoint_port(&endpoint)
        .unwrap_or_else(|| panic!("no port in served endpoint {endpoint}"));
    session(world)
        .wait_for_screen_where(
            "instance detail showing the model READY on the served port",
            |screen| {
                e2e_cucumber::dash_screen::instance_detail_matches(
                    screen,
                    "Qwen3-0.6B",
                    "READY",
                    &port,
                )
            },
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| {
            panic!("instance detail must show Qwen3-0.6B READY on port {port}: {e}")
        });
}

#[when("the user closes instance detail")]
async fn close_instance_detail(world: &mut E2eWorld) {
    // Esc closes the popup, but with no popup open it opens the menu. So send
    // it once and wait for the popup to be gone.
    let tui = session(world);
    tui.send("\x1b")
        .unwrap_or_else(|e| panic!("failed to send Esc: {e}"));
    tui.wait_for_screen_where(
        "instance detail closed",
        |screen| !screen.contains("Instance · "),
        default_timeout(),
    )
    .await
    .unwrap_or_else(|e| panic!("instance detail did not close: {e}"));
}

#[then("Lemonade is shown as a managed runtime installation")]
async fn real_lemonade_installed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Lemonade — managed runtime installed", default_timeout())
        .await
        .expect("managed engine Detect evidence must reach dashboard");
}

// ── Given ──────────────────────────────────────────────────────────

#[given("interactive chat uses an offline assistant")]
async fn chat_offline(world: &mut E2eWorld) {
    // The offline assistant is the CLI's own `--chat-mock` backend: consent is
    // pre-accepted and a fixed reply is returned, so the journey needs no live
    // model or network. The launch step reads this flag.
    world.chat_use_mock = true;
}

async fn setup_managed_model(
    world: &mut E2eWorld,
    options: ServiceRecordOptions,
    with_metrics: bool,
) {
    let model = "TestModel/E2E-1B";
    let mock = if with_metrics {
        MockServer::start_with_metrics(model).await
    } else {
        MockServer::start(model).await
    };
    world.endpoint = Some(mock.base_url());
    world.model_name = Some(model.to_string());
    world.mock = Some(mock);
    world.register_mock_service_with(options);
}

#[given("a managed model is still loading")]
async fn managed_model_loading(world: &mut E2eWorld) {
    setup_managed_model(
        world,
        ServiceRecordOptions {
            status: "starting",
            startup_phase: Some("loading"),
            ..ServiceRecordOptions::default()
        },
        false,
    )
    .await;
}

#[given("a managed model exposes serving metrics")]
async fn managed_model_with_metrics(world: &mut E2eWorld) {
    setup_managed_model(world, ServiceRecordOptions::default(), true).await;
}

#[given("a running managed model is available locally")]
async fn running_managed_model(world: &mut E2eWorld) {
    setup_managed_model(world, ServiceRecordOptions::default(), false).await;
}

/// Re-point "e2e-mock"'s `supervisor_pid`/`engine_pid` at a disposable `sleep`
/// child instead of this harness's own PID. `register_mock_service_with`
/// always points those fields at the harness process, which is fine for a
/// scenario that only ever reads the record — but `dash-28` goes on to run the
/// literal `rocm services stop e2e-mock --yes` the prompt prints, and that
/// command signals whatever PID the record names. Without this step it
/// signals the cucumber harness itself (SIGTERM, `KillScope::Tree`).
///
/// Writes the child's own start-time token (`supervisor_start_ticks`)
/// alongside its pid, the same pair a real launch path writes, so the stop
/// exercises `identity_state`'s start-time-verified match rather than the
/// `(None, _) => Matches` best-effort arm a record with no token falls back
/// to.
#[given("the managed model's record points at a disposable process")]
async fn managed_model_record_points_at_disposable_process(world: &mut E2eWorld) {
    let child = std::process::Command::new("sleep")
        .arg("300")
        .spawn()
        .expect("failed to spawn disposable process for dash-28's stop-command step");
    let pid = child.id();
    let supervisor_start_ticks = rocm_core::process_start_ticks(pid);
    world.recorded_process = Some(child);

    let root = world.isolated_root.as_ref().expect("no isolated root");
    let services_dir = root.path().join("data").join("services");
    let model = world.model_name.as_deref().expect("no model name set");
    let port = world.mock.as_ref().expect("no mock server running").port();
    e2e_cucumber::mock_server::write_service_record_with(
        &services_dir,
        model,
        port,
        ServiceRecordOptions {
            supervisor_pid: pid,
            engine_pid: Some(pid),
            supervisor_start_ticks,
            ..ServiceRecordOptions::default()
        },
    );
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user replays a recording that does not exist")]
async fn replay_missing_recording(world: &mut E2eWorld) {
    // Drive this under a real pseudo-terminal, not a pipe. The point of the fix is
    // fail-fast *before the terminal takeover*, and that property is unobservable
    // through a pipe: a piped `dash` can't enter the alt-screen either way (and
    // the pre-fix binary already exits non-zero through a pipe when raw-mode
    // fails, so a piped run detects nothing). Under a PTY the pre-fix binary
    // enters the alt-screen (`ESC[?1049h`) and hangs — that is the regression this
    // scenario pins.
    let missing = std::env::temp_dir().join(format!(
        "rocm-cli-e2e-no-such-recording-eai-8366-{}.ndjson",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&missing);
    let path = missing.to_string_lossy().to_string();
    disable_lemonade_discovery_by_default(world);
    let session = TuiSession::spawn(world, &["dash", "--replay", &path])
        .unwrap_or_else(|e| panic!("failed to spawn dash under a pty: {e}"));
    world.tui = Some(session);
}

#[when("the user opens the dashboard with demo data")]
async fn open_dashboard_demo(world: &mut E2eWorld) {
    // `--demo` replays a deterministic synthetic session, so the dashboard
    // populates with no GPU and no daemon — a stable, mock-tier target.
    disable_lemonade_discovery_by_default(world);
    let session = TuiSession::spawn(world, &["dash", "--demo"])
        .unwrap_or_else(|e| panic!("failed to open the dashboard: {e}"));
    world.tui = Some(session);
}

/// The one warning `dash-38`'s recording reports. Short and ASCII so it stays on
/// one screen row inside the popup and is found by a plain substring match.
const RECORDED_WARNING: &str = "amd-smi unavailable (GPU device inaccessible)";

#[when("the user opens the dashboard replaying a recording that reports a warning")]
async fn open_dashboard_replaying_warning(world: &mut E2eWorld) {
    // A snapshot with a working GPU *and* a warning: the combination where the
    // warning text used to be drawn nowhere in the dashboard. The dashboard
    // forgets its latest snapshot the moment a replay ends, and replay caps the
    // wait between entries at two seconds, so a single line would be gone before
    // the scenario looked at it. Sixty entries two seconds apart keep the
    // recording playing for two minutes, far longer than the scenario runs.
    let snapshot = |ts_us: u64| {
        format!(
            concat!(
                r#"{{"ts_us":{ts_us},"event":{{"kind":"snapshot","#,
                r#""timestamp":"2023-11-15T00:00:00Z","#,
                r#""host":{{"cpu_overall_pct":0.0,"cpu_per_core_pct":[],"memory_used_mb":0,"#,
                r#""memory_total_mb":0,"swap_used_mb":0,"swap_total_mb":0,"#,
                r#""disk_read_bps":0,"disk_write_bps":0,"net_rx_bps":0,"net_tx_bps":0}},"#,
                r#""gpus":[{{"device_id":"0","vram_used_mb":0,"vram_total_mb":1024,"#,
                r#""gpu_utilization_pct":0.0,"temperature_c":0.0,"power_w":0.0,"clock_mhz":null}}],"#,
                r#""gpu_system_info":null,"instances":[],"warnings":["{warning}"]}}}}"#,
            ),
            ts_us = ts_us,
            warning = RECORDED_WARNING
        )
    };
    let recording_text: String = (0..60u64)
        .map(|i| snapshot(1_700_000_000_000_000 + i * 2_000_000) + "\n")
        .collect();
    let root = world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .to_path_buf();
    let recording = root.join("warning-recording.ndjson");
    std::fs::write(&recording, recording_text)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", recording.display()));
    let path = recording.to_string_lossy().to_string();
    let session = TuiSession::spawn(world, &["dash", "--replay", &path])
        .unwrap_or_else(|e| panic!("failed to open the dashboard replaying a recording: {e}"));
    world.tui = Some(session);
}

#[then("the header shows that one warning is reported")]
async fn header_shows_one_warning(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("⚠ 1", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the header warning badge did not appear: {e}"));
}

#[when("the user opens the warnings")]
async fn open_warnings(world: &mut E2eWorld) {
    // `!` toggles the popup, so it is not safe to resend; the previous step
    // already waited for a frame that shows the badge, so the event loop is up.
    session(world)
        .send("!")
        .unwrap_or_else(|e| panic!("failed to open the warnings: {e}"));
}

#[then("the warning text is displayed")]
async fn warning_text_is_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen(RECORDED_WARNING, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the warning text was not shown: {e}"));
}

#[then("the warning text is no longer displayed")]
async fn warning_text_is_gone(world: &mut E2eWorld) {
    session(world)
        .wait_until_gone(RECORDED_WARNING, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the warnings popup did not close: {e}"));
}

/// The name of a child directory `dash-25` plants under the scenario's cwd;
/// `existing_folder_browser_shows_scenario_folder` asserts the browser lists
/// it by this fixed, short label rather than its full path — the browser's
/// entry list labels children by `file_name()` alone (`build_entries` in
/// `folder_browser.rs`), so this stays robust however long the host's own
/// temp directory is, unlike the browser's *header* line, which renders
/// `current_dir` verbatim in an unwrapped `Paragraph` and would silently
/// clip (and falsely fail this scenario) on a long enough temp path. The
/// space in the name is explained in dash-25's scenario comment.
const ADOPT_SCENARIO_CHILD: &str = "adopt me";

#[when("the user opens the dashboard in a scenario-owned folder")]
async fn open_dashboard_in_scenario_folder(world: &mut E2eWorld) {
    let root = world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .to_path_buf();
    let child = root.join(ADOPT_SCENARIO_CHILD);
    std::fs::create_dir_all(&child).unwrap_or_else(|e| {
        panic!(
            "failed to create scenario-owned folder {}: {e}",
            child.display()
        )
    });
    let session = TuiSession::spawn_in(world, &["dash", "--demo"], &root)
        .unwrap_or_else(|e| panic!("failed to open the dashboard in {}: {e}", root.display()));
    world.tui = Some(session);
}

/// Name of the scenario-owned directory `dash-36` (#693) spawns directly
/// into. Unlike [`ADOPT_SCENARIO_CHILD`] — a whitespace-containing *child*
/// entry listed by a browser rooted elsewhere — the space here is in this
/// directory's own name, and this directory itself is the `spawn_in` cwd.
/// Onboarding's Adopt flow seeds its browser from `std::env::current_dir()`
/// (`activate_choice` in `onboarding.rs`) with no intervening navigation, so
/// choosing "[ use this folder ]" stages an approval whose `--root` argument
/// is this exact whitespace path — the one #500's `quote_display_arg`
/// (`crates/rocm-dash-tui/src/ui/exec.rs`) is supposed to quote for display,
/// and the one dash-25 never reaches.
///
/// Kept to two short words only for readability on screen, not for wrap
/// avoidance: the approval preview's command line wraps at word boundaries
/// (`Wrap { trim: false }` in `draw_approval`) once it outgrows the popup's
/// own content width, and on a long enough host temp path it reliably does,
/// regardless of how short this constant is (PR #707 review, 2026-10-16: a
/// runner `_temp` prefix long enough to push even the bare quoted value past
/// one row). `approval_preview_quotes_whitespace_root` handles that by
/// reconstructing the popup's own wrapped body
/// (`TuiSession::popup_body_text`) rather than requiring this value to fit
/// on one physical row — see that step for why. The `--python
/// <root>/bin/python` tail is not part of what that step asserts.
const ADOPT_WHITESPACE_ROOT: &str = "a b";

#[when("the user opens the dashboard in a scenario folder whose own name has a space")]
async fn open_dashboard_in_whitespace_named_folder(world: &mut E2eWorld) {
    let root = world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .to_path_buf();
    let cwd = root.join(ADOPT_WHITESPACE_ROOT);
    std::fs::create_dir_all(&cwd).unwrap_or_else(|e| {
        panic!(
            "failed to create scenario-owned folder {}: {e}",
            cwd.display()
        )
    });
    let session = TuiSession::spawn_in(world, &["dash", "--demo"], &cwd)
        .unwrap_or_else(|e| panic!("failed to open the dashboard in {}: {e}", cwd.display()));
    world.tui = Some(session);
}

#[when("the user opens interactive chat")]
async fn open_chat(world: &mut E2eWorld) {
    let args: &[&str] = if world.chat_use_mock {
        &["chat", "--chat-mock"]
    } else {
        &["chat"]
    };
    // `rocm chat` starts the embedded daemon too, and discovery is on by default.
    if world.isolated_root.is_some() {
        disable_lemonade_discovery_by_default(world);
    }
    let session = TuiSession::spawn(world, args)
        .unwrap_or_else(|e| panic!("failed to open interactive chat: {e}"));
    world.tui = Some(session);
}

#[when("the user opens the dashboard")]
async fn open_dashboard(world: &mut E2eWorld) {
    disable_lemonade_discovery_by_default(world);
    let tui = TuiSession::spawn(world, &["dash"])
        .unwrap_or_else(|e| panic!("failed to open the dashboard: {e}"));
    world.tui = Some(tui);
}

#[when("the user opens the launcher")]
async fn open_launcher(world: &mut E2eWorld) {
    // Bare `rocm` (no subcommand) opens the launcher front door under an
    // interactive terminal — the PTY slave satisfies `interactive_terminal()`.
    disable_lemonade_discovery_by_default(world);
    let tui = TuiSession::spawn(world, &[])
        .unwrap_or_else(|e| panic!("failed to open the launcher: {e}"));
    world.tui = Some(tui);
}

#[when("the user opens the ROCm view")]
async fn open_rocm_view(world: &mut E2eWorld) {
    // Dashboard tabs are currently ordered Home, ROCm, Serving, Observe; these
    // numeric shortcuts intentionally exercise that user-visible ordering.
    session(world)
        .send("2")
        .unwrap_or_else(|e| panic!("failed to switch to the ROCm tab: {e}"));
}

#[when("the user opens the Observe view")]
async fn open_observe_view(world: &mut E2eWorld) {
    let tui = session(world);
    tui.use_detail_size()
        .unwrap_or_else(|e| panic!("failed to enlarge the dashboard: {e}"));
    // Every scenario using this step launches the dashboard and comes straight
    // here, with no assertion in between to prove the TUI is reading input yet
    // (unlike the ROCm journey, which asserts the home view first). A key
    // written that early can be swallowed before the event loop exists, so
    // repeat it until the Observe tab is actually selected — the `●` marks the
    // active chip. The step then fails only if the dashboard never gets there,
    // not if it was slow to start.
    tui.send_until("4", "● Observe", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("failed to switch to the Observe tab: {e}"));
}

#[when("the user opens the managed services overlay")]
async fn open_services_overlay(world: &mut E2eWorld) {
    // `s` opens the services overlay, but only from the Observe tab, so the
    // step before this one is load-bearing.
    //
    // Sent exactly once, never through `send_until`: once the overlay has focus
    // the *same* key stages a stop approval for the selected row
    // (`services_manager::on_key` -> `request_lifecycle`), so a second copy
    // still queued in the terminal when the title appears would put a stop
    // modal over the overlay in any scenario whose list is not empty.
    // `send_until` says so itself - idempotent keys only.
    //
    // One send is enough here because the preceding step (`the user opens the
    // Observe view`) returns only after the dashboard has *acted on* a key, so
    // the event loop is provably reading by the time this runs. That startup
    // race is the only thing the retry bought, and it is already closed. Any
    // future ordering that drops that guarantee has to re-establish it before
    // this step, not restore the retry.
    let tui = session(world);
    tui.send("s")
        .unwrap_or_else(|e| panic!("failed to send the services overlay key: {e}"));
    tui.wait_for_screen(SERVICES_OVERLAY_TITLE, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("failed to open the services overlay: {e}"));
}

#[then("the Services overlay has no managed servers")]
async fn external_services_are_read_only(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen(
            "No managed services. Start one with `rocm serve <model> --managed`.",
            default_timeout(),
        )
        .await
        .expect("discovered external models must not become lifecycle targets");
}

#[then("the Services overlay lists the managed model with its port")]
async fn services_overlay_lists_managed_model(world: &mut E2eWorld) {
    let model = world.model_name.clone().expect("no model name set");
    let port =
        e2e_cucumber::dash_screen::endpoint_port(world.endpoint.as_deref().expect("no endpoint"))
            .expect("endpoint has a port");
    let port_cell = format!(":{port}");
    session(world)
        .wait_for_screen_where(
            "a Services row for the managed model and its port",
            |screen| {
                !screen.contains("No managed services.")
                    && screen
                        .lines()
                        .any(|line| line.contains(&model) && line.contains(&port_cell))
            },
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| panic!("the managed service is not listed in the overlay: {e}"));
}

#[when("the user asks to stop the listed service")]
async fn user_stages_stop(world: &mut E2eWorld) {
    // Sent once: the key stages an approval, and a second copy would land on the
    // approval prompt rather than the list.
    session(world)
        .send("s")
        .unwrap_or_else(|e| panic!("failed to send the stop key: {e}"));
}

#[then("a stop approval prompt names the managed model and its port")]
async fn stop_approval_is_shown(world: &mut E2eWorld) {
    let model = world.model_name.clone().expect("no model name set");
    let port =
        e2e_cucumber::dash_screen::endpoint_port(world.endpoint.as_deref().expect("no endpoint"))
            .expect("endpoint has a port");
    let detail = format!("model: {model}   port: {port}");
    session(world)
        .wait_for_screen_where(
            "a stop approval for the managed service",
            |screen| {
                screen.contains("Review: stop service")
                    && screen.contains("Approve (y)")
                    && screen.contains(&detail)
            },
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| panic!("no stop approval prompt appeared: {e}"));
}

#[when("the user declines the approval prompt")]
async fn user_declines_approval(world: &mut E2eWorld) {
    // Esc declines without confirming. The prompt itself going away is the
    // proof the key was read; the overlay list stays open behind it.
    let tui = session(world);
    tui.send("\u{1b}")
        .unwrap_or_else(|e| panic!("failed to decline the approval: {e}"));
    tui.wait_until_gone("Review: stop service", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the approval prompt did not close: {e}"));
}

#[when("the user attempts to stop and restart the external models")]
async fn external_lifecycle_keys_do_nothing(world: &mut E2eWorld) {
    let tui = session(world);
    tui.send("sr").expect("attempt stop and restart");
    // One Escape closes the list only if neither key staged an approval or a
    // lifecycle console. In either modal, it dismisses that modal instead and
    // leaves the Services overlay open, which the next step then catches.
    tui.send("\u{1b}").expect("close Services list");
}

#[then("the Services overlay closes without an approval prompt or job")]
async fn external_lifecycle_keys_left_no_modal(world: &mut E2eWorld) {
    session(world)
        .wait_until_gone(SERVICES_OVERLAY_TITLE, default_timeout())
        .await
        .expect("external lifecycle keys must not stage approval or launch a job");
}

#[then("both loaded models are still displayed")]
async fn external_models_still_visible(world: &mut E2eWorld) {
    external_models_visible(world).await;
}

#[then("the overlay reports the record that is no longer running")]
async fn services_overlay_reports_past_attempts(world: &mut E2eWorld) {
    // The overlay renders only the live instances the daemon scrapes, so the
    // failed record left no trace here at all. The count is read from the
    // registry at launch, so it survives the daemon never having seen it.
    //
    // Asserted as the one note line the overlay renders, pointer included: the
    // count is only useful attached to the command that can show the record,
    // which the overlay itself cannot. Waiting for the pointer separately would
    // also pass with it rendered anywhere else on screen, or detached from the
    // count it belongs to.
    session(world)
        .wait_for_screen(
            "1 local server record(s) are no longer running - see `rocm services list --all`",
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| {
            panic!("the overlay never counted the failed record and named how to see it: {e}")
        });
}

#[when("the user closes the managed services overlay")]
async fn close_services_overlay(world: &mut E2eWorld) {
    // An open overlay eats the quit key - it closes the overlay instead - so a
    // scenario that opened one has to close it before the quit step, or the
    // dashboard is still running when that step gives up.
    //
    // What proves it closed is the overlay's *own* title going away. A panel
    // title the overlay was covering proves nothing on its own: which Observe
    // rows the overlay's rectangle covers depends on that tab's layout - the
    // no-live-data banner shifts the band down a row - so with live telemetry
    // such a title moves out from under the overlay and is visible while the
    // overlay is still up. This step would then return early and the quit key
    // would be eaten after all. `SERVICES_OVERLAY_TITLE` is drawn by the
    // overlay itself, so its absence cannot be satisfied while it is open.
    //
    // Sent exactly once, for the reason `open_services_overlay` gives: Esc past
    // the overlay is not idempotent either - on the Observe tab it opens the
    // launcher menu (`KeyAction::OpenMenu`) - and the event loop has provably
    // been reading keys since the overlay opened.
    let tui = session(world);
    tui.send("\u{1b}")
        .unwrap_or_else(|e| panic!("failed to send the services overlay close key: {e}"));
    tui.wait_until_gone(SERVICES_OVERLAY_TITLE, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("failed to close the services overlay: {e}"));
}

#[when("the user opens the Chat view")]
async fn open_chat_view(world: &mut E2eWorld) {
    // Same resend-until-it-takes rationale as `open_observe_view`: nothing
    // before this step proves the event loop is reading input yet.
    session(world)
        .send_until("5", "● Chat", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("failed to switch to the Chat tab: {e}"));
}

#[when("the user opens instance detail")]
async fn open_instance_detail(world: &mut E2eWorld) {
    // `Enter` on the Observe tab opens the selected instance's detail popup
    // (`KeyAction::OpenDetail`); the demo session always seeds at least one
    // instance, so the default selection (index 0) is always present. Enter
    // toggles `Modal::Detail` open/closed, so it is NOT safe to resend via
    // `send_until` (its own doc comment restricts that to idempotent keys) —
    // a resend while the popup is already open would immediately close it.
    // Plain `send` + `wait_for_screen` instead.
    let tui = session(world);
    // The `● Observe` marker asserted by `open_observe_view` only proves the
    // tab switch rendered — the demo replay's `InstanceDiscovered` events
    // land afterward. Sending Enter before they do finds an empty instance
    // list (`selection_len()` == 0), so `OpenDetail` is silently ignored.
    // Wait for the populated table before the (non-retryable) Enter.
    tui.wait_for_screen("Instances · AI metrics", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("instance list did not populate: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to send Enter: {e}"));
    tui.wait_for_screen("Instance · ", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("instance detail did not open: {e}"));
}

#[when("the user shrinks the terminal until the detail body overflows")]
async fn shrink_until_detail_overflows(world: &mut E2eWorld) {
    // `open_observe_view` already enlarged the terminal (`use_detail_size`)
    // before this scenario reached the detail popup, and the demo fixtures'
    // `launch_args`/`env_vars` don't overflow the args/env panes at *that*
    // size — this is the terminal size small enough to force it relative to
    // the size the scenario is actually at, not relative to the true default.
    let tui = session(world);
    tui.use_overflow_size()
        .unwrap_or_else(|e| panic!("failed to shrink the dashboard: {e}"));
    // The resize is synchronous in the emulator but the app only learns of it
    // asynchronously via SIGWINCH, so this step does not itself prove a
    // redraw at the new geometry happened — `"Instance · "` was already on
    // screen before the resize (see `open_instance_detail`), so waiting on it
    // here is satisfied immediately regardless of whether the app redrew.
    // The `Then` step's own `wait_for_screen` on the scroll hint is what
    // actually gates on the post-resize render.
}

#[when("the user opens the services manager")]
async fn open_services_manager(world: &mut E2eWorld) {
    // Bound to `s` only on the Observe tab (`OpenServices`) — a manager opened
    // from a non-domain tab, which is exactly the case
    // `should_pane_back_out`'s doc comment calls out as needing Esc to close it.
    session(world)
        .send("s")
        .unwrap_or_else(|e| panic!("failed to open the services manager: {e}"));
}

#[when("the user presses Escape")]
async fn press_escape(world: &mut E2eWorld) {
    session(world)
        .send("\x1b")
        .unwrap_or_else(|e| panic!("failed to send Escape: {e}"));
}

#[when("the user opens dashboard help")]
async fn open_dashboard_help(world: &mut E2eWorld) {
    session(world)
        .send("?")
        .unwrap_or_else(|e| panic!("failed to open dashboard help: {e}"));
}

#[when("the user closes dashboard help")]
async fn close_dashboard_help(world: &mut E2eWorld) {
    session(world)
        .send("?")
        .unwrap_or_else(|e| panic!("failed to close dashboard help: {e}"));
}

#[when("the user opens the command palette")]
async fn open_command_palette(world: &mut E2eWorld) {
    session(world)
        .send(":")
        .unwrap_or_else(|e| panic!("failed to open the command palette: {e}"));
}

#[when("the user opens the theme picker")]
async fn open_theme_picker(world: &mut E2eWorld) {
    let tui = session(world);
    // `t` toggles `Modal::ThemePicker` open/closed, so it is NOT safe to resend
    // via `send_until` (its own doc comment restricts that to idempotent
    // keys) — a resend after the picker is already open would immediately
    // close it. Nothing before this step proves the event loop is reading
    // input yet, so wait for the Home tab's readiness marker before the
    // (non-retryable) `t`, same rationale as `open_instance_detail`.
    tui.wait_for_screen("Updates", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("dashboard home view did not become ready: {e}"));
    tui.send("t")
        .unwrap_or_else(|e| panic!("failed to open the theme picker: {e}"));
    tui.wait_for_screen(
        "Theme — j/k select, Enter apply, Esc cancel",
        default_timeout(),
    )
    .await
    .unwrap_or_else(|e| panic!("theme picker did not open: {e}"));
}

#[when("the user chooses Serving")]
async fn choose_serving(world: &mut E2eWorld) {
    let tui = session(world);
    // The palette initially selects Home; Serving is the third destination, so
    // two downward moves intentionally assert the current destination ordering.
    tui.send("jj")
        .unwrap_or_else(|e| panic!("failed to select Serving: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to open Serving: {e}"));
}

#[when("the user opens onboarding setup")]
async fn open_onboarding_setup(world: &mut E2eWorld) {
    // `n` opens the onboarding wizard from the Observe tab
    // (`KeyAction::OpenOnboarding`). Same resend-until-it-takes rationale as
    // `open_observe_view`: nothing before this proves the event loop is
    // reading input yet. The wizard's panel title is step-independent, so it
    // is a safe marker regardless of which step renders first.
    session(world)
        .send_until("n", "Welcome to ROCm — first-run setup", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("failed to open onboarding setup: {e}"));
}

#[when("the user continues past the onboarding welcome screen")]
async fn continue_past_onboarding_welcome(world: &mut E2eWorld) {
    session(world)
        .send("\r")
        .unwrap_or_else(|e| panic!("failed to continue past the welcome screen: {e}"));
}

#[when("the user chooses to install the ROCm SDK")]
async fn choose_install_rocm_sdk(world: &mut E2eWorld) {
    // "Install ROCm SDK (pip)" is the choose-menu's default (first) entry, so
    // confirming needs no prior navigation keys.
    session(world)
        .send("\r")
        .unwrap_or_else(|e| panic!("failed to choose Install ROCm SDK: {e}"));
}

#[when("the user chooses to adopt an existing folder")]
async fn choose_adopt_existing_folder(world: &mut E2eWorld) {
    // "Adopt an existing ROCm folder" is the choose-menu's second entry, one
    // `Down` past the default "Install ROCm SDK (pip)".
    session(world)
        .send("j\r")
        .unwrap_or_else(|e| panic!("failed to choose Adopt an existing folder: {e}"));
}

#[when("the user browses for an install folder")]
async fn browse_for_install_folder(world: &mut E2eWorld) {
    session(world)
        .send("\t")
        .unwrap_or_else(|e| panic!("failed to open the install-folder browser: {e}"));
}

#[when("the user chooses the current folder")]
async fn choose_current_folder(world: &mut E2eWorld) {
    // "[ use this folder ]" is the browser's first, already-selected entry.
    session(world)
        .send("\r")
        .unwrap_or_else(|e| panic!("failed to choose the current folder: {e}"));
}

#[when("the user closes onboarding setup")]
async fn close_onboarding_setup(world: &mut E2eWorld) {
    // Two `Esc`: the first backs out of whichever sub-view currently has
    // focus — the SDK Configure channel/pin sub-view (reached via Install
    // SDK; its own folder browser, opened with `Tab`, is always closed by
    // the time this step runs, as in dash-23) or the Adopt-path's own folder
    // browser — the second closes the wizard entirely; it owns every key
    // while open (see `draw_onboarding`'s doc comment), so `q` cannot reach
    // the dashboard until it is gone. Each sub-view leaves its own marker on
    // screen (the Configure sub-view's "Tab browse folder" footer hint, or
    // the Adopt browser's own popup title); wait for one of them to actually
    // render — a single snapshot can race a step immediately before this
    // one that is still mid-repaint — then wait for that marker to clear
    // before the second `Esc`, or the two `Esc` bytes can land in the same
    // PTY read and coalesce into one `KeyCode::Esc` (crossterm reads
    // back-to-back ESC bytes as a single Esc key), which only backs out of
    // the sub-view and leaves the wizard open. Configure's own folder
    // browser ("Pick an install folder") is rejected outright rather than
    // silently waited past: closing it is the caller's job, not this
    // step's, so seeing it here means a scenario called this step too
    // early.
    const CONFIGURE_FOOTER: &str = "Tab browse folder";
    const ADOPT_BROWSER_TITLE: &str = "Pick an existing ROCm folder";
    const CONFIGURE_BROWSER_TITLE: &str = "Pick an install folder";
    let tui = session(world);
    // Recorded from inside the predicate below, not re-read from a second
    // `screen_text()` call afterward — the screen can keep moving between
    // the predicate matching and a later read, so only the frame the
    // predicate actually saw can be trusted to name which marker matched.
    let mut open_sub_view_marker: Option<&'static str> = None;
    tui.wait_for_screen_where(
        &format!("{CONFIGURE_FOOTER:?} or {ADOPT_BROWSER_TITLE:?} to appear"),
        |screen| {
            assert!(
                !screen.contains(CONFIGURE_BROWSER_TITLE),
                "the install-folder browser is still open; close it before calling this step:\n{screen}"
            );
            open_sub_view_marker = if screen.contains(ADOPT_BROWSER_TITLE) {
                Some(ADOPT_BROWSER_TITLE)
            } else if screen.contains(CONFIGURE_FOOTER) {
                Some(CONFIGURE_FOOTER)
            } else {
                None
            };
            open_sub_view_marker.is_some()
        },
        default_timeout(),
    )
    .await
    .unwrap_or_else(|e| {
        panic!("no sub-view is open, so there is nothing to close before the wizard itself: {e}")
    });
    let open_sub_view_marker =
        open_sub_view_marker.expect("wait_for_screen_where succeeded without recording a marker");
    tui.send("\u{1b}")
        .unwrap_or_else(|e| panic!("failed to leave the open sub-view: {e}"));
    tui.wait_until_gone(open_sub_view_marker, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("sub-view did not close: {e}"));
    tui.send("\u{1b}")
        .unwrap_or_else(|e| panic!("failed to close onboarding: {e}"));
    tui.wait_until_gone("Welcome to ROCm — first-run setup", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("onboarding wizard did not close: {e}"));
}

/// The approval modal's own popup title (`format!("Review: {}", req.title)`
/// in `draw_approval`, with `req.title` set from `OnboardingChoice::title()`
/// in `stage_approval`), used by `deny_adopt_approval_and_close_onboarding`
/// both to wait for the modal and as proof it closed.
const ADOPT_APPROVAL_TITLE: &str = "Review: adopt existing ROCm folder";

#[when("the user denies the adopt approval and closes onboarding setup")]
async fn deny_adopt_approval_and_close_onboarding(world: &mut E2eWorld) {
    // `close_onboarding_setup` above closes the two sub-views it was written
    // for (Configure, and the Adopt folder browser), but a staged approval is
    // a third, later sub-view neither of its markers matches — this scenario
    // (dash-36, #693) is the first to reach one. Same double-Esc shape:
    // the first Esc denies the approval (`on_key`'s approval-focused branch
    // treats Esc the same as `n`/`q` — `ApprovalVerdict::Cancel`), landing
    // back on the bare Choose step with nothing else open; the second Esc
    // then closes the wizard itself from there (`OnboardingStep::Choose`'s
    // own Esc arm). Waiting for the approval's own title to clear between
    // the two keys avoids the same coalescing risk `close_onboarding_setup`
    // guards against: two bare ESC bytes arriving in the same PTY read
    // collapse into a single `KeyCode::Esc`, which would only dismiss the
    // approval and leave the wizard open.
    let tui = session(world);
    tui.send("\u{1b}")
        .unwrap_or_else(|e| panic!("failed to deny the adopt approval: {e}"));
    tui.wait_until_gone(ADOPT_APPROVAL_TITLE, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the adopt approval prompt did not close: {e}"));
    tui.send("\u{1b}")
        .unwrap_or_else(|e| panic!("failed to close onboarding: {e}"));
    tui.wait_until_gone("Welcome to ROCm — first-run setup", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("onboarding wizard did not close: {e}"));
}

#[when("the user accepts the local endpoint")]
async fn accept_local_endpoint(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("Your request only leaves this machine", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("local endpoint consent did not appear: {e}"));
    tui.send("y")
        .unwrap_or_else(|e| panic!("failed to accept local endpoint: {e}"));
}

#[when("the user sends a message to the managed model")]
async fn send_managed_model_message(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("No messages yet.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("chat surface never became ready: {e}"));
    // No `i` here: accepting local-endpoint consent (`accept_chat_consent`) already
    // focused the input, so an extra `i` would be typed as a literal character
    // instead of a focus gesture (contrast the offline path in
    // `send_gpu_message`, whose `--chat-mock` consent does not focus the input).
    tui.send(MANAGED_MODEL_PROMPT)
        .unwrap_or_else(|e| panic!("failed to type the chat message: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to submit the chat message: {e}"));
}

#[when("the user sends a message about GPU health")]
async fn send_gpu_message(world: &mut E2eWorld) {
    let tui = session(world);
    // Wait for the accepted, empty chat surface before typing so the input is
    // ready to receive focus.
    tui.wait_for_screen("No messages yet.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("chat surface never became ready: {e}"));
    // `i` focuses the input; then the message, then Enter to submit.
    tui.send("i")
        .unwrap_or_else(|e| panic!("failed to focus the chat input: {e}"));
    tui.send("how is gpu-2 doing")
        .unwrap_or_else(|e| panic!("failed to type the chat message: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to submit the chat message: {e}"));
}

/// Counterpart of `send_gpu_message` for a chat input that is already
/// focused — `dash-29`'s `/quit`-then-decline round trip (`try_quit_chat`
/// presses `i` to focus before typing `/quit`) leaves `chat_focused` set,
/// since nothing on the decline path blurs it. Sending `send_gpu_message`'s
/// own leading `i` there would be read as a literal character rather than a
/// focus gesture (`actions.rs`'s `chat.focused` branch maps `Char('i')` to
/// `ChatInput('i')`, not `ChatFocus`), corrupting the message into
/// `"ihow is gpu-2 doing"` — not caught by this scenario only because the
/// mock replies to anything that isn't the approval-trigger phrase. Mirrors
/// `send_managed_model_message`'s own already-focused pattern.
#[when("the user sends a message about GPU health without refocusing the input")]
async fn send_gpu_message_already_focused(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("No messages yet.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("chat surface never became ready: {e}"));
    tui.send("how is gpu-2 doing")
        .unwrap_or_else(|e| panic!("failed to type the chat message: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to submit the chat message: {e}"));
}

#[when("the user sends a message that triggers a tool approval")]
async fn send_approval_trigger_message(world: &mut E2eWorld) {
    let tui = session(world);
    // Wait for the accepted, empty chat surface before typing so the input is
    // ready to receive focus.
    tui.wait_for_screen("No messages yet.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("chat surface never became ready: {e}"));
    // `i` focuses the input; then the message, then Enter to submit. The
    // phrase must match `MockAgentClient`'s trigger ("install the sdk") without
    // colliding with `send_gpu_message`'s "how is gpu-2 doing".
    tui.send("i")
        .unwrap_or_else(|e| panic!("failed to focus the chat input: {e}"));
    tui.send("please install the sdk")
        .unwrap_or_else(|e| panic!("failed to type the chat message: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to submit the chat message: {e}"));
}

#[when("the user confirms the approval prompt without moving the cursor")]
async fn confirm_approval_without_moving(world: &mut E2eWorld) {
    session(world)
        .send("\r")
        .unwrap_or_else(|e| panic!("failed to press Enter on the approval prompt: {e}"));
}

async fn quit_tui(world: &mut E2eWorld, surface: &str) {
    session(world)
        .quit_and_wait(default_timeout())
        .await
        .unwrap_or_else(|e| panic!("{surface} did not exit cleanly: {e}"));
}

#[when("the user quits the dashboard")]
async fn quit_dashboard(world: &mut E2eWorld) {
    quit_tui(world, "the dashboard").await;
}

/// Send the plain quit gesture and nothing else — unlike `quit_dashboard`,
/// this does not wait for (or resolve) the quit-confirm prompt that may open
/// (issue #145), since the point of this step is to observe that prompt.
#[when("the user tries to quit the dashboard")]
async fn try_quit_dashboard(world: &mut E2eWorld) {
    session(world)
        .send("q")
        .unwrap_or_else(|e| panic!("failed to send the quit key: {e}"));
}

/// Deliberately bypasses `quit_and_wait`: that helper's own retry loop
/// answers "y" to any quit-confirm prompt it finds on screen, so it would
/// silently paper over a regression that opened the prompt here. `q` with
/// nothing serving must exit the process outright on the first keystroke; if
/// the prompt opened instead, nothing resolves it and this times out rather
/// than passing.
#[then("the dashboard exits immediately with no quit confirmation")]
async fn dashboard_exits_without_confirmation(world: &mut E2eWorld) {
    session(world)
        .wait_for_exit(default_timeout())
        .await
        .unwrap_or_else(|e| {
            panic!(
                "`q` with nothing serving must exit immediately, with no \
                 quit-confirm prompt: {e}"
            )
        });
}

/// Chat counterpart of `try_quit_dashboard`: `/quit` shares the exact same
/// `request_quit` gate as the dashboard's `q` and the launcher's own `q`
/// (issue #145) — `rocm chat` is the same dashboard with the Chat tab focused
/// (`run_chat` calls the same `run_async`) — but had no scenario coverage of
/// its own before this. Sends `/quit` and nothing else: unlike
/// `quit_interactive_chat`, this does not resolve the prompt that may open,
/// since the point of this step is to observe it.
#[when("the user tries to quit chat")]
async fn try_quit_chat(world: &mut E2eWorld) {
    let tui = session(world);
    // Wait for the accepted, empty chat surface before typing so the input is
    // ready to receive focus — mirrors `send_gpu_message`'s own wait.
    tui.wait_for_screen("No messages yet.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("chat surface never became ready: {e}"));
    // `i` focuses the input (a fresh session starts unfocused — slash commands
    // are plain text input, same as a chat message); then the command, then
    // Enter to submit it (`handle_slash_command` routes `/quit` to the shared
    // gate, same as a plain `q` keypress on the dashboard).
    tui.send("i")
        .unwrap_or_else(|e| panic!("failed to focus the chat input: {e}"));
    tui.send("/quit")
        .unwrap_or_else(|e| panic!("failed to type /quit: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to submit /quit: {e}"));
}

#[then("the dashboard asks whether to quit while a model is still being served")]
async fn quit_confirm_is_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen(QUIT_CONFIRM_MARKER, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("quit-confirm prompt never appeared: {e}"));
}

/// Proves the prompt's remediation line itself (not just that some prompt
/// appeared) names a command that is real: the fixture's own managed-service
/// record always carries `"service_id": "e2e-mock"` (`mock_server.rs`'s
/// `write_service_record_with`), so this checks for the literal, fillable
/// `rocm services stop e2e-mock --yes` — not a `<id>` placeholder or a bare
/// mention of the model name — directly answering AGENTS.md §3's rule that a
/// remediation command must be asserted, not just its wording.
///
/// The command sits on its own screen row (`ui::quit_confirm_body` pushes it
/// as a separate line from the model-name row above it) rather than trailing
/// the model name on one combined line, so at this PTY's width it is not
/// expected to word-wrap. Whitespace is still
/// collapsed to single spaces before matching — this also merges the row
/// boundary between the model-name line and the command line into one
/// contiguous string — so the match is robust to that line break (and to any
/// wrap that does occur) without depending on exact column positions.
#[then("the quit prompt shows the real stop command for the managed model")]
async fn quit_confirm_shows_real_stop_command(world: &mut E2eWorld) {
    let expected = "rocm services stop e2e-mock --yes";
    session(world)
        .wait_for_screen_where(
            &format!("the quit prompt shows {expected:?}"),
            |screen| {
                screen
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .contains(expected)
            },
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| panic!("{e}"));
}

/// Issue #145's actual stop action, not just the warning: `s` on the
/// dashboard's quit-confirm prompt is meant to close it and open the Services
/// overlay instead of resolving an Approve/Deny/Cancel verdict
/// (`AppState::quit_confirm_open_services_key`). Sent exactly once — the
/// prompt is already confirmed open by the preceding `Then` step, so there is
/// no startup race to retry through, the same reasoning `open_services_overlay`
/// gives for its own single send.
#[when("the user presses s to manage the serving model")]
async fn quit_confirm_presses_s_to_manage(world: &mut E2eWorld) {
    session(world)
        .send("s")
        .unwrap_or_else(|e| panic!("failed to send the manage-service key: {e}"));
}

/// Proves the hint that makes `s` discoverable is actually on screen before
/// it's pressed, not just that pressing it happens to work
/// (`services_overlay_opens_showing_managed_model` only proves the latter).
/// `quit_confirm_request`'s own `offer_stop_action` doc comment is what gates
/// this to the dashboard's copy of the prompt — the launcher's copy shares
/// the same body text but passes `false` and must never show it (see
/// `rocm_dash_tui::ui::launcher::tests::launcher_renders_the_quit_confirm_prompt_when_open`,
/// the unit-level counterpart for that half).
#[then("the quit prompt offers to open Services for the serving model")]
async fn quit_confirm_offers_services_hint(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Press s to open Services", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("quit prompt never offered the Services hint: {e}"));
}

/// Proves `s` actually did something, not just that it was sent: the
/// quit-confirm prompt's own marker must be gone (it closed, rather than
/// merely sitting underneath whatever rendered), the Services overlay's own
/// title must be on screen (`SERVICES_OVERLAY_TITLE`, drawn only by that
/// overlay), and the overlay's own row must show the fixture's real service
/// id (`e2e-mock` — `draw_services_manager` renders `r.id` as its own
/// column, `services_manager.rs`).
///
/// The id, not the model name: the model name is already on screen from the
/// earlier `the managed model is displayed` step and would still read true
/// from the dimmed Observe background behind a centered modal that doesn't
/// fully cover it (see `close_services_overlay`'s own comment on exactly
/// this), so a model-name match alone wouldn't prove the overlay's own
/// content, only that it rendered on top of something that already had it.
#[then("the services overlay opens showing the managed model")]
async fn services_overlay_opens_showing_managed_model(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_until_gone(QUIT_CONFIRM_MARKER, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the quit prompt never closed after `s`: {e}"));
    tui.wait_for_screen(SERVICES_OVERLAY_TITLE, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("`s` never opened the services overlay: {e}"));
    tui.wait_for_screen("e2e-mock", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the services overlay did not show its own row: {e}"));
}

/// Launcher counterpart of `try_quit_dashboard`: the launcher's own `q`/Esc
/// is a second, independent quit entry point (issue #145) and must not wait
/// for (or resolve) the prompt it may open, since the point of this step is
/// to observe that prompt.
#[when("the user tries to quit the launcher")]
async fn try_quit_launcher(world: &mut E2eWorld) {
    session(world)
        .send("q")
        .unwrap_or_else(|e| panic!("failed to send the quit key: {e}"));
}

/// Launcher counterpart of `quit_confirm_is_displayed`: same marker, same
/// prompt (`ui::launcher::draw`'s own quit_confirm arm renders the identical
/// `ui::quit_confirm_body`), reached from the launcher's own event loop
/// instead of the dashboard's.
#[then("the launcher asks whether to quit while a model is still being served")]
async fn launcher_quit_confirm_is_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen(QUIT_CONFIRM_MARKER, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("quit-confirm prompt never appeared: {e}"));
}

#[when("the user declines the quit prompt")]
async fn decline_quit_prompt(world: &mut E2eWorld) {
    session(world)
        .send("n")
        .unwrap_or_else(|e| panic!("failed to decline the quit prompt: {e}"));
    // The prompt's own body names the serving model (`quit_confirm_body` in
    // `ui/mod.rs`), so the very next assertion — "the managed model is
    // displayed" (dash-26) or "the launcher shows the model serving"
    // (dash-27; a different step, same risk) — could pass on the prompt's
    // own text even if `n` did nothing and the prompt never actually closed.
    // Wait for it to actually leave the screen so that assertion proves the
    // decline worked, not just that the model's name is on screen somewhere.
    session(world)
        .wait_until_gone(QUIT_CONFIRM_MARKER, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("quit-confirm prompt never closed after declining: {e}"));
}

/// Whether `services list --json` currently lists the scenario's model as
/// `ready`/`running` — the same statuses `Instance::status.is_serving()`
/// treats as live on the TUI side. Shared by the "still listed as running"
/// and "no longer listed as running" `Then` steps below so the two assert the
/// literal opposite of the same live read rather than each re-deriving it
/// (and risking drifting on what "running" means).
///
/// `--json` (no `--all`) rather than the plain `--all` table: `--all` also
/// keeps stopped/failed records around, so a substring match against it would
/// pass even if the service had actually stopped and merely left its record
/// behind.
fn managed_model_is_listed_as_running(world: &E2eWorld) -> (bool, String) {
    let (stdout, _, _) = crate::run_rocm(world, &["services", "list", "--json"]);
    let model = world
        .model_name
        .as_deref()
        .expect("no model name set")
        .to_lowercase();
    let records: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("services list --json did not parse: {e}\n{stdout}"));
    let found = records.as_array().into_iter().flatten().any(|record| {
        let names_model = [record.get("model_ref"), record.get("canonical_model_id")]
            .into_iter()
            .flatten()
            .filter_map(serde_json::Value::as_str)
            .any(|v| v.to_lowercase().contains(&model));
        let is_running = record
            .get("status")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|s| matches!(s, "ready" | "running"));
        names_model && is_running
    });
    (found, stdout)
}

/// Proves the prompt's own claim ("it may still be running in the background
/// after you quit") against actual state, not just its wording (AGENTS.md §3) — a
/// plain CLI check against the managed-service registry, run after the TUI
/// process has already exited (or, for the declined-prompt assertion earlier
/// in the same scenario, while it's still open).
#[then("the managed model is still listed as running")]
async fn managed_model_still_listed_as_running(world: &mut E2eWorld) {
    let (found, stdout) = managed_model_is_listed_as_running(world);
    assert!(
        found,
        "model not listed as ready/running — quitting (or declining) must not stop it:\n{stdout}"
    );
}

/// Proves the quit prompt's own remediation line — the real, fillable
/// `rocm services stop <id> --yes` command (`quit_confirm_shows_real_stop_command`
/// pins the string) — actually clears the condition it warns about, not just
/// that the text is right (AGENTS.md §3). Runs the exact command a user would
/// copy from the prompt, as a plain CLI invocation independent of the TUI
/// under test (the fixture's own managed-service record always carries
/// `"service_id": "e2e-mock"` — `mock_server.rs`'s `write_service_record_with`).
#[when("the user runs the quit prompt's stop command")]
async fn run_quit_prompt_stop_command(world: &mut E2eWorld) {
    crate::run_rocm_ok(world, &["services", "stop", "e2e-mock", "--yes"]);
}

/// The negation of `managed_model_still_listed_as_running`: proves the stop
/// command above actually took effect on the registry `rocm services` itself
/// reads, not merely that the CLI invocation exited zero.
#[then("the managed model is no longer listed as running")]
async fn managed_model_no_longer_listed_as_running(world: &mut E2eWorld) {
    let (found, stdout) = managed_model_is_listed_as_running(world);
    assert!(
        !found,
        "model still listed as ready/running after running the prompt's own \
         stop command — the remediation it prints does not actually work:\n{stdout}"
    );
}

/// Proves the stop command signalled the real process, not only the
/// registry record (AGENTS.md §3: "actually clear the condition"). Without
/// this, a `rocm services stop` that merely rewrote the record's status to
/// `stopped` without signalling anything would still leave
/// `managed_model_no_longer_listed_as_running` above green — it only reads
/// the registry, never the process `dash-28`'s own
/// "the managed model's record points at a disposable process" step pointed
/// the record's `supervisor_pid`/`engine_pid` at.
#[then("the disposable process has exited")]
async fn disposable_process_has_exited(world: &mut E2eWorld) {
    let child = world
        .recorded_process
        .as_mut()
        .expect("no disposable process recorded for this scenario");
    let mut exited = false;
    for _ in 0..20 {
        if child.try_wait().ok().flatten().is_some() {
            exited = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        exited,
        "the stop command must actually terminate the process it names, not \
         just rewrite the registry record to stopped"
    );
}

#[when("the user quits interactive chat")]
async fn quit_interactive_chat(world: &mut E2eWorld) {
    quit_tui(world, "interactive chat").await;
}

#[when("the user quits the launcher")]
async fn quit_launcher(world: &mut E2eWorld) {
    quit_tui(world, "the launcher").await;
}

/// Deliver a termination signal to the TUI under test and wait for it to exit,
/// stashing the observed exit code for the `Then` steps. Shared by the
/// SIGTERM/SIGINT `When` steps so the two cannot drift.
///
/// Deliberately not named for the dashboard: the process under test is the
/// launcher in the hub round-trip scenario, and the signal handling being
/// asserted is process-wide, not dashboard-specific.
async fn signal_tui(world: &mut E2eWorld, signal: TermSignal) {
    session(world)
        .deliver_signal_and_wait(signal, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the process under test did not exit after {signal:?}: {e}"));
}

#[when("the user opens the dashboard from the launcher")]
async fn open_dashboard_from_launcher(world: &mut E2eWorld) {
    let tui = session(world);
    // Sync on the launcher front door before sending a key, so `d` is not
    // swallowed before the launcher's synchronous event loop is reading input.
    tui.wait_for_screen("Set up this system", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the launcher front door never appeared: {e}"));
    // `d` escalates straight into the full dashboard (LauncherChoice::OpenDashboard),
    // which builds and then, on quit, drops its own Tokio runtime.
    tui.send("d")
        .unwrap_or_else(|e| panic!("failed to open the dashboard from the launcher: {e}"));
    tui.wait_for_screen("Updates", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the dashboard did not open from the launcher: {e}"));
}

#[when("the user quits back to the launcher")]
async fn quit_back_to_launcher(world: &mut E2eWorld) {
    let tui = session(world);
    // `q` quits the dashboard; the hub loop drops the session runtime and
    // redraws the launcher front door — the exact "back at the menu after a
    // session" state where a per-session signal watcher would have gone deaf.
    tui.send("q")
        .unwrap_or_else(|e| panic!("failed to quit the dashboard: {e}"));
    // A launcher-hosted session (dash-30) must exit on this one `q` with no
    // confirm prompt of its own, even while a model is serving — only the
    // launcher's own gate should ever show it. If `AppState::launcher_hosted`
    // failed to suppress the dashboard's own gate, this session would open
    // its own quit-confirm prompt instead of exiting, and the launcher front
    // door below would never come back — the panic message names that
    // specific failure mode so a timeout here reads as "the double-prompt
    // regression" rather than an unexplained hang.
    tui.wait_for_screen("Set up this system", default_timeout())
        .await
        .unwrap_or_else(|e| {
            panic!(
                "the launcher front door did not return after the session — \
                 if a quit-confirm prompt is still showing instead, \
                 `AppState::launcher_hosted` failed to suppress this \
                 session's own gate (issue #145's double-prompt fix): {e}"
            )
        });
}

#[when("the user opens chat from the launcher")]
async fn open_chat_from_launcher(world: &mut E2eWorld) {
    let tui = session(world);
    // Sync on the launcher front door before sending keys, same rationale as
    // `open_dashboard_from_launcher`.
    tui.wait_for_screen("Set up this system", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the launcher front door never appeared: {e}"));
    // Chat has no single-key shortcut like `d` (`OpenDashboard`'s) — it's the
    // 4th row (`LauncherChoice::Chat`, after Set up/Serve/Diagnose), reached
    // by moving down three times from the default (Set up) selection, then
    // `Enter` (`choice_for(sel)`).
    tui.send("jjj")
        .unwrap_or_else(|e| panic!("failed to select Chat in the launcher: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to open chat from the launcher: {e}"));
    // Unlike every other chat-opening step in this file, this one can't pass
    // `--chat-mock`: the bare launcher's own top-level dispatch hardcodes
    // `run_launcher(false)` (`main.rs`), so there is no flag to carry it
    // through to the `Chat` escalation. The fixture's mock HTTP endpoint
    // (`a running managed model is available locally`) is still real enough
    // for chat's own auto-detection to find, so it always offers the
    // one-time "use this endpoint?" consent prompt (`ChatConsent::Pending`)
    // first — accept it before the empty chat surface appears.
    tui.wait_for_screen("Chat — use this endpoint?", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("chat's endpoint-consent prompt never appeared: {e}"));
    tui.send("y")
        .unwrap_or_else(|e| panic!("failed to accept the detected chat endpoint: {e}"));
    tui.wait_for_screen("No messages yet.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("chat did not open from the launcher: {e}"));
}

#[when("the user quits chat back to the launcher")]
async fn quit_chat_back_to_launcher(world: &mut E2eWorld) {
    let tui = session(world);
    // Same `/quit` route `try_quit_chat` uses (`handle_slash_command` routes
    // it to the shared `request_quit` gate), but unlike that step this one
    // resolves straight through: a launcher-hosted chat session
    // (`run_chat`'s own `launcher_hosted`) must exit this inner loop
    // unconditionally rather than open a quit-confirm prompt of its own —
    // see `request_quit`'s doc comment.
    //
    // No `i` here, unlike `try_quit_chat`: that step's own comment notes "a
    // fresh session starts unfocused", but `accept_chat_consent` (`app/mod.rs`)
    // sets `chat_focused = true` as part of accepting the endpoint prompt this
    // step's caller (`open_chat_from_launcher`) already sent `y` to, so the
    // input here is already focused. Sending `i` first — as a first attempt
    // at this step did — typed a literal leading `i` into the box instead of
    // focusing it, submitting "i/quit" as a chat message rather than running
    // the slash command at all.
    tui.send("/quit")
        .unwrap_or_else(|e| panic!("failed to type /quit: {e}"));
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to submit /quit: {e}"));
    // Same failure-mode framing as `quit_back_to_launcher`: if `run_chat`
    // failed to propagate `launcher_hosted` into `AppState`, this session
    // would open its own quit-confirm prompt instead of exiting, and the
    // launcher front door below would never come back.
    tui.wait_for_screen("Set up this system", default_timeout())
        .await
        .unwrap_or_else(|e| {
            panic!(
                "the launcher front door did not return after the chat session — \
                 if a quit-confirm prompt is still showing instead, `run_chat` \
                 failed to propagate `launcher_hosted` into this session's \
                 `AppState` (issue #145's double-prompt fix): {e}"
            )
        });
}

#[when("the launcher receives a SIGTERM")]
async fn launcher_receives_sigterm(world: &mut E2eWorld) {
    signal_tui(world, TermSignal::Term).await;
}

#[when("the dashboard receives a SIGTERM")]
async fn dashboard_receives_sigterm(world: &mut E2eWorld) {
    signal_tui(world, TermSignal::Term).await;
}

#[when("the dashboard receives a SIGINT")]
async fn dashboard_receives_sigint(world: &mut E2eWorld) {
    signal_tui(world, TermSignal::Int).await;
}

/// Type a literal Ctrl-C at the running TUI and wait for it to exit. Shared by
/// the dashboard and launcher wordings, which press the same key at the two
/// separate key loops the process runs.
async fn press_ctrl_c(world: &mut E2eWorld, subject: &str) {
    session(world)
        .press_ctrl_c_and_wait(default_timeout())
        .await
        .unwrap_or_else(|e| panic!("{subject} did not exit after Ctrl-C: {e}"));
}

#[when("the user presses Ctrl-C in the dashboard")]
async fn dashboard_ctrl_c(world: &mut E2eWorld) {
    press_ctrl_c(world, "the dashboard").await;
}

#[when("the user presses Ctrl-C in the launcher")]
async fn launcher_ctrl_c(world: &mut E2eWorld) {
    press_ctrl_c(world, "the launcher").await;
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the dashboard is refused before taking over the terminal")]
async fn dashboard_refused_before_takeover(world: &mut E2eWorld) {
    // Fail-fast contract: the child must exit non-zero *promptly*. Under a PTY the
    // pre-fix binary takes over the terminal and hangs, so `wait_for_refusal`
    // times out there — the timeout IS the regression, not a flake.
    let tui = session(world);
    tui.wait_for_refusal(default_timeout())
        .await
        .unwrap_or_else(|e| panic!("{e}"));
    // And it never entered the alt-screen: the refusal happened before the
    // dashboard could take over the terminal.
    assert!(
        !tui.in_alternate_screen(),
        "dash entered the alt-screen before refusing a missing replay file:\n{}",
        tui.screen_text(),
    );
}

#[then("the user is told the replay file was not found")]
async fn told_replay_file_not_found(world: &mut E2eWorld) {
    // Read the screen from the PTY session itself rather than stashing it in
    // `world.cli_output`, which carries piped stdout for the non-PTY steps.
    // Draining first lets the reader thread commit the final buffered frame, so
    // this does not race the PTY being drained after the child exits.
    let screen = session(world).drain_final_screen().await;
    assert!(
        screen.to_lowercase().contains("replay file not found"),
        "expected a clear 'replay file not found' error on screen, got:\n{screen}"
    );
}

#[then("the dashboard home view is displayed")]
async fn home_view_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    // The Home tab's summary cards (Running / Health / Updates) are drawn at any
    // size; the wider "GPU UTILIZATION" hero is not, so assert on the cards.
    tui.wait_for_screen("Updates", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the home view did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("Running") && screen.contains("Health"),
        "home summary cards missing:\n{screen}"
    );
}

#[then("the dashboard reports live GPU telemetry")]
async fn gpu_telemetry_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    tui.use_wide_size()
        .unwrap_or_else(|e| panic!("failed to enlarge the dashboard: {e}"));

    tui.wait_for_screen_where(
        "a nonzero GPU count with a known model",
        |screen| {
            let nonzero_gpu_count = screen
                .lines()
                .any(|line| line.contains("GPUs · ") && !line.contains("GPUs · 0"));
            nonzero_gpu_count && !screen.contains("Unknown GPU")
        },
        default_timeout(),
    )
    .await
    .unwrap_or_else(|e| panic!("GPU telemetry did not appear: {e}"));
}

#[when("the user selects the ROCm diagnose action")]
async fn select_rocm_diagnose_action(world: &mut E2eWorld) {
    // The ROCm actions list opens on "Set up / Install ROCm"; "Diagnose & fix"
    // is two rows down, after "Check for updates".
    session(world)
        .send("jj")
        .unwrap_or_else(|e| panic!("failed to move to the diagnose action: {e}"));
}

#[then("the diagnose action says it runs rocm examine")]
async fn diagnose_action_runs_examine(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("Runs: rocm examine", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the diagnose action's `Runs:` line did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("Diagnose & fix") && !screen.contains("rocm doctor"),
        "the `Runs:` line must belong to the diagnose action and name a real \
         subcommand, not `rocm doctor`:\n{screen}"
    );
}

#[then("ROCm setup actions are displayed")]
async fn rocm_actions_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Set up / Install ROCm", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the ROCm setup actions did not appear: {e}"));
}

#[then("the assistant's GPU status response is displayed")]
async fn gpu_response_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("GPU-2 is running hot", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the assistant's response did not appear: {e}"));
}

#[then("a tool approval prompt is displayed")]
async fn approval_prompt_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("Review: Install TheRock ROCm SDK?", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the approval prompt did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("Approve (y)") && screen.contains("Deny (n)"),
        "approval prompt is missing its Approve/Deny buttons:\n{screen}"
    );
}

#[then("the tool call is shown as declined")]
async fn tool_call_shown_declined(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_until_gone("Review: Install TheRock ROCm SDK?", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the approval prompt is still open after Enter: {e}"));
    tui.wait_for_screen("Action declined.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the declined-tool-call message did not appear: {e}"));
}

#[then("the managed model's response is displayed")]
async fn managed_model_response_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("mock response for testing", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the managed model's response did not appear: {e}"));
}

#[then("the mock received the typed prompt")]
async fn managed_chat_request_carried_prompt(world: &mut E2eWorld) {
    // The canned reply above is fixed regardless of what was sent, so it alone
    // can't prove the TUI actually submitted `MANAGED_MODEL_PROMPT` (a stray
    // keystroke corrupting the prompt would still show that reply). Assert on
    // the request the mock actually received instead. `wait_for_chat_request`
    // polls rather than reading a single snapshot: the response already
    // rendering on screen only proves the reply arrived, not that the mock's
    // handler finished recording the request into shared state first.
    let body = world
        .mock
        .as_ref()
        .expect("no mock server running")
        .wait_for_chat_request(default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the mock never received a chat request: {e}"));
    let messages = body
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("chat request had no messages array:\n{body}"));
    let last_user_content = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(serde_json::Value::as_str) == Some("user"))
        .and_then(|m| m.get("content"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| panic!("no user message found in chat request:\n{body}"));
    assert_eq!(
        last_user_content, MANAGED_MODEL_PROMPT,
        "mock did not receive the exact typed prompt; full request:\n{body}"
    );
}

/// The recorded chat request's message contents, in order.
///
/// The grounding steps look across every role rather than only `system`: what
/// matters is that the model was told, not which envelope carried it (the
/// built-in local provider folds system text into the user turn).
///
/// Waits for the request carrying `MANAGED_MODEL_PROMPT` specifically. Accepting
/// any chat request instead picks up the local-endpoint detection probe, which
/// is sent before the user types and carries no system prompt at all — the
/// grounding then looks absent when it was simply asserted against the wrong
/// request. Unlike `chat-03`, these steps have no screen wait ahead of them to
/// order the two.
async fn recorded_chat_messages(world: &mut E2eWorld) -> Vec<String> {
    let body = world
        .mock
        .as_ref()
        .expect("no mock server running")
        .wait_for_chat_request_where(default_timeout(), |body| {
            body.get("messages")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|messages| {
                    messages.iter().any(|m| {
                        m.get("role").and_then(serde_json::Value::as_str) == Some("user")
                            && message_text(m.get("content").unwrap_or(&serde_json::Value::Null))
                                .contains(MANAGED_MODEL_PROMPT)
                    })
                })
        })
        .await
        .unwrap_or_else(|e| panic!("the mock never received the user's chat turn: {e}"));
    body.get("messages")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("chat request had no messages array:\n{body}"))
        .iter()
        .filter_map(|m| m.get("content"))
        .map(message_text)
        .collect()
}

/// The text of one OpenAI-format message. `content` is a bare string on the
/// turns the TUI builds, but an array of typed parts on the system message the
/// chat client emits — read both, or the grounding looks absent when it is
/// simply wrapped.
fn message_text(content: &serde_json::Value) -> String {
    content.as_str().map_or_else(
        || {
            content
                .as_array()
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(serde_json::Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default()
        },
        str::to_owned,
    )
}

/// The single line of the sent prompt that opens with `label`, or a failure
/// naming what was actually sent. Asserting on the request — never on the
/// canned reply — is the point: the mock answers identically whatever it is
/// told, so only the request can show the assistant was grounded.
fn sent_fact_line(messages: &[String], label: &str) -> String {
    messages
        .iter()
        .flat_map(|m| m.lines())
        .map(str::trim)
        .find(|line| line.starts_with(label))
        .unwrap_or_else(|| {
            panic!(
                "the assistant was never told `{label}`; the request carried:\n{}",
                messages.join("\n---\n")
            )
        })
        .to_owned()
}

#[then("the assistant is told which operating system this machine runs")]
async fn assistant_told_the_operating_system(world: &mut E2eWorld) {
    let messages = recorded_chat_messages(world).await;
    let line = sent_fact_line(&messages, "- Operating system:");
    let host = e2e_cucumber::capability::host_capability();
    let expected = if host.os_family.eq_ignore_ascii_case("windows") {
        "Windows"
    } else {
        "Linux"
    };
    assert!(
        line.contains(expected),
        "this machine runs {}, but the assistant was told: {line}",
        host.os_family
    );
    // WSL is the case the old prompt got wrong — it told WSL users vLLM was
    // unavailable — so a WSL host must be named as one, not flattened to Linux.
    assert_eq!(
        line.contains("WSL"),
        host.is_wsl,
        "WSL must be stated exactly when this machine is WSL (is_wsl={}): {line}",
        host.is_wsl
    );
}

#[then("the assistant is told which GPU this machine has")]
async fn assistant_told_the_gpu(world: &mut E2eWorld) {
    let messages = recorded_chat_messages(world).await;
    let line = sent_fact_line(&messages, "- AMD GPU:");
    let host = e2e_cucumber::capability::host_capability();
    match host.gfx_target.as_deref() {
        // A host with a real GPU must see that GPU named, not a placeholder.
        Some(target) => assert!(
            line.contains(target),
            "this machine's GPU is {target}, but the assistant was told: {line}"
        ),
        // A host without one must be told so explicitly, rather than left to
        // fill the silence from pretraining.
        None => assert!(
            line.contains("no AMD GPU detected"),
            "no GPU is detectable here, so the assistant must be told that: {line}"
        ),
    }
}

#[then("the managed model is shown as loading rather than ready")]
async fn managed_model_shown_loading(world: &mut E2eWorld) {
    let model = world.model_name.clone().expect("no model name set");
    let tui = session(world);
    tui.wait_for_screen(&model, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the managed model did not appear: {e}"));
    // The compact Observe table intentionally omits lifecycle status; opening
    // the selected instance exposes the status a user uses to diagnose startup.
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to open instance details: {e}"));
    tui.wait_for_screen("LOADING", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the loading state did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(screen.contains(&model), "managed model missing:\n{screen}");
    assert!(
        !screen
            .lines()
            .any(|line| line.contains(&model) && line.contains("READY")),
        "loading model was presented as ready:\n{screen}"
    );
}

#[then("live serving metrics are displayed for the managed model")]
async fn managed_model_metrics_displayed(world: &mut E2eWorld) {
    let model = world.model_name.clone().expect("no model name set");
    let tui = session(world);
    tui.wait_for_screen(&model, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the managed model did not appear: {e}"));
    tui.wait_for_screen("50ms", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("TTFT metrics did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(screen.contains("20ms"), "TPOT metrics missing:\n{screen}");
    let row = screen
        .lines()
        .find(|line| line.contains(&model))
        .unwrap_or_default();
    assert!(
        row.contains("50ms") && row.contains("20ms") && row.contains("1/0") && row.contains("25%"),
        "managed serving metrics were incomplete:\n{screen}"
    );
}

#[then("navigation and next-step guidance are displayed")]
async fn navigation_guidance_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("toggle this help", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("dashboard help did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("next / previous tab")
            && screen.contains("Home tab")
            && screen.contains("jump ±60s"),
        "navigation or contextual guidance missing:\n{screen}"
    );
}

#[then("the services manager is displayed")]
async fn services_manager_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Services — managed inference servers", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the services manager did not appear: {e}"));
}

#[then("the services manager is closed")]
async fn services_manager_closed(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_until_gone("Services — managed inference servers", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the services manager is still open after Escape: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("● Observe"),
        "Escape left the Observe tab entirely, not just the manager:\n{screen}"
    );
    // Belt-and-suspenders: `wait_until_gone` above is the primary regression
    // check (the manager itself closed). This additionally guards against
    // Esc falling through to open the main menu instead — "Options"/"Quit"
    // are unique to `Modal::Menu`.
    assert!(
        !screen.contains("Options") && !screen.contains("Quit"),
        "the main menu is open on top of the closed manager:\n{screen}"
    );
}

#[then("instance details are displayed")]
async fn instance_details_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Instance · ", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("instance details did not appear: {e}"));
}

#[then("the instance detail footer shows the scroll hint")]
async fn detail_footer_shows_scroll_hint(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("↑/↓ scroll", default_timeout())
        .await
        .unwrap_or_else(|e| {
            panic!("footer did not show the scroll hint once the detail body overflowed: {e}")
        });
}

#[then("the instance detail footer does not show the scroll hint")]
async fn detail_footer_does_not_show_scroll_hint(world: &mut E2eWorld) {
    // Pins the precondition the later shrink step's barrier depends on: the
    // enlarged pre-shrink geometry must genuinely have no hint yet, or the
    // shrink step's own wait would silently revert to a no-op (its marker
    // already present) for the same reason a prior round of this scenario
    // was flagged for. A plain read is correct here — this runs right after
    // `open_instance_detail`'s own wait, with no action in between that
    // could still be in flight.
    let screen = session(world).screen_text();
    assert!(
        !screen.contains("↑/↓ scroll"),
        "footer must not show the scroll hint before the terminal shrinks:\n{screen}"
    );
}

#[then("the instance detail body shows a scrollbar")]
async fn detail_body_shows_scrollbar(world: &mut E2eWorld) {
    // Runs immediately after the scroll-hint `Then`, which already
    // synchronized to the post-resize frame via `wait_for_screen` — no
    // further redraw is expected between the two assertions, so a plain
    // read is correct here too.
    let screen = session(world).screen_text();
    assert!(
        screen.contains('║') || screen.contains('█'),
        "detail body did not show a scrollbar once it overflowed:\n{screen}"
    );
}

#[then("the backdrop behind the popup is dimmed")]
async fn backdrop_is_dimmed(world: &mut E2eWorld) {
    let tui = session(world);
    assert!(
        tui.corner_backdrop_is_dimmed(),
        "the screen behind the popup was not dimmed:\n{}",
        tui.screen_text()
    );
}

#[then("dashboard destinations are displayed")]
async fn dashboard_destinations_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("Go to", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("command palette did not appear: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("Home") && screen.contains("Serving") && screen.contains("Observe"),
        "command-palette destinations missing:\n{screen}"
    );
}

#[then("the dashboard menu is displayed")]
async fn dashboard_menu_is_displayed(world: &mut E2eWorld) {
    // "Quit" is used here as a marker for `Modal::Menu`'s three items
    // (Options/Help/Quit). Unlike "Options", "Quit" appears nowhere else in
    // the TUI's rendered chrome, so it unambiguously identifies the menu.
    session(world)
        .wait_for_screen("Quit", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("dashboard menu did not appear: {e}"));
}

/// Navigates `Modal::Menu`'s items (`["Options", "Help", "Quit"]`,
/// `ui/modal.rs`) down to "Quit" and activates it — `j` twice from the
/// default `menu_sel: 0`, then `Enter` (`KeyAction::MenuActivate`). Requires
/// the menu to already be open (`the dashboard menu is displayed`).
#[when("the user selects Quit from the dashboard menu")]
async fn select_quit_from_dashboard_menu(world: &mut E2eWorld) {
    session(world)
        .send("jj\r")
        .unwrap_or_else(|e| panic!("failed to select Quit from the dashboard menu: {e}"));
}

#[then("the dashboard menu is closed")]
async fn dashboard_menu_is_closed(world: &mut E2eWorld) {
    // A bare Escape send is not guaranteed to have been acted on yet by the
    // time the next step runs — confirm `Modal::Menu` actually closed before
    // quitting, the same way `services_manager_closed` does. Without this,
    // an unlanded close leaves the menu open and swallows the subsequent
    // quit keystroke (`Modal::Menu` has no `q` arm), hanging until the
    // quit step's timeout.
    session(world)
        .wait_until_gone("Options", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the dashboard menu is still open after Escape: {e}"));
}

#[then("Serving actions are displayed")]
async fn serving_actions_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Serving actions", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("Serving actions did not appear: {e}"));
}

#[then("the onboarding welcome screen is displayed")]
async fn onboarding_welcome_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Let's get ROCm set up on this machine.", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the onboarding welcome screen did not appear: {e}"));
}

#[then("the onboarding setup choices are displayed")]
async fn onboarding_choices_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Install ROCm SDK (pip)", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the onboarding setup choices did not appear: {e}"));
}

#[then("the SDK Configure step is displayed")]
async fn sdk_configure_step_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    tui.wait_for_screen("default managed folder", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the SDK Configure step did not appear: {e}"));
}

#[then("the install-folder browser is displayed")]
async fn install_folder_browser_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("Pick an install folder", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the install-folder browser did not appear: {e}"));
}

#[then("the existing-folder browser lists the scenario folder's contents")]
async fn existing_folder_browser_shows_scenario_folder(world: &mut E2eWorld) {
    // The browser lists `ADOPT_SCENARIO_CHILD` as a child entry only if it's
    // browsing our scenario's own cwd (seeded from `std::env::current_dir()`
    // in `onboarding.rs`'s `AdoptExisting` arm) — so this is only reachable if
    // `spawn_in` actually set the child process's working directory to our
    // scenario root, exactly as the full path would prove, but without
    // depending on the host's temp-directory length (see
    // `ADOPT_SCENARIO_CHILD`'s doc comment).
    let marker = format!("{ADOPT_SCENARIO_CHILD}/");
    session(world)
        .wait_for_screen(&marker, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the existing-folder browser did not show {marker:?}: {e}"));
}

#[then("the existing-folder browser is displayed")]
async fn existing_folder_browser_displayed(world: &mut E2eWorld) {
    // The Adopt browser's own popup title (`FolderBrowser::new("Pick an
    // existing ROCm folder", ...)` in `onboarding.rs`'s `activate_choice`) —
    // distinct from the Configure-path browser's "Pick an install folder"
    // the existing `install_folder_browser_displayed` step waits for.
    // Synchronizes before `choose_current_folder`, which has no wait of its
    // own, the same way `install_folder_browser_displayed` does for dash-23.
    session(world)
        .wait_for_screen("Pick an existing ROCm folder", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the existing-folder browser did not appear: {e}"));
}

#[then("the approval preview quotes the whitespace-containing root path")]
async fn approval_preview_quotes_whitespace_root(world: &mut E2eWorld) {
    // The real path the Adopt browser resolved: `activate_choice` seeds it
    // from the spawned child's own `std::env::current_dir()`, which on
    // Linux always reports the fully resolved (symlink-free) absolute path
    // of the `spawn_in` cwd — the same string `Path::canonicalize` computes
    // here on the directory `open_dashboard_in_whitespace_named_folder`
    // created, so the two agree without hardcoding a path.
    let root = world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .join(ADOPT_WHITESPACE_ROOT)
        .canonicalize()
        .unwrap_or_else(|e| panic!("failed to canonicalize the scenario root: {e}"));
    // `quote_display_arg` (`crates/rocm-dash-tui/src/ui/exec.rs`, #500) wraps
    // any whitespace-containing argument value in a pair of quote characters
    // with no other transformation (our path has no embedded quote to
    // escape), so the rendered `--root` argument is this exact string,
    // quoted. #693's contract is "rendered as one unambiguous, quoted unit" —
    // not any particular quote character — so this accepts either common
    // convention rather than pinning the double-quote `quote_display_arg`
    // happens to use today; a future change to the escaping character alone
    // should not read as a regression here.
    //
    // Deliberately *without* a `--root ` prefix, and matched against
    // `TuiSession::popup_body_text` rather than the raw `screen_text()` grid
    // (PR #707 review, 2026-10-16, and a follow-up review of that fix on the
    // same date): the popup wraps its command line at word boundaries, and
    // on a long enough host temp path a plain single-row substring match —
    // of the `--root ` prefix and its quoted value together, or even of the
    // quoted value alone — times out even though the product rendered the
    // path quoted correctly, because the match can't see across the row
    // break. `popup_body_text` reconstructs the popup's own body across
    // however many rows it actually wrapped onto, so this holds regardless
    // of host path length. The quoted value alone is still unambiguous
    // within that reconstruction: the only other quoted argument in this
    // body is `--python`, and `quote_display_arg` quotes that argument's
    // *entire* value (`<root>/bin/python`), so its closing quote never lands
    // right after `<root>` the way it does for `--root`'s value alone.
    let expected_double_quoted = format!("\"{}\"", root.display());
    let expected_single_quoted = format!("'{}'", root.display());
    session(world)
        .wait_for_popup_body_where(
            &format!(
                "the approval preview to contain {expected_double_quoted:?} or {expected_single_quoted:?}"
            ),
            ADOPT_APPROVAL_TITLE,
            |body| body.contains(&expected_double_quoted) || body.contains(&expected_single_quoted),
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| {
            panic!(
                "the approval preview never showed {expected_double_quoted:?} or {expected_single_quoted:?}: {e}"
            )
        });
}

#[then("the Configure step shows the chosen folder instead of the placeholder")]
async fn configure_shows_chosen_folder(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen_where(
            "the Folder row shows a chosen path rather than the unset placeholder, with the install-folder browser closed",
            |screen| {
                screen.contains("Folder:")
                    && !screen.contains("default managed folder · Tab to browse")
                    && !screen.contains("Pick an install folder")
            },
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| panic!("the Folder row never showed a chosen path with the browser closed: {e}"));
}

#[then("the managed model is displayed")]
async fn managed_model_displayed(world: &mut E2eWorld) {
    let model = world
        .model_name
        .as_deref()
        .expect("no model name set")
        .to_string();
    session(world)
        .wait_for_screen(&model, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("managed model did not appear: {e}"));
}

fn assert_tui_opened(world: &E2eWorld) {
    // `quit_and_wait` (in the "quits" step) already reaped the process and
    // asserted a zero exit; reaching here means the whole launch→interact→quit
    // round trip through the real terminal succeeded.
    assert!(
        world.tui.is_some(),
        "no TUI session was opened for this scenario"
    );
}

#[then("the dashboard exits successfully")]
async fn dashboard_exited(world: &mut E2eWorld) {
    assert_tui_opened(world);
}

/// Assert the exit code stashed by the terminating `When` step. Shared by the
/// dashboard and launcher wordings — the assertion is identical, only the
/// process under test differs, and `subject` keeps the failure message honest
/// about which one it was. `gesture` names how the exit was requested, so a
/// failure says whether the signal path or the keystroke path is broken.
fn assert_exited_with(world: &mut E2eWorld, subject: &str, gesture: &str, expected: i32) {
    let observed = session(world)
        .observed_exit_code()
        .expect("no exit code was recorded; terminate the session first");
    assert_eq!(
        observed, expected,
        "{subject} exited with {observed} after {gesture}, expected {expected}"
    );
}

#[then(expr = "the dashboard exits from the signal with code {int}")]
async fn dashboard_exited_from_signal(world: &mut E2eWorld, expected: i32) {
    assert_exited_with(world, "the dashboard", "the signal", expected);
}

// The launcher hub is a different process shape from a dashboard session (it
// outlives each session's runtime), so scenarios that signal the hub say so
// rather than borrowing the dashboard's wording.
#[then(expr = "the launcher exits from the signal with code {int}")]
async fn launcher_exited_from_signal(world: &mut E2eWorld, expected: i32) {
    assert_exited_with(world, "the launcher", "the signal", expected);
}

// Separate wording from the signal steps on purpose: a typed Ctrl-C never
// becomes a signal while the terminal is in raw mode, so a scenario that says
// "from the signal" here would assert the wrong thing about how the exit
// happened, even though the code it lands on is the same 130.
#[then(expr = "the dashboard exits from the keystroke with code {int}")]
async fn dashboard_exited_from_keystroke(world: &mut E2eWorld, expected: i32) {
    assert_exited_with(world, "the dashboard", "the keystroke", expected);
}

#[then(expr = "the launcher exits from the keystroke with code {int}")]
async fn launcher_exited_from_keystroke(world: &mut E2eWorld, expected: i32) {
    assert_exited_with(world, "the launcher", "the keystroke", expected);
}

#[then("the launcher front door is displayed")]
async fn launcher_front_door_displayed(world: &mut E2eWorld) {
    let tui = session(world);
    // "Set up this system" is the front door's first menu entry, drawn at any
    // size. Waiting (rather than reading the screen once) synchronises on the
    // first paint after a launch or after a session hands control back.
    tui.wait_for_screen("Set up this system", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the launcher front door was not displayed: {e}"));
}

#[then("the terminal is restored to the normal screen")]
async fn terminal_restored(world: &mut E2eWorld) {
    // The dashboard's signal handler must leave the alternate screen and show
    // the cursor before exiting; otherwise the shell is left in the broken
    // raw/alt-screen state that needs a `reset`.
    session(world)
        .expect_terminal_restored()
        .unwrap_or_else(|e| panic!("{e}"));
}

#[then("the launcher shows the model serving")]
async fn launcher_shows_serving(world: &mut E2eWorld) {
    let model = world
        .model_name
        .as_deref()
        .expect("no model name set")
        .to_string();
    let tui = session(world);
    // The front door's status strip renders "Serving <model>" for a live
    // registry instance; wait on the model name to synchronise with the first
    // paint before inspecting the whole screen.
    tui.wait_for_screen(&model, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the launcher never showed the serving model: {e}"));
    let screen = tui.screen_text();
    assert!(
        screen.contains("Serving"),
        "launcher did not show the model as serving:\n{screen}"
    );
    assert!(
        !screen.contains("Idle — nothing serving"),
        "launcher still reported idle despite a live registry instance:\n{screen}"
    );
}

#[then("the launcher exits successfully")]
async fn launcher_exited(world: &mut E2eWorld) {
    assert_tui_opened(world);
}

#[then("interactive chat exits successfully")]
async fn interactive_chat_exited(world: &mut E2eWorld) {
    assert_tui_opened(world);
}

// ── Then: chat privacy consent gate (real endpoint) ──

#[then("the local endpoint is shown for confirmation")]
async fn local_endpoint_shown(world: &mut E2eWorld) {
    // The consent gate echoes the detected endpoint; its port is the mock's
    // OS-assigned one, proving the CLI discovered the planted managed service
    // (not a hard-coded default) before offering it.
    let port = world
        .mock
        .as_ref()
        .expect("no mock server running")
        .port()
        .to_string();
    session(world)
        .wait_for_screen(&port, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the detected endpoint was not shown: {e}"));
}

#[then("the privacy notice is shown before any message is sent")]
async fn privacy_notice_shown(world: &mut E2eWorld) {
    // The gate is reached before any message can be submitted, so seeing this
    // line proves the notice precedes the first request.
    session(world)
        .wait_for_screen(
            "Your request only leaves this machine after you accept.",
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| panic!("the privacy notice was not shown: {e}"));
}

// ── EAI-7960: scripted metrics / validity-window regression ────────────────

/// Start the mock in Growing mode so the daemon builds a positive gen_tps
/// baseline before the scenario injects the Failure transition.
#[given("a managed model exposes scripted serving metrics")]
async fn managed_model_scripted_metrics(world: &mut E2eWorld) {
    let model = "TestModel/E2E-1B";
    let mock = MockServer::start_with_scripted_metrics(model).await;
    world.endpoint = Some(mock.base_url());
    world.model_name = Some(model.to_string());
    world.mock = Some(mock);
    world.register_mock_service_with(ServiceRecordOptions::default());
}

/// Plant the clock file before the dashboard is launched.
///
/// `rocm dash` decides once, at daemon construction, whether a logical clock is
/// in play — it takes the file's presence as the signal. So this must run before
/// the "opens the dashboard" step, which the scenarios guarantee by ordering
/// this `Given` ahead of them.
#[given("dashboard observation time is deterministic")]
async fn dashboard_observation_time_is_deterministic(world: &mut E2eWorld) {
    write_dash_clock(&dash_clock_path(world), "0");
}

/// Path of this scenario's test-clock file, inside the isolated data root the
/// CLI resolves from `ROCM_CLI_DATA_DIR` — no env var of its own, so the
/// binary under test carries no test-only branch.
///
/// Resolved through `rocm_core::AppPaths::dash_test_clock_file`, the same method
/// `rocm dash` reads it with, so where the harness plants the file and where the
/// dashboard looks for it cannot drift apart. Only the data root is the harness's
/// own: it is the directory this suite hands the CLI as `ROCM_CLI_DATA_DIR`.
fn dash_clock_path(world: &E2eWorld) -> std::path::PathBuf {
    world.app_paths().dash_test_clock_file()
}

/// Wait until `rocm dash`'s own client log records that it found the planted
/// clock file — the WARN it logs only when it switches off wall time.
///
/// Without this, nothing proves the dashboard under test is on the logical
/// clock at all. Measured: with `rocm dash` made to ignore the file, dash-09
/// still passes, because the 6 s validity window simply elapses on wall time
/// inside the expiry step's wait. The held-clock assertions then hold only by
/// timing, which is the flake the clock exists to remove. Checking the log is
/// deterministic: the line names the exact file, or it never appears.
///
/// Polled rather than read once: the client log is written by a non-blocking
/// appender, so the line can trail the launch slightly.
///
/// Takes the resolved paths rather than the world: a `&E2eWorld` held across the
/// `.await` below would need `E2eWorld: Sync`, which it is not.
async fn assert_dashboard_reads_the_planted_clock(paths: rocm_core::AppPaths) {
    let clock = paths.dash_test_clock_file().display().to_string();
    let log_dir = paths.client_log_dir();
    let deadline = std::time::Instant::now() + default_timeout();
    loop {
        let found = std::fs::read_dir(&log_dir)
            .into_iter()
            .flatten()
            .filter_map(|entry| std::fs::read_to_string(entry.ok()?.path()).ok())
            .any(|log| {
                log.lines()
                    .any(|line| line.contains("TEST mode") && line.contains(&clock))
            });
        if found {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "`rocm dash` never logged that it found the planted clock file {clock} \
             (searched {}), so it is on wall time and the held-clock assertions \
             below would pass or fail by timing alone",
            log_dir.display()
        );
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
}

/// Publish a clock directive atomically (write a sibling temp file, then
/// rename). The daemon re-reads this file every cycle, so a plain truncating
/// write can be observed mid-update as an empty file; rename makes each
/// directive visible all-at-once instead.
fn write_dash_clock(path: &std::path::Path, directive: &str) {
    // The telemetry state dir is created by `AppPaths::ensure()` on the first
    // CLI run, which for these scenarios happens after this step.
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", parent.display()));
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, directive).expect("failed to stage the dashboard test clock");
    std::fs::rename(&tmp, path).expect("failed to publish the dashboard test clock");
}

/// The TTFT cell `model`'s row is currently rendering, or `None` while that row
/// is not on screen at all.
///
/// Cells are whitespace-separated and a model id carries no spaces, so counting
/// fields from the id yields one field per column — including the `—`
/// placeholder a cleared cell renders, which keeps the columns aligned.
fn scripted_ttft_cell<'a>(screen: &'a str, model: &str) -> Option<&'a str> {
    screen
        .lines()
        .find(|line| line.contains(model))?
        .split_whitespace()
        .skip_while(|field| *field != model)
        .nth(TTFT_COLUMN)
}

/// The Observe tab's node-throughput hero shows the "tok/s" unit whenever
/// `gen_tps` is `Some(_)`. Wait for it to confirm a positive baseline was
/// established through at least two successful Growing-mode scrapes.
#[then("positive generation throughput is displayed for the managed model")]
async fn positive_gen_tps_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen("tok/s", default_timeout())
        .await
        .unwrap_or_else(|e| {
            panic!("positive gen_tps (\"tok/s\") never appeared after Growing-mode scrapes: {e}")
        });
}

/// Stop the daemon's logical observation clock where it stands.
///
/// Free-running, that clock advances one `gpu_tick` per daemon cycle and the
/// cycles are paced by a wall-clock interval — so it tracks wall time, and a
/// scenario descheduled between the failure below and its assertion spends
/// validity budget it never meant to. That is not hypothetical: on the
/// 64-concurrent-scenario mock lane this step's successor was reached four
/// failed scrapes (8 logical seconds) late, past the 6 s window, and the
/// scenario reported a regression the daemon had not committed.
///
/// Held, the clock cannot be moved by anything except this scenario rewriting
/// the file, so the assertions below hold at any later moment, and only the
/// explicit advance in `validity_window_elapsed` crosses the boundary.
///
/// Held *before* the failure, deliberately: the daemon adopts the directive
/// within one cycle of the write, independently of how the harness is
/// scheduled, so the last successful observation is at most
/// `instance_tick + gpu_tick` (3 s) older than the frozen instant — inside the
/// 6 s window with margin, and it stays there.
#[when("dashboard observation time is held")]
async fn dashboard_observation_time_is_held(world: &mut E2eWorld) {
    // Holding a clock the dashboard never picked up would do nothing, silently.
    assert_dashboard_reads_the_planted_clock(world.app_paths()).await;
    write_dash_clock(&dash_clock_path(world), "hold");
}

/// Switch the scripted mock to Failure mode and wait for the failure to reach
/// the screen: first the mock's own counter proves the daemon was served a 503,
/// then the cleared TTFT cell proves the frame on display is one the daemon
/// assembled after that scrape. Both waits are synchronizations on observed
/// events, not fixed sleeps, and — the clock being held — neither can consume
/// the validity window they precede.
#[when("the metrics endpoint fails transiently")]
async fn metrics_endpoint_fails(world: &mut E2eWorld) {
    let mock = world.mock.as_ref().expect("no mock server running");
    mock.set_metrics_mode(MetricsMode::Failure);

    // Poll until the daemon delivers at least one 503 to the mock endpoint.
    // The production instance_tick is 2 s, so this converges in 2–3 s.
    let budget = default_timeout();
    let deadline = Instant::now() + budget;
    loop {
        if mock.metrics_failure_count() >= 1 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "scripted failure was never served within {budget:?}; \
             check instance_tick and scrape cadence"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let model = world
        .model_name
        .clone()
        .expect("the scripted-metrics Given records the model this row belongs to");
    session(world)
        .wait_for_screen_where(
            &format!("the scripted instance's TTFT cell leaves {SCRIPTED_TTFT_CELL:?}"),
            |screen| {
                scripted_ttft_cell(screen, &model).is_some_and(|cell| cell != SCRIPTED_TTFT_CELL)
            },
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| {
            panic!(
                "the failed scrape never reached the screen, so no frame here is known \
                 to postdate the failure: {e}"
            )
        });
}

/// EAI-7960 principal regression assertion.
///
/// The frame under assertion is provably post-failure (the TTFT cell it used to
/// show is gone) and the logical clock is held, so the only way "tok/s" can be
/// missing here is the regression itself: the daemon clearing a held rate on a
/// failed scrape instead of keeping it for the validity window.
#[then("generation throughput remains visible within the validity window")]
async fn gen_tps_held_after_failure(world: &mut E2eWorld) {
    let screen = session(world).screen_text();
    assert!(
        screen.contains("tok/s"),
        "EAI-7960 REGRESSION: gen throughput (\"tok/s\") was cleared immediately \
         after the first failed scrape instead of being held for the validity \
         window (clamp(3 × instance_tick, 6 s, 30 s)).\n\n\
         Last screen:\n{screen}"
    );
}

// ── EAI-7960: expiry boundary helpers ───────────────────────────────────────

/// Step the held clock 7 s past where it was held — one second beyond the 6 s
/// window, from an observation at most 3 s older than the hold point, so the
/// held value is unambiguously expired and stays expired. Nothing else moves
/// this clock, so the assertion below is about the daemon's arithmetic alone.
#[when("the validity window has elapsed")]
async fn validity_window_elapsed(world: &mut E2eWorld) {
    write_dash_clock(&dash_clock_path(world), "hold 7");
}

/// Assert that gen_tps is no longer rendered after the scenario steps the held
/// clock past the validity boundary.
///
/// The expired state is published every cycle and, the clock being held, it is
/// permanent — so waiting for it to reach the screen cannot mask a daemon that
/// kept the value: that daemon simply never clears it and this times out.
#[then("generation throughput is no longer displayed")]
async fn gen_tps_no_longer_displayed(world: &mut E2eWorld) {
    session(world)
        .wait_for_screen_where(
            "generation throughput leaves the screen",
            |screen| !screen.contains("tok/s"),
            default_timeout(),
        )
        .await
        .unwrap_or_else(|e| {
            panic!(
                "EAI-7960 BOUNDARY-2: gen_tps (\"tok/s\") is still visible after the \
                 validity window clamp(3 × instance_tick, 6 s, 30 s) elapsed. Expected \
                 the daemon to have cleared the held value and the TUI to show the \
                 unavailable placeholder: {e}"
            )
        });
}

/// Terminal rows tall enough that the Observe tab's 35%-height GPU sub-panel
/// (`super::hardware::draw` called from `tabs/observe.rs`) has enough of its
/// own area to show GPU rows at all. The demo session's 8 GPUs are already far
/// past `plan_gpu_rows`'s full-panel threshold at this height (that would need
/// the GPU sub-panel's own area north of 65 rows), so this lands solidly in
/// its compact one-line-per-GPU layout — the regime `gpu_compact_line` draws.
/// `grow_rows` keeps the standard 80-column width, comfortably over the 54
/// columns `gpu_compact_line`'s own truncation budget needs to keep the vram
/// segment on the line (`gpu_compact_line_vram_truncation_threshold` pins
/// that exact 54 in `hardware.rs`).
const GPU_COMPACT_ROWS: u16 = 70;

#[when("the user grows the terminal until the GPU section shows compact rows")]
async fn grow_for_gpu_compact_rows(world: &mut E2eWorld) {
    let tui = session(world);
    tui.grow_rows(GPU_COMPACT_ROWS)
        .unwrap_or_else(|e| panic!("failed to resize the terminal: {e}"));
    // The demo replay's GPU telemetry lands asynchronously, same as the
    // instance table `open_instance_detail` waits on above; wait for a GPU
    // row's own marker rather than assuming the first post-resize frame
    // already carries it.
    tui.wait_for_screen("°C", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("GPU compact rows never rendered: {e}"));
}

#[then("the GPU section reports VRAM in binary units")]
async fn gpu_section_reports_binary_units(world: &mut E2eWorld) {
    let screen = session(world).screen_text();
    // Scoped to the compact GPU row itself (the line carrying its "°C"
    // cell) rather than the whole screen: "GiB" alone also appears in the
    // section header's per-logical-GPU figure and the instances table, so a
    // screen-wide positive assertion would pass even if this row regressed.
    let gpu_row = screen
        .lines()
        .find(|line| line.contains("°C"))
        .unwrap_or_else(|| panic!("no compact GPU row on screen:\n{screen}"));
    assert!(
        gpu_row.contains("GiB"),
        "expected the GPU row's VRAM figure in binary (GiB) units, got:\n{gpu_row}"
    );
    assert!(
        !gpu_row.contains(" GB"),
        "#673 REGRESSION: GPU VRAM is labeled in decimal-looking GB units \
         instead of binary GiB, got:\n{gpu_row}"
    );
}

// ── Doctor report ──────────────────────────────────────────────────

/// The Diagnose screen's own panel title, drawn by `draw_examine_manager`. It is
/// on screen exactly while the screen is open, so it proves the screen opened
/// and, as an absence, that it closed.
const DIAGNOSE_SCREEN_TITLE: &str = "Examine — environment check";

#[when("the user opens the Diagnose screen")]
async fn open_diagnose_screen(world: &mut E2eWorld) {
    // `d` opens it from the Observe tab (`KeyAction::OpenExamine`). Resent until
    // it takes, for the same reason `open_observe_view` resends: nothing before
    // this proves the event loop is reading input yet.
    session(world)
        .send_until("d", DIAGNOSE_SCREEN_TITLE, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("failed to open the Diagnose screen: {e}"));
}

#[when("the user prepares a Doctor report")]
async fn prepare_doctor_report(world: &mut E2eWorld) {
    session(world)
        .send("p")
        .unwrap_or_else(|e| panic!("failed to ask for a Doctor report: {e}"));
}

#[then("the report is displayed for review")]
async fn report_displayed_for_review(world: &mut E2eWorld) {
    let tui = session(world);
    // Whichever way this machine goes, the command prints one JSON object that
    // carries a `schema`: the report, or the refusal envelope.
    tui.wait_for_screen("\"schema\"", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the report preview did not appear: {e}"));
    // The console's dismiss hint appears only once the job has finished, so
    // the key that follows cannot land on a job still running.
    tui.wait_for_screen("Enter/Esc dismiss", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the report preview never finished: {e}"));
    // What the CLI said, read from its own output on the console: the refusal
    // envelope is the one with a `refused` key. The card is checked against it
    // after the console is left.
    world.dash_report_refused = Some(tui.screen_text().contains("\"refused\""));
}

#[when("the user leaves the report review")]
async fn leave_report_review(world: &mut E2eWorld) {
    let tui = session(world);
    tui.send("\r")
        .unwrap_or_else(|e| panic!("failed to leave the report review: {e}"));
    tui.wait_for_screen("Prepare a Doctor report", default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the Diagnose screen did not come back: {e}"));
}

#[then("the dashboard offers to send a report only if the CLI prepared one")]
async fn dashboard_offers_send_only_if_the_cli_prepared_one(world: &mut E2eWorld) {
    let refused = world
        .dash_report_refused
        .expect("the preview step records whether the CLI refused");
    let screen = session(world).screen_text();
    let ready = screen.contains("A report is ready");
    let offers_send = screen.contains("· s review & send");
    // The CLI decides. A report is ready exactly when the CLI printed one, and
    // the dashboard may offer to send exactly then.
    assert_eq!(
        ready, !refused,
        "the card must say a report is ready exactly when the CLI did not refuse \
         (CLI refused: {refused}, card ready: {ready}):\n{screen}"
    );
    assert_eq!(
        offers_send, !refused,
        "send must be offered exactly when the CLI did not refuse \
         (CLI refused: {refused}, send offered: {offers_send}):\n{screen}"
    );
    // A refusal reaches the card as the CLI's own explanation, not as the
    // dashboard's "could not be read". Every refusal the CLI gives says that no
    // report was prepared. The card puts the outcome first, so a short terminal
    // does not cut it off.
    if refused {
        assert!(
            screen.contains("prepared"),
            "the CLI's explanation did not reach the card:\n{screen}"
        );
        assert!(
            !screen.contains("could not be read"),
            "a refusal was shown as unreadable output:\n{screen}"
        );
    }
}

/// Title of the send approval the `s` key opens.
const SEND_REVIEW_TITLE: &str = "Report this diagnosis";

/// The part of the dashboard's sender notice this suite matches on, mirrored
/// from `REPORT_SENDER_NOTICE` in `crates/rocm-dash-tui/src/ui/examine_manager.rs`
/// (this crate runs the built binary and links neither).
const SENDER_NOTICE_FRAGMENT: &str = "it goes from your own mail account and carries your address";

/// The send review's title as the shared approval popup stamps it, which is
/// what `TuiSession::wait_for_popup_body_where` finds the popup by.
const SEND_REVIEW_TITLE_BAR: &str = "Review: Report this diagnosis";

#[when("the user asks to review the report for sending")]
async fn ask_to_review_for_sending(world: &mut E2eWorld) {
    // `s` only opens the review. Nothing is sent before an approve, and this
    // scenario never approves.
    session(world)
        .send("s")
        .unwrap_or_else(|e| panic!("failed to ask to review the report for sending: {e}"));
}

#[then("the review says the mail carries the user's own address before showing the report")]
async fn review_states_the_sender_notice_first(world: &mut E2eWorld) {
    let refused = world
        .dash_report_refused
        .expect("the preview step records whether the CLI refused");
    let tui = session(world);
    if refused {
        // Nothing to send, so no review may open. Give a stray dialog the
        // time it would take to draw before looking.
        tokio::time::sleep(Duration::from_millis(500)).await;
        let screen = tui.screen_text();
        assert!(
            !screen.contains(SEND_REVIEW_TITLE),
            "a refusal opened a send review:\n{screen}"
        );
        return;
    }
    // Read the popup's own body, not the whole screen. The review is drawn
    // over the Diagnose screen and the Observe panels, which share its rows,
    // so the whole screen joined into one line puts their text (a panel's
    // "MODEL" heading, say) into the middle of any sentence the review
    // wrapped. `wait_for_popup_body_where` slices each row to the popup's own
    // columns and undoes the wrap. It waits for the report's first key so the
    // body is read once the review has drawn the report.
    //
    // Both sides are then compared with all whitespace removed. A wrap that
    // ends flush with the popup's right border cannot be told apart from a
    // word split mid-word, so the rejoined body can read "carriesyour" there
    // (see `popup_body_text`). Without whitespace the fragment still occurs
    // only in the notice.
    let mut body = String::new();
    tui.wait_for_popup_body_where(
        "the send review shows the report",
        SEND_REVIEW_TITLE_BAR,
        |text| {
            text.clone_into(&mut body);
            text.contains("\"schema\"")
        },
        default_timeout(),
    )
    .await
    .unwrap_or_else(|e| panic!("the send review did not open with the report: {e}"));
    let screen = tui.screen_text();
    let unspaced = |text: &str| text.split_whitespace().collect::<String>();
    let compact = unspaced(&body);
    let notice = compact.find(&unspaced(SENDER_NOTICE_FRAGMENT));
    let report = compact.find("\"schema\"");
    assert!(
        notice.is_some(),
        "the send review does not say the mail carries the user's address:\n{body}\n{screen}"
    );
    // On the first screen of the review, above the report: the body scrolls
    // and Approve does not, so a notice below the report could be approved
    // past without ever being seen. The report's first key, `"schema"`, is on
    // that first screen too; requiring it keeps the order check from passing
    // only because the report was missing.
    assert!(
        report.is_some(),
        "the review's first screen does not show the report it asks about:\n{screen}"
    );
    assert!(
        notice < report,
        "the notice must come before the report in the review:\n{screen}"
    );
}

#[when("the user declines the send")]
async fn decline_the_send(world: &mut E2eWorld) {
    if world.dash_report_refused == Some(true) {
        return;
    }
    let tui = session(world);
    tui.send("n")
        .unwrap_or_else(|e| panic!("failed to decline the send: {e}"));
    tui.wait_until_gone(SEND_REVIEW_TITLE, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the send review is still open after declining: {e}"));
}

#[when("the user closes the Diagnose screen")]
async fn close_diagnose_screen(world: &mut E2eWorld) {
    // An open overlay eats the quit key, so a scenario that opened one closes it
    // before the quit step.
    let tui = session(world);
    tui.send("\u{1b}")
        .unwrap_or_else(|e| panic!("failed to close the Diagnose screen: {e}"));
    tui.wait_until_gone(DIAGNOSE_SCREEN_TITLE, default_timeout())
        .await
        .unwrap_or_else(|e| panic!("the Diagnose screen is still open after Escape: {e}"));
}
