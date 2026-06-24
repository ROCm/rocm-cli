//! Characterization safety-net for the dash TUI (Supergoal Phase 0, updated P3).
//!
//! Freezes `ui::draw` behaviour for every tab in the 5-tab IA
//! (Home / ROCm / Serving / Observe / Chat) as TestBackend buffer-text
//! assertions, plus a squeezed-height no-panic sweep and the Phase-3 inline-
//! manager (de-modal) rendering contract.
//!
//! Ponytail: reuse the existing `TestBackend` → `Terminal` → `ui::draw` →
//! flatten-buffer pattern; no new test framework, no demo NDJSON.

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use rocm_dash_core::metrics::{
    GpuMetrics, GpuSystemInfo, Instance, InstanceStatus, Snapshot, SystemMetrics,
};
use rocm_dash_tui::app::{ActiveTab, AppState, ConnState, PaneFocus};
use rocm_dash_tui::ui;

/// A synthetic single-GPU snapshot so each tab body has real content to paint.
fn synthetic_snapshot() -> Snapshot {
    Snapshot {
        host: SystemMetrics {
            cpu_overall_pct: 37.0,
            cpu_per_core_pct: vec![20.0, 40.0, 60.0, 80.0],
            memory_used_mb: 32_000,
            memory_total_mb: 128_000,
            disk_read_bps: 1_200_000,
            net_rx_bps: 2_500_000,
            ..Default::default()
        },
        gpus: vec![GpuMetrics {
            device_id: "GPU0".into(),
            vram_used_mb: 40_000,
            vram_total_mb: 192_000,
            gpu_utilization_pct: 72.0,
            temperature_c: 58.0,
            power_w: 420.0,
            clock_mhz: Some(2100.0),
        }],
        gpu_system_info: Some(GpuSystemInfo {
            gpu_model: "Instinct MI355X".into(),
            physical_gpu_count: 1,
            logical_gpu_count: 1,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Build a connected `AppState` parked on `tab` with the synthetic snapshot.
fn state_on(tab: ActiveTab) -> AppState {
    let mut s = AppState::new("test-connect".into(), "default-dark".into());
    s.active_tab = tab;
    s.conn = ConnState::Connected {
        host: "localhost".into(),
        version: "1.0".into(),
    };
    s.latest = Some(synthetic_snapshot());
    s
}

/// Render the full `ui::draw` chrome to a flat buffer string at `cols`×`rows`.
fn render(state: &mut AppState, cols: u16, rows: u16) -> String {
    let backend = TestBackend::new(cols, rows);
    let mut term = Terminal::new(backend).unwrap();
    term.draw(|f| ui::draw(f, state)).unwrap();
    term.backend()
        .buffer()
        .content()
        .iter()
        .map(ratatui::buffer::Cell::symbol)
        .collect()
}

/// The tab bar always paints every tab label; assert it is present so the
/// chrome itself is characterized once.
fn assert_tab_bar(out: &str) {
    for label in ["Home", "ROCm", "Serving", "Observe", "Chat"] {
        assert!(out.contains(label), "tab bar missing {label:?}: {out:?}");
    }
}

#[test]
fn home_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Home), 160, 44);
    assert_tab_bar(&out);
    assert!(
        out.contains("GPU UTILIZATION"),
        "home hero missing: {out:?}"
    );
}

#[test]
fn action_tab_renders_key_labels() {
    // ROCm shows the platform verbs; Serving shows the serving verbs.
    let out = render(&mut state_on(ActiveTab::Rocm), 160, 44);
    assert_tab_bar(&out);
    assert!(
        out.contains("Set up / Install ROCm") && out.contains("Runtimes"),
        "ROCm verbs missing: {out:?}"
    );
    let serving = render(&mut state_on(ActiveTab::Serving), 160, 44);
    assert!(
        serving.contains("Serve a model") && serving.contains("Engines"),
        "serving verbs missing: {serving:?}"
    );
}

#[test]
fn observe_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Observe), 160, 44);
    assert_tab_bar(&out);
    // Observe folds the former Overview/Hardware (host telemetry), Instances and
    // Bench surfaces into one tab.
    assert!(
        out.contains("CPU") && out.contains("Instances") && out.contains("Bench"),
        "observe folded surfaces missing: {out:?}"
    );
}

#[test]
fn chat_tab_renders_key_labels() {
    let out = render(&mut state_on(ActiveTab::Chat), 160, 44);
    assert_tab_bar(&out);
    assert!(out.contains("Chat"), "chat missing Chat block: {out:?}");
}

#[test]
fn wide_layout_shows_logs_or_context_dock_not_composer() {
    // Operational tab → LOGS dock; the dock must never be a chat composer.
    let observe = render(&mut state_on(ActiveTab::Observe), 200, 50);
    assert!(
        observe.contains("LOGS"),
        "wide Observe missing LOGS dock: {observe:?}"
    );
    // Home → CONTEXT rail (RUNNING SERVICES section).
    let home = render(&mut state_on(ActiveTab::Home), 200, 50);
    assert!(
        home.contains("CONTEXT") && home.contains("RUNNING SERVICES"),
        "wide Home missing CONTEXT rail: {home:?}"
    );
    // Neither dock leaks the chat composer.
    assert!(
        !observe.contains("press i to type"),
        "composer in Observe dock"
    );
    assert!(!home.contains("press i to type"), "composer in Home dock");
}

#[test]
fn narrow_layout_is_single_column_no_dock() {
    // Below the 180×45 threshold there is no LOGS/CONTEXT dock (fallback path).
    let out = render(&mut state_on(ActiveTab::Observe), 160, 44);
    assert!(
        !out.contains("LOGS"),
        "narrow layout must not show the dock: {out:?}"
    );
    assert!(
        !out.contains("CONTEXT"),
        "narrow layout must not show CONTEXT dock"
    );
}

// --- Phase 7: empty / loading / error states, honesty, a11y of color ---

#[test]
fn observe_empty_state_shows_placeholders() {
    // Connected but with no instances → honest empty placeholder, no banner.
    let mut s = AppState::new("c".into(), "default-dark".into());
    s.active_tab = ActiveTab::Observe;
    s.conn = ConnState::Connected {
        host: "h".into(),
        version: "1".into(),
    };
    s.latest = Some(synthetic_snapshot());
    let out = render(&mut s, 160, 44);
    assert!(
        out.contains("no instances"),
        "empty instances placeholder: {out:?}"
    );
}

#[test]
fn loading_state_connecting_banner() {
    // Connecting (no snapshot) → header shows the loading status + demo banner.
    let mut s = AppState::new("c".into(), "default-dark".into());
    s.active_tab = ActiveTab::Observe;
    s.conn = ConnState::Connecting;
    let out = render(&mut s, 160, 44);
    assert!(
        out.contains("connecting"),
        "loading status missing: {out:?}"
    );
    assert!(
        out.contains("demo data"),
        "demo banner expected while loading"
    );
}

#[test]
fn disconnected_banner_present() {
    // Disconnected → error status in header + demo banner on Observe.
    let mut s = AppState::new("c".into(), "default-dark".into());
    s.active_tab = ActiveTab::Observe;
    s.conn = ConnState::Disconnected {
        reason: "daemon gone".into(),
    };
    let out = render(&mut s, 160, 44);
    assert!(
        out.contains("disconnected"),
        "error status missing: {out:?}"
    );
    assert!(
        out.contains("demo data"),
        "demo banner expected when disconnected"
    );
}

#[test]
fn honesty_demo_banner_absent_when_connected_with_telemetry() {
    let mut s = AppState::new("c".into(), "default-dark".into());
    s.active_tab = ActiveTab::Observe;
    s.conn = ConnState::Connected {
        host: "h".into(),
        version: "1".into(),
    };
    s.latest = Some(synthetic_snapshot());
    let out = render(&mut s, 160, 44);
    assert!(
        !out.contains("demo data"),
        "banner must be hidden when live: {out:?}"
    );
}

#[test]
fn a11y_status_carries_text_label_not_color_only() {
    // Connection status is conveyed in words, not by color alone.
    let mut connected = state_on(ActiveTab::Home);
    let out = render(&mut connected, 160, 44);
    assert!(
        out.contains("connected"),
        "connected text label missing: {out:?}"
    );

    let mut s = AppState::new("c".into(), "default-dark".into());
    s.active_tab = ActiveTab::Home;
    s.conn = ConnState::Disconnected { reason: "x".into() };
    let dis = render(&mut s, 160, 44);
    assert!(
        dis.contains("disconnected"),
        "disconnected text label missing: {dis:?}"
    );
}

#[test]
fn control_legend_is_on_the_bottom_row_not_the_top() {
    // The footer keyboard legend must render on the LAST row, never inside the
    // body near the top (regression: footer rect once collapsed onto the body).
    let cols = 160u16;
    let rows = 44u16;
    let mut s = state_on(ActiveTab::Home);
    let backend = TestBackend::new(cols, rows);
    let mut term = Terminal::new(backend).unwrap();
    term.draw(|f| ui::draw(f, &mut s)).unwrap();
    let buf = term.backend().buffer().clone();

    let row_text = |y: u16| -> String {
        (0..cols)
            .map(|x| {
                buf.cell((x, y))
                    .map_or(" ", ratatui::buffer::Cell::symbol)
                    .to_string()
            })
            .collect()
    };
    // "quit" (legend tail) is on the bottom row.
    assert!(
        row_text(rows - 1).contains("quit"),
        "legend must be on the bottom row: {:?}",
        row_text(rows - 1)
    );
    // The top rows (header band) must NOT carry the body-level legend chips like
    // "select"/"jump" — only the small chrome hint may live up top.
    let top = (0..6).map(row_text).collect::<String>();
    assert!(
        !top.contains(" jump "),
        "body legend leaked to the top: {top:?}"
    );
}

/// Phase 3 de-modal contract: when a manager is open on a ROCm/Serving tab it
/// renders INLINE in the Details pane — the manager's bento title is painted AND
/// the left Actions list survives (proving a two-column inline layout, not a
/// full-screen-centered overlay). One assertion per manager (all 12).
#[test]
fn managers_render_inline_in_the_details_pane() {
    use rocm_dash_tui::app::PaneFocus;

    use rocm_dash_tui::ui::{
        automations_manager::AutomationsManagerState, command_screen::CommandScreenState,
        config_manager::ConfigManagerState, doctor_manager::DoctorManagerState,
        engine_manager::EngineManagerState, install_manager::InstallManagerState,
        logs_view::LogsViewState, onboarding::OnboardingState,
        runtime_manager::RuntimeManagerState, serve_wizard::ServeWizardState,
        services_manager::ServicesManagerState, update_manager::UpdateManagerState,
    };

    // (tab, open-closure, manager-title-needle, actions-list-needle)
    type Open = fn(&mut AppState);
    let cases: &[(ActiveTab, Open, &str, &str)] = &[
        // Serving-group managers (opened on the Serving tab).
        (
            ActiveTab::Serving,
            |s| s.engine_manager = Some(EngineManagerState::default()),
            "serving backends",
            "Serving actions",
        ),
        (
            ActiveTab::Serving,
            |s| s.services = Some(ServicesManagerState::default()),
            "managed inference servers",
            "Serving actions",
        ),
        (
            ActiveTab::Serving,
            |s| s.logs_view = Some(LogsViewState::default()),
            "recent ROCm CLI activity",
            "Serving actions",
        ),
        (
            ActiveTab::Serving,
            |s| s.config_manager = Some(ConfigManagerState::default()),
            "Config & providers",
            "Serving actions",
        ),
        // serve_wizard opened on ROCm so its "Serve a model" title can't be
        // confused with the Serving Actions list row of the same name.
        (
            ActiveTab::Rocm,
            |s| s.serve_wizard = Some(ServeWizardState::default()),
            "Serve a model",
            "ROCm actions",
        ),
        // ROCm-group managers (opened on the ROCm tab).
        (
            ActiveTab::Rocm,
            |s| s.install_manager = Some(InstallManagerState::default()),
            "ROCm SDK",
            "ROCm actions",
        ),
        (
            ActiveTab::Rocm,
            |s| s.update_manager = Some(UpdateManagerState::default()),
            "ROCm packages",
            "ROCm actions",
        ),
        (
            ActiveTab::Rocm,
            |s| s.doctor_manager = Some(DoctorManagerState::default()),
            "environment check",
            "ROCm actions",
        ),
        (
            ActiveTab::Rocm,
            |s| s.runtime_manager = Some(RuntimeManagerState::default()),
            "ROCm installs",
            "ROCm actions",
        ),
        (
            ActiveTab::Rocm,
            |s| s.command_screen = Some(CommandScreenState::default()),
            "Run a command",
            "ROCm actions",
        ),
        (
            ActiveTab::Rocm,
            |s| s.onboarding = Some(OnboardingState::default()),
            "first-run setup",
            "ROCm actions",
        ),
        (
            ActiveTab::Rocm,
            |s| s.automations_manager = Some(AutomationsManagerState::default()),
            "background checks",
            "ROCm actions",
        ),
    ];

    for (tab, open, manager_needle, actions_needle) in cases {
        let mut s = state_on(*tab);
        s.pane_focus = PaneFocus::Detail;
        open(&mut s);
        let out = render(&mut s, 160, 44);
        assert!(
            out.contains(manager_needle),
            "manager {manager_needle:?} did not render inline: {out:?}"
        );
        assert!(
            out.contains(actions_needle),
            "Actions list {actions_needle:?} was covered (not inline): {out:?}"
        );
    }
}

#[test]
fn every_tab_survives_squeezed_height() {
    // The body rect can collapse to 0–1 inner rows on a short terminal; assert
    // no tab panics when squeezed (the historical ActiveTab footgun).
    for tab in [
        ActiveTab::Home,
        ActiveTab::Rocm,
        ActiveTab::Serving,
        ActiveTab::Observe,
        ActiveTab::Chat,
    ] {
        let mut s = state_on(tab);
        for h in [1u16, 2, 3, 5, 8] {
            let _ = render(&mut s, 80, h);
        }
    }
}

// --- Phase 6: harden — a11y across all themes + demo buffer-dump ---

/// A connected demo-style state: synthetic snapshot + two instances, one with
/// live TTFT/TPOT/tok-watt and one with none (the honest `—` path).
fn demo_state() -> AppState {
    let mut s = state_on(ActiveTab::Observe);
    let live = Instance {
        container_id: "vllm-a".into(),
        container_name: "vllm-a".into(),
        status: InstanceStatus::Running,
        model_name: "deepseek-r1".into(),
        gpu_ids: vec!["0".into()],
        gen_tps: Some(180.0),
        tokens_per_watt: Some(0.6),
        ttft_ms: Some(150.0),
        tpot_ms: Some(22.0),
        running_reqs: Some(3),
        waiting_reqs: Some(1),
        kv_cache_usage_pct: Some(42.0),
        ..Default::default()
    };
    let cold = Instance {
        container_id: "vllm-b".into(),
        container_name: "vllm-b".into(),
        status: InstanceStatus::Running,
        model_name: "llama-3".into(),
        gpu_ids: vec!["1".into()],
        ..Default::default()
    };
    if let Some(snap) = s.latest.as_mut() {
        snap.instances = vec![live.clone(), cold.clone()];
    }
    s.instances.insert(live.container_id.clone(), live);
    s.instances.insert(cold.container_id.clone(), cold);
    s
}

#[test]
fn a11y_every_theme_renders_every_tab_with_chrome_intact() {
    // Render all 5 tabs under every registered theme: no panic, the rounded
    // btop tab chrome survives, and the tab labels are present. The single
    // background is structurally guaranteed by ui::draw painting the whole
    // frame with theme.bg first (no second competing surface).
    let themes = rocm_dash_tui::ui::theme::theme_names();
    assert!(
        themes.len() >= 15,
        "expected the full theme registry, got {}",
        themes.len()
    );
    for name in &themes {
        for tab in [
            ActiveTab::Home,
            ActiveTab::Rocm,
            ActiveTab::Serving,
            ActiveTab::Observe,
            ActiveTab::Chat,
        ] {
            let mut s = demo_state();
            s.active_tab = tab;
            s.theme_name = (*name).to_string();
            s.theme = rocm_dash_tui::ui::theme::Theme::from_name(name);
            let out = render(&mut s, 160, 50);
            assert!(
                out.contains('╭') && out.contains('╰'),
                "theme {name}/{tab:?}: rounded tab chrome missing"
            );
            assert!(
                out.contains("Serving"),
                "theme {name}/{tab:?}: tab labels missing"
            );
        }
    }
}

#[test]
fn demo_buffer_dump_all_tabs_inline_details_and_heroes() {
    // The `rocm dash --demo` equivalent: drive ui::draw over a connected demo
    // snapshot and confirm every surface renders. Excerpts are printed for the
    // transcript (run with --nocapture to view).
    // 1) all 5 tabs render their signature content.
    let checks: &[(ActiveTab, &str)] = &[
        (ActiveTab::Home, "GPU UTILIZATION"),
        (ActiveTab::Rocm, "ROCm actions"),
        (ActiveTab::Serving, "Serving actions"),
        (ActiveTab::Observe, "Node efficiency"),
        (ActiveTab::Chat, "Chat"),
    ];
    for (tab, needle) in checks {
        let mut s = demo_state();
        s.active_tab = *tab;
        let out = render(&mut s, 180, 50);
        assert!(out.contains(needle), "tab {tab:?}: {needle:?} missing");
        println!("DEMO_DUMP {tab:?}: ok (found {needle:?})");
    }

    // 2) inline Details: a manager renders in-pane on ROCm and Serving.
    let mut r = demo_state();
    r.active_tab = ActiveTab::Rocm;
    r.pane_focus = PaneFocus::Detail;
    r.install_manager = Some(rocm_dash_tui::ui::install_manager::InstallManagerState::default());
    let rout = render(&mut r, 180, 50);
    assert!(rout.contains("ROCm SDK"), "ROCm inline Details missing");
    assert!(
        rout.contains("ROCm actions"),
        "ROCm Actions list missing alongside"
    );
    println!("DEMO_DUMP Rocm inline Details: install_manager in-pane ✓");

    let mut sv = demo_state();
    sv.active_tab = ActiveTab::Serving;
    sv.pane_focus = PaneFocus::Detail;
    sv.serve_wizard = Some(rocm_dash_tui::ui::serve_wizard::ServeWizardState::default());
    let svout = render(&mut sv, 180, 50);
    assert!(
        svout.contains("Serve a model"),
        "Serving inline Details missing"
    );
    println!("DEMO_DUMP Serving inline Details: serve_wizard in-pane ✓");

    // 3) Observe heroes show tok/watt + live TTFT/TPOT, honest `—` for the cold
    //    instance.
    let mut o = demo_state();
    o.active_tab = ActiveTab::Observe;
    let oout = render(&mut o, 180, 50);
    assert!(
        oout.contains("Node efficiency") && oout.contains("Node throughput"),
        "heroes missing"
    );
    assert!(
        oout.contains("tok/W") || oout.contains("TOK/W"),
        "tok/watt missing"
    );
    assert!(oout.contains("150ms"), "live TTFT value missing");
    assert!(
        oout.contains("—"),
        "honest placeholder for cold instance missing"
    );
    println!("DEMO_DUMP Observe: heroes + tok/watt + live TTFT 150ms + honest — ✓");
}
