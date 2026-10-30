// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Instances Observe sub-panel — the per-instance metrics table. The detail
//! modal (model / partition / launch_args / env / log) lives in
//! `ui/instance_detail.rs`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Paragraph, Row, Table};

use rocm_dash_core::metrics::{
    Instance, InstanceStatus, ObservationFreshness, ObservationMetadata,
};

use crate::app::{AppState, ConnState};
use crate::ui::format;
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;
use crate::ui::widgets::trunc;

/// AI-serving per-instance table (Phase 5 Observe view).
///
/// Columns: model · throughput (tok/s) · tok/watt · TTFT · TPOT · power · queue
/// (running/waiting) · kv-cache%. Missing `Option` metrics render `—` (honest
/// placeholder, never a fabricated number); tok/watt is surfaced prominently
/// (accent). Keyboard + scroll-wheel select; left-click is not wired here.
pub fn draw_table(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    if state.instances.is_empty() {
        draw_empty(f, area, state, theme);
        return;
    }
    let instances = sorted_instances(&state.instances);
    let sel = clamp_sel(state.instance_sel, instances.len());
    let gpus: &[rocm_dash_core::metrics::GpuMetrics] =
        state.latest.as_ref().map_or(&[], |s| s.gpus.as_slice());

    let inner = panel::bento(
        f,
        area,
        Some("Instances · AI metrics"),
        BoxRole::Secondary,
        false,
        theme,
    );
    if inner.height == 0 {
        return;
    }

    // Show HELD_LEGEND only when at least one *displayed* row would actually
    // render a held marker. `Table` (no scroll state) draws rows from the
    // start of `instances` until the area runs out of height, one row per
    // instance after the header — so anything at or past `visible_rows`
    // never reaches the screen and must not be allowed to conjure a legend.
    // `inner.height - 1` (header only, legend not yet subtracted) is used as
    // an upper bound. This can overcount by exactly the one row the legend
    // itself claims: if the *only* held row sits at that last scanned index,
    // showing the legend consumes a row and pushes that exact row off-screen,
    // so the legend ends up explaining a marker that is no longer visible
    // (see `table_held_legend_shown_even_when_boundary_row_scrolls_off`).
    // That is intentional and the safe direction: the alternative (scanning
    // a tighter bound that already accounts for the legend's own row) can
    // never overcount, but can then *undercount* instead — a genuinely
    // visible marker with no legend at all, which is the failure mode this
    // mechanism exists to prevent. Overcounting by one boundary row is the
    // accepted cost of never doing that.
    // The finite checks mirror `gen_tps_cell`/`tokens_per_watt_cell`, which
    // never print a marker for `None`/non-finite values regardless of
    // freshness metadata. Both cells key off the same `gen_tps_observation`
    // (tok/W derives from the same per-tick `gen_tps` sample), but each
    // gates independently on its *own* value's finiteness — an instance can
    // have non-finite/missing `gen_tps` (no marker on TOK/S) while
    // `tokens_per_watt` is still finite and held (marker on TOK/W), so the
    // row needs the legend even though the `gen_tps` half of this check
    // alone would say no.
    let visible_rows = inner.height.saturating_sub(1) as usize;
    let any_held = instances.iter().take(visible_rows).any(|inst| {
        let held = inst
            .gen_tps_observation
            .as_ref()
            .is_some_and(|m| m.freshness == ObservationFreshness::Held);
        let is_held = |m: &Option<ObservationMetadata>| {
            m.as_ref()
                .is_some_and(|m| m.freshness == ObservationFreshness::Held)
        };
        (held
            && (inst.gen_tps.is_some_and(f64::is_finite)
                || inst.tokens_per_watt.is_some_and(f64::is_finite)))
            || (is_held(&inst.ttft_observation) && inst.ttft_ms.is_some_and(f64::is_finite))
            || (is_held(&inst.tpot_observation) && inst.tpot_ms.is_some_and(f64::is_finite))
    });
    let (table_area, legend_area) = if any_held {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(1)])
            .split(inner);
        (split[0], Some(split[1]))
    } else {
        (inner, None)
    };

    let header = Row::new([
        "MODEL", "TOK/S", "TOK/W", "TTFT", "TPOT", "POWER", "QUEUE", "KV%",
    ])
    .style(
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    );

    let dash = "—";
    let rows = instances.iter().enumerate().map(|(i, inst)| {
        let model = trunc(&inst.model_name, 22);
        let tps = format::gen_tps_cell(inst.gen_tps, inst.gen_tps_observation.as_ref());
        let tpw =
            format::tokens_per_watt_cell(inst.tokens_per_watt, inst.gen_tps_observation.as_ref());
        let ttft = format::latency_cell(inst.ttft_ms, inst.ttft_observation.as_ref());
        let tpot = format::latency_cell(inst.tpot_ms, inst.tpot_observation.as_ref());
        let power = rocm_dash_core::efficiency::instance_power_w(&inst.gpu_ids, gpus)
            .map_or_else(|| dash.to_string(), |w| format!("{w:.0}W"));
        let queue = match (inst.running_reqs, inst.waiting_reqs) {
            (None, None) => dash.to_string(),
            (r, w) => format!(
                "{}/{}",
                r.map_or_else(|| dash.to_string(), |v| v.to_string()),
                w.map_or_else(|| dash.to_string(), |v| v.to_string()),
            ),
        };
        let kv = inst
            .kv_cache_usage_pct
            .map_or_else(|| dash.to_string(), |v| format!("{v:.0}%"));

        let base = if i == sel {
            Style::default().fg(theme.fg).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.fg)
        };
        Row::new([
            Cell::from(model),
            Cell::from(tps),
            // tok/watt prominent (the headline efficiency metric).
            Cell::from(tpw).style(Style::default().fg(theme.accent)),
            Cell::from(ttft),
            Cell::from(tpot),
            Cell::from(power),
            Cell::from(queue),
            Cell::from(kv),
        ])
        .style(base)
    });

    let widths = [
        Constraint::Min(12),
        Constraint::Length(7),
        // Wide enough for "{v:.2} tok/W*" (tokens_per_watt_cell's longest
        // rendering) so the held marker never gets clipped mid-unit.
        Constraint::Length(12),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(5),
    ];
    let table = Table::new(rows, widths).header(header).column_spacing(1);
    f.render_widget(table, table_area);
    if let Some(legend_area) = legend_area {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format::HELD_LEGEND,
                Style::default().fg(theme.muted),
            ))),
            legend_area,
        );
    }
}

/// Sort instances deterministically by container_name so that an index from
/// AppState always maps to the same row in the table.
pub(in crate::ui) fn sorted_instances(
    instances: &std::collections::HashMap<String, Instance>,
) -> Vec<&Instance> {
    let mut v: Vec<&Instance> = instances.values().collect();
    v.sort_by(|a, b| a.container_name.cmp(&b.container_name));
    v
}

/// Clamp a selection index into `[0, len)`. Returns 0 for an empty list.
pub(in crate::ui) fn clamp_sel(sel: usize, len: usize) -> usize {
    if len == 0 { 0 } else { sel.min(len - 1) }
}

pub(in crate::ui) const fn status_meta(
    status: InstanceStatus,
    theme: &Theme,
) -> (ratatui::style::Color, &'static str) {
    // The text label (incl. the DOWNLOADING/LOADING/WARMUP startup phases) is
    // owned by `InstanceStatus::label`; the color by the shared
    // `Theme::instance_status_color`, so every screen agrees on it.
    (theme.instance_status_color(status), status.label())
}

fn draw_empty(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let body = if state.conn.is_in_process() {
        "no instance registry on this platform · vLLM containers aren't available here"
    } else {
        match &state.conn {
            ConnState::Connected { .. } => {
                "no instances · start daemon with --enable-docker on a host with vLLM containers"
            }
            _ => "waiting for daemon…",
        }
    };
    let inner = panel::bento(f, area, Some("Instances"), BoxRole::Neutral, false, theme);
    let p = Paragraph::new(Line::from(Span::styled(
        body,
        Style::default().fg(theme.muted),
    )));
    f.render_widget(p, inner);
}

#[cfg(test)]
pub(in crate::ui) mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};

    pub(in crate::ui) fn mk_inst(name: &str) -> Instance {
        Instance {
            container_id: format!("id-{name}"),
            container_name: name.to_string(),
            status: InstanceStatus::Running,
            model_name: "m".into(),
            gpu_ids: vec!["0".into()],
            partition_info: None,
            quantization: None,
            tensor_parallel_size: 1,
            port: Some(8000),
            vram_used_mb: 0,
            vram_total_mb: 0,
            kv_cache_usage_pct: None,
            running_reqs: None,
            waiting_reqs: None,
            gen_tps: None,
            gen_tps_observation: None,
            tokens_per_watt: None,
            ttft_ms: None,
            ttft_observation: None,
            tpot_ms: None,
            tpot_observation: None,
            launch_args: vec![],
            env_vars: BTreeMap::new(),
            log_file: None,
        }
    }

    fn map_with(names: &[&str]) -> HashMap<String, Instance> {
        let mut m = HashMap::new();
        for n in names {
            let inst = mk_inst(n);
            m.insert(inst.container_id.clone(), inst);
        }
        m
    }

    #[test]
    fn draw_empty_names_the_platform_reason_for_the_in_process_collector() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // The Windows in-process collector (see `ConnState::is_in_process`)
        // has no instance registry at all, ever — distinct from "a real
        // daemon is connected but nothing is running" or "still connecting".
        let mut state = mk_state(HashMap::new(), 0);
        state.conn = ConnState::Connected {
            host: "localhost".into(),
            version: ConnState::in_process_version("0.0.0"),
        };
        let mut term = Terminal::new(TestBackend::new(80, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let grid = buffer_text(&term);
        assert!(
            grid.contains("no instance registry on this platform"),
            "should name the platform limitation, not claim a daemon could be started; got:\n{grid}"
        );
        assert!(
            !grid.contains("start daemon"),
            "must not suggest starting a daemon that doesn't exist on this platform; got:\n{grid}"
        );
    }

    #[test]
    fn status_meta_maps_each_variant() {
        let theme = Theme::default_dark();
        assert_eq!(status_meta(InstanceStatus::Running, &theme).1, "RUNNING");
        assert_eq!(
            status_meta(InstanceStatus::Starting { phase: None }, &theme).1,
            "STARTING"
        );
        assert_eq!(
            status_meta(
                InstanceStatus::Starting {
                    phase: Some(rocm_dash_core::metrics::StartupPhase::Downloading)
                },
                &theme
            )
            .1,
            "DOWNLOADING"
        );
        assert_eq!(status_meta(InstanceStatus::Stopped, &theme).1, "STOPPED");
        assert_eq!(status_meta(InstanceStatus::Error, &theme).1, "ERROR");
        assert_eq!(status_meta(InstanceStatus::Unknown, &theme).1, "UNKNOWN");
    }

    #[test]
    fn clamp_sel_clamps_into_bounds() {
        assert_eq!(clamp_sel(0, 0), 0);
        assert_eq!(clamp_sel(7, 0), 0);
        assert_eq!(clamp_sel(0, 1), 0);
        assert_eq!(clamp_sel(0, 3), 0);
        assert_eq!(clamp_sel(2, 3), 2);
        assert_eq!(clamp_sel(99, 3), 2);
    }

    #[test]
    fn sorted_instances_orders_by_container_name() {
        let m = map_with(&["charlie", "alpha", "bravo"]);
        let v = sorted_instances(&m);
        assert_eq!(
            v.iter()
                .map(|i| i.container_name.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "bravo", "charlie"],
        );
    }

    #[test]
    fn selected_instance_lookup_by_sorted_index() {
        // HashMap insertion order is irrelevant — sorted_instances orders by name.
        let m = map_with(&["zeta", "alpha", "mu"]);
        let v = sorted_instances(&m);
        let sel = clamp_sel(1, v.len());
        assert_eq!(v[sel].container_name, "mu");

        // Out-of-range cursor clamps to the last item.
        let sel = clamp_sel(99, v.len());
        assert_eq!(v[sel].container_name, "zeta");
    }

    pub(in crate::ui) fn mk_state(instances: HashMap<String, Instance>, sel: usize) -> AppState {
        AppState {
            services_past_attempts: 0,
            managed_registry_probe: None,
            launcher_hosted: false,
            connect: "test".into(),
            conn: ConnState::Initial,
            latest: None,
            history: std::collections::VecDeque::new(),
            bench_rows: std::collections::VecDeque::new(),
            instances,
            has_received_snapshot: false,
            has_received_gpu_snapshot: false,
            active_tab: crate::app::ActiveTab::Observe,
            modal: crate::app::Modal::None,
            menu_sel: 0,
            palette_sel: 0,
            options_tab: 0,
            rocm_sel: 0,
            serving_sel: 0,
            pane_focus: crate::app::PaneFocus::Actions,
            instance_sel: sel,
            bench_sel: 0,
            gpu_sel: 0,
            gpu_scroll: 0,
            theme_name: "default-dark".into(),
            theme: Theme::default_dark(),
            theme_picker_sel: 0,
            instance_detail_scroll: 0,
            instance_detail_max_scroll: 0,
            console_scroll: 0,
            console_hscroll: 0,
            tick_count: 0,
            dock_logs_scroll: 0,
            last_dock_area: None,
            scrollbars: std::cell::RefCell::new(Vec::new()),
            scroll_drag: None,
            chat: Vec::new(),
            chat_input: String::new(),
            chat_sending: false,
            chat_dispatch: false,
            chat_endpoint_rebuild: None,
            chat_focused: false,
            chat_scroll: 0,
            chat_max_scroll: 0,
            chat_follow: true,
            chat_llm: None,
            chat_consent: crate::app::ChatConsent::Unavailable,
            chat_detect_offer: None,
            chat_detecting: false,
            chat_detect_dispatch: false,
            chat_detect_msg: None,
            chat_persist_dispatch: false,
            replay: None,
            simulated: false,
            last_body_area: None,
            last_tab_bar_area: None,
            last_header_badge: None,
            last_footer_chips: Vec::new(),
            jobs: rocm_dash_core::state::State::default(),
            services: None,
            serve_wizard: None,
            engine_manager: None,
            examine_manager: None,
            update_manager: None,
            install_manager: None,
            logs_view: None,
            runtime_manager: None,
            onboarding: None,
            automations_manager: None,
            command_screen: None,
            config_manager: None,
            bench_run: None,
            model_recipes: Vec::new(),
            runtimes: Vec::new(),
            automations: Vec::new(),
            tool_executor: None,
            bench_results_dir: None,
            should_quit: false,
            slash_tool: None,
            plan_request: None,
            approval: None,
            quit_confirm: None,
            active_provider: crate::app::ChatProvider::default(),
            provider_switch: None,
            update_status: crate::app::UpdateStatus::Unknown,
            update_status_pending: false,
            update_check_due_at: std::time::Instant::now(),
        }
    }

    /// Flatten a rendered TestBackend buffer into one newline-joined string so
    /// substring assertions can confirm what reached the screen.
    pub(in crate::ui) fn buffer_text(
        term: &ratatui::Terminal<ratatui::backend::TestBackend>,
    ) -> String {
        let buf = term.backend().buffer();
        let area = buf.area;
        let mut out = String::new();
        for y in 0..area.height {
            for x in 0..area.width {
                out.push_str(buf.cell((x, y)).unwrap().symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn nonfinite_ttft_tpot_render_dash_never_nan() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // A NaN TTFT and an infinite TPOT must render the `—` placeholder via the
        // same is_finite() guard tokens_per_watt uses — never "NaNms"/"infms".
        let mut inst = mk_inst("vllm");
        inst.ttft_ms = Some(f64::NAN);
        inst.tpot_ms = Some(f64::INFINITY);

        let mut m = HashMap::new();
        m.insert(inst.container_id.clone(), inst);
        let state = mk_state(m, 0);

        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(!out.contains("NaN"), "NaN leaked to the screen:\n{out}");
        assert!(!out.contains("infms"), "inf leaked to the screen:\n{out}");
        assert!(
            out.contains('—'),
            "non-finite metrics must render the em-dash placeholder:\n{out}"
        );
    }

    // ── EAI-7960: observation-aware renderer tests ───────────────────────────
    // RED: assert held marker "*" in table output, freshness in detail.
    // These fail until draw_table / render_summary use the new
    // gen_tps_cell / gen_tps_compact / gen_tps_detail_freshness formatters.

    use chrono::{TimeZone, Utc};
    use rocm_dash_core::metrics::{ObservationFreshness, ObservationMetadata};

    pub(in crate::ui) fn mk_inst_obs(
        name: &str,
        gen_tps: Option<f64>,
        obs: Option<ObservationMetadata>,
    ) -> Instance {
        let mut i = mk_inst(name);
        i.gen_tps = gen_tps;
        i.gen_tps_observation = obs;
        i
    }

    pub(in crate::ui) fn obs(
        freshness: ObservationFreshness,
        secs_before_snap: i64,
    ) -> ObservationMetadata {
        // snap_ts will be SNAP_TS; this obs was secs_before_snap seconds earlier.
        let snap_secs = 1_700_000_060_i64;
        ObservationMetadata {
            observed_at: Utc.timestamp_opt(snap_secs - secs_before_snap, 0).unwrap(),
            freshness,
        }
    }

    const SNAP_SECS: i64 = 1_700_000_060;

    pub(in crate::ui) fn state_with_snap(inst: Instance) -> AppState {
        let mut m = HashMap::new();
        m.insert(inst.container_id.clone(), inst.clone());
        let mut state = mk_state(m, 0);
        let snap = rocm_dash_core::metrics::Snapshot {
            timestamp: Utc.timestamp_opt(SNAP_SECS, 0).unwrap(),
            instances: vec![inst],
            ..Default::default()
        };
        state.latest = Some(snap);
        state
    }

    #[test]
    fn table_held_gen_tps_shows_held_marker() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs(
            "held",
            Some(123.0),
            Some(obs(ObservationFreshness::Held, 30)),
        );
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("123*"),
            "held gen_tps must show '123*' in table; got:\n{out}"
        );
    }

    #[test]
    fn table_held_gen_tps_shows_held_marker_on_tok_w_too() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut inst = mk_inst_obs(
            "held-tpw",
            Some(123.0),
            Some(obs(ObservationFreshness::Held, 30)),
        );
        inst.tokens_per_watt = Some(0.42);
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("0.42 tok/W*"),
            "held tok/W must carry the held marker in the table; got:\n{out}"
        );
    }

    #[test]
    fn table_shows_held_legend_for_tok_w_even_when_gen_tps_is_missing() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // gen_tps itself is None (so TOK/S renders no marker at all), but
        // tokens_per_watt is finite and held — the legend must still appear
        // because the TOK/W cell independently prints a marker for it.
        let mut inst = mk_inst_obs(
            "held-tpw-only",
            None,
            Some(obs(ObservationFreshness::Held, 30)),
        );
        inst.tokens_per_watt = Some(0.42);
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("0.42 tok/W*"),
            "held tok/W must carry the held marker even without gen_tps; got:\n{out}"
        );
        assert!(
            out.contains(format::HELD_LEGEND),
            "HELD_LEGEND must appear when only the tok/W cell shows a held marker; got:\n{out}"
        );
    }

    #[test]
    fn table_at_80_columns_keeps_tok_w_marker_and_model_readable() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // TOK/W was widened to Length(12) to fit its longest rendering
        // ("{v:.2} tok/W*") without clipping the held marker — at the cost
        // of MODEL's headroom above its Min(12) floor. Every other render
        // test in this file uses a 160-column backend; cover the narrow
        // 80-column case where that headroom is tightest.
        let mut inst = mk_inst_obs(
            "narrow",
            Some(123.0),
            Some(obs(ObservationFreshness::Held, 30)),
        );
        inst.model_name = "llama-3.1-8b-instruct".into();
        inst.tokens_per_watt = Some(0.42);
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(80, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("0.42 tok/W*"),
            "held tok/W marker must not be clipped at 80 columns; got:\n{out}"
        );
        assert!(
            out.contains("llama-3.1-8b"),
            "MODEL must stay readable above its Min(12) floor at 80 columns; got:\n{out}"
        );
    }

    #[test]
    fn table_fresh_gen_tps_has_no_held_marker() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs(
            "fresh",
            Some(456.0),
            Some(obs(ObservationFreshness::Fresh, 5)),
        );
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("456"),
            "fresh gen_tps value must appear; got:\n{out}"
        );
        assert!(
            !out.contains("456*"),
            "fresh must NOT have held marker; got:\n{out}"
        );
    }

    #[test]
    fn table_legacy_meta_none_has_no_held_marker() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs("legacy", Some(789.0), None);
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("789") && !out.contains("789*"),
            "legacy (no metadata) must NOT show held marker; got:\n{out}"
        );
    }

    #[test]
    fn table_held_legend_visible_when_gen_tps_held() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs(
            "held",
            Some(123.0),
            Some(obs(ObservationFreshness::Held, 30)),
        );
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains(format::HELD_LEGEND),
            "HELD_LEGEND must appear when a shown gen_tps is held; got:\n{out}"
        );
    }

    #[test]
    fn table_held_ttft_shows_marker_and_legend_even_with_fresh_gen_tps() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut inst = mk_inst_obs(
            "idle",
            Some(456.0),
            Some(obs(ObservationFreshness::Fresh, 5)),
        );
        inst.ttft_ms = Some(150.0);
        inst.ttft_observation = Some(obs(ObservationFreshness::Held, 30));
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains("150ms*"),
            "held TTFT must be marked; got:
{out}"
        );
        assert!(
            out.contains(format::HELD_LEGEND),
            "legend must explain a held TTFT marker; got:
{out}"
        );
    }

    #[test]
    fn table_held_legend_absent_when_all_fresh() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs(
            "fresh",
            Some(456.0),
            Some(obs(ObservationFreshness::Fresh, 5)),
        );
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            !out.contains(format::HELD_LEGEND),
            "HELD_LEGEND must not appear when all shown gen_tps are fresh; got:\n{out}"
        );
    }

    #[test]
    fn table_held_legend_absent_for_legacy_none_metadata() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs("legacy", Some(789.0), None);
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            !out.contains(format::HELD_LEGEND),
            "HELD_LEGEND must not appear for legacy None metadata; got:\n{out}"
        );
    }

    #[test]
    fn table_held_legend_absent_for_non_finite_gen_tps() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // gen_tps_cell never prints HELD_MARKER for a non-finite value (it
        // renders "—" instead), so held metadata here must not conjure a
        // legend either.
        let inst = mk_inst_obs(
            "nonfinite",
            Some(f64::NAN),
            Some(obs(ObservationFreshness::Held, 30)),
        );
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            !out.contains(format::HELD_LEGEND),
            "HELD_LEGEND must not appear for a non-finite gen_tps; got:\n{out}"
        );
    }

    #[test]
    fn table_held_legend_absent_when_held_row_is_off_screen() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Twenty instances, sorted by (zero-padded) name; only the very last
        // one is held. A short viewport only fits a few rows, so that held
        // row never reaches the screen and must not add a legend.
        let mut m = HashMap::new();
        for n in 0..20 {
            let name = format!("n{n:02}");
            let held = n == 19;
            let inst = mk_inst_obs(
                &name,
                Some(1.0),
                if held {
                    Some(obs(ObservationFreshness::Held, 30))
                } else {
                    Some(obs(ObservationFreshness::Fresh, 5))
                },
            );
            m.insert(inst.container_id.clone(), inst);
        }
        let state = mk_state(m, 0);
        let mut term = Terminal::new(TestBackend::new(160, 8)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            !out.contains("n19"),
            "test setup assumption broken: the held row must be off-screen; got:\n{out}"
        );
        assert!(
            !out.contains(format::HELD_LEGEND),
            "HELD_LEGEND must not appear when the only held row is off-screen; got:\n{out}"
        );
    }

    #[test]
    fn table_held_legend_shown_even_when_boundary_row_scrolls_off() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Nine instances in a viewport sized so the any_held scan bound
        // (`inner.height - 1`) is exactly 9, i.e. its last scanned index is
        // 8 — the *only* held row. The scan finds it and shows the legend,
        // but showing the legend claims a row, shrinking the actual visible
        // data rows to 8 (indices 0..=7) and pushing this row off-screen.
        // This is the documented, accepted trade-off (see the comment above
        // `visible_rows` in `draw_table`): never hide a legend a visible
        // marker needs, even if that means occasionally showing one whose
        // triggering row is no longer on screen.
        let mut m = HashMap::new();
        for n in 0..9 {
            let name = format!("n{n:02}");
            let held = n == 8;
            let inst = mk_inst_obs(
                &name,
                Some(1.0),
                if held {
                    Some(obs(ObservationFreshness::Held, 30))
                } else {
                    Some(obs(ObservationFreshness::Fresh, 5))
                },
            );
            m.insert(inst.container_id.clone(), inst);
        }
        let state = mk_state(m, 0);
        let mut term = Terminal::new(TestBackend::new(160, 13)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            !out.contains("n08"),
            "test setup assumption broken: the boundary row must scroll off once the legend claims its row; got:\n{out}"
        );
        assert!(
            out.contains(format::HELD_LEGEND),
            "HELD_LEGEND must still show for a boundary-row match, even though that row is no longer visible; got:\n{out}"
        );
    }

    #[test]
    fn table_none_gen_tps_shows_em_dash() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let inst = mk_inst_obs("none-tps", None, None);
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        assert!(
            out.contains('—'),
            "missing gen_tps must render em-dash; got:\n{out}"
        );
    }

    #[test]
    fn table_zero_gen_tps_is_not_dash() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Real zero gen_tps (counter is active but rate is zero) must NOT collapse to dash.
        let inst = mk_inst_obs("zero-tps", Some(0.0), None);
        let state = state_with_snap(inst);
        let mut term = Terminal::new(TestBackend::new(160, 20)).unwrap();
        term.draw(|f| draw_table(f, f.area(), &state, &state.theme))
            .unwrap();
        let out = buffer_text(&term);
        // "0" appears in the TOK/S column; the em-dash should NOT appear in that slot.
        // (em-dash may appear for other missing cols, so we check the tok/s specific cell)
        assert!(
            out.contains("0 ") || out.contains("0\n") || out.contains("0*"),
            "zero gen_tps must render as '0', not '—'; got:\n{out}"
        );
    }
}
