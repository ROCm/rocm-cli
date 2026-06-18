//! Instances tab — full-screen instance grid with kv-cache / requests / args,
//! plus a detail modal showing model / partition / launch_args / env / log.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use rocm_dash_core::metrics::{Instance, InstanceStatus};

use crate::app::{AppState, ConnState, KeyAction};
use crate::ui::format;
use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::theme::Theme;
use crate::ui::widgets::trunc;

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    if state.instances.is_empty() {
        draw_empty(f, area, state, theme);
        return;
    }

    let instances = sorted_instances(&state.instances);
    let sel = clamp_sel(state.instance_sel, instances.len());

    // When we have ≥2 instances AND ≥3 snapshots in history, surface a
    // kv-cache × time heatmap above the card grid. Single-instance / cold-
    // start cases skip the heatmap so we don't waste rows on something
    // tautological.
    let show_heatmap = instances.len() >= 2 && state.history.len() >= 3;
    let (heatmap_area, grid_area) = if show_heatmap {
        let heatmap_rows = compute_heatmap_height(instances.len(), area.height);
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(heatmap_rows), Constraint::Min(0)])
            .split(area);
        (Some(split[0]), split[1])
    } else {
        (None, area)
    };

    if let Some(heat_area) = heatmap_area {
        draw_kv_heatmap(f, heat_area, state, &instances, theme);
    }
    draw_card_grid(f, grid_area, &instances, sel, theme);
}

/// How tall to make the heatmap block.
/// Each row of the table = one instance; +2 for borders + 1 for footer hint.
fn compute_heatmap_height(n_instances: usize, total_height: u16) -> u16 {
    // Cap at total/2 so the card grid still gets meaningful space.
    let max = (total_height / 2).max(5);
    (n_instances as u16 + 3).min(max).max(5)
}

fn draw_card_grid(f: &mut Frame, area: Rect, instances: &[&Instance], sel: usize, theme: &Theme) {
    let cols = pick_cols(area.width);
    let rows = instances.len().div_ceil(cols);
    if rows == 0 {
        return;
    }
    let row_constraints: Vec<Constraint> = (0..rows)
        .map(|_| Constraint::Ratio(1, rows as u32))
        .collect();
    let row_slots = Layout::default()
        .direction(Direction::Vertical)
        .constraints(row_constraints)
        .split(area);

    for (row_idx, row_slot) in row_slots.iter().enumerate() {
        let col_constraints: Vec<Constraint> = (0..cols)
            .map(|_| Constraint::Ratio(1, cols as u32))
            .collect();
        let col_slots = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(col_constraints)
            .split(*row_slot);
        for (col_idx, cell) in col_slots.iter().enumerate() {
            let idx = row_idx * cols + col_idx;
            if let Some(inst) = instances.get(idx) {
                draw_card(f, *cell, inst, theme, idx == sel);
            }
        }
    }
}

fn draw_kv_heatmap(
    f: &mut Frame,
    area: Rect,
    state: &AppState,
    instances: &[&Instance],
    theme: &Theme,
) {
    use crate::ui::heatmap::Heatmap;

    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(
            " kv-cache % · {} instances · last {} ticks ",
            instances.len(),
            state.history.len(),
        ))
        .border_style(theme.border_style())
        .title_style(theme.title_style());
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height == 0 {
        return;
    }

    let rows = build_kv_heatmap_rows(&state.history, instances);
    if rows.is_empty() {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "no kv-cache samples yet — start a vLLM container and wait a tick",
                Style::default().fg(theme.muted),
            ))),
            inner,
        );
        return;
    }

    let heat = Heatmap::new(&rows)
        .stops(theme.ok, theme.warn, theme.err)
        .track_bg(theme.surface_2)
        .label_style(Style::default().fg(theme.muted))
        .label_width(label_width_for(instances));
    f.render_widget(heat, inner);
}

/// Width budget for instance labels in the heatmap, capped so the data
/// region stays usable on narrow terminals.
fn label_width_for(instances: &[&Instance]) -> u16 {
    instances
        .iter()
        .map(|i| i.container_name.chars().count() as u16)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
        .clamp(8, 20)
}

/// Build one heatmap row per instance: kv-cache % over each snapshot in
/// history. Instances missing from a given snapshot contribute 0 at that
/// column (renders as `track_bg` because of the zero-value guard).
/// Pure — exposed for tests.
pub(crate) fn build_kv_heatmap_rows(
    history: &std::collections::VecDeque<rocm_dash_core::metrics::Snapshot>,
    instances: &[&Instance],
) -> Vec<crate::ui::heatmap::HeatmapRow> {
    use crate::ui::heatmap::HeatmapRow;

    let mut out = Vec::with_capacity(instances.len());
    for inst in instances {
        let id = &inst.container_id;
        let label = inst.container_name.clone();
        let data: Vec<f64> = history
            .iter()
            .map(|snap| {
                snap.instances
                    .iter()
                    .find(|i| &i.container_id == id)
                    .and_then(|i| i.kv_cache_usage_pct.map(f64::from))
                    .unwrap_or(0.0)
            })
            .collect();
        out.push(HeatmapRow::new(label, data, 100.0));
    }
    out
}

const fn pick_cols(width: u16) -> usize {
    if width >= 160 {
        3
    } else if width >= 100 {
        2
    } else {
        1
    }
}

/// Sort instances deterministically by container_name so that an index from
/// AppState always maps to the same card in the grid.
fn sorted_instances(instances: &std::collections::HashMap<String, Instance>) -> Vec<&Instance> {
    let mut v: Vec<&Instance> = instances.values().collect();
    v.sort_by(|a, b| a.container_name.cmp(&b.container_name));
    v
}

/// Clamp a selection index into `[0, len)`. Returns 0 for an empty list.
fn clamp_sel(sel: usize, len: usize) -> usize {
    if len == 0 { 0 } else { sel.min(len - 1) }
}

const fn status_meta(
    status: InstanceStatus,
    theme: &Theme,
) -> (ratatui::style::Color, &'static str) {
    match status {
        InstanceStatus::Running => (theme.ok, "RUNNING"),
        InstanceStatus::Starting => (theme.warn, "STARTING"),
        InstanceStatus::Stopped => (theme.err, "STOPPED"),
        InstanceStatus::Error => (theme.err, "ERROR"),
        InstanceStatus::Unknown => (theme.muted, "UNKNOWN"),
    }
}

fn draw_empty(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let body = match &state.conn {
        ConnState::Connected { .. } => {
            "no instances · start daemon with --enable-docker on a host with vLLM containers"
        }
        _ => "waiting for daemon…",
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Instances ")
        .border_style(theme.border_style())
        .title_style(theme.title_style());
    let p = Paragraph::new(Line::from(Span::styled(
        body,
        Style::default().fg(theme.muted),
    )))
    .block(block);
    f.render_widget(p, area);
}

fn draw_card(f: &mut Frame, area: Rect, inst: &Instance, theme: &Theme, selected: bool) {
    let (status_color, status_text) = status_meta(inst.status, theme);
    let name = trunc(&inst.container_name, 24);
    let title = format!(" {name} · {status_text} ");

    let border_style = if selected {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        theme.border_style()
    };
    let mut title_style = Style::default()
        .fg(status_color)
        .add_modifier(Modifier::BOLD);
    if selected {
        title_style = title_style.add_modifier(Modifier::BOLD);
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(border_style)
        .title_style(title_style);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let inner_w = inner.width as usize;

    // Too short -> compact one-liner. Highlight still flows through the border.
    if inner.height < 5 {
        let line = compact_line(inst, theme, inner_w);
        f.render_widget(Paragraph::new(line), inner);
        return;
    }

    let port_str = inst.port.map_or_else(|| "-".into(), |p| p.to_string());
    let gpus_str = if inst.gpu_ids.is_empty() {
        "-".to_string()
    } else if inst.gpu_ids.len() == 1 {
        inst.gpu_ids[0].clone()
    } else {
        inst.gpu_ids.join(",")
    };

    let kv = format::pct_opt(inst.kv_cache_usage_pct);
    let run = format::reqs_opt(inst.running_reqs);
    let wait = format::reqs_opt(inst.waiting_reqs);

    let mut lines: Vec<Line> = Vec::with_capacity(8);

    // 1. model
    lines.push(Line::from(vec![
        Span::styled("model ", Style::default().fg(theme.muted)),
        Span::styled(
            trunc(&inst.model_name, inner_w.saturating_sub(6).max(1)),
            Style::default().fg(theme.fg),
        ),
    ]));

    // 2. port · tp · gpus
    lines.push(Line::from(vec![
        Span::styled(format!("port {port_str}"), Style::default().fg(theme.fg)),
        Span::styled(" · ", Style::default().fg(theme.muted)),
        Span::styled(
            format!("tp {}", inst.tensor_parallel_size),
            Style::default().fg(theme.fg),
        ),
        Span::styled(" · ", Style::default().fg(theme.muted)),
        Span::styled(
            format!("gpus {}", trunc(&gpus_str, 20)),
            Style::default().fg(theme.accent),
        ),
    ]));

    // 3. kv_cache / run / wait
    lines.push(Line::from(vec![
        Span::styled("kv_cache ", Style::default().fg(theme.muted)),
        Span::styled(kv, Style::default().fg(theme.fg)),
        Span::styled(" · run ", Style::default().fg(theme.muted)),
        Span::styled(run, Style::default().fg(theme.fg)),
        Span::styled(" · wait ", Style::default().fg(theme.muted)),
        Span::styled(wait, Style::default().fg(theme.fg)),
    ]));

    // 4. efficiency: tok/W · gen throughput
    lines.push(Line::from(vec![
        Span::styled("tok/W ", Style::default().fg(theme.muted)),
        Span::styled(
            format::tokens_per_watt(inst.tokens_per_watt),
            Style::default().fg(theme.accent),
        ),
        Span::styled(" · gen ", Style::default().fg(theme.muted)),
        Span::styled(format::tps_opt(inst.gen_tps), Style::default().fg(theme.fg)),
    ]));

    // 5. vram (only if total > 0)
    if inst.vram_total_mb > 0 {
        lines.push(Line::from(vec![
            Span::styled("vram ", Style::default().fg(theme.muted)),
            Span::styled(
                format::mib_pair(inst.vram_used_mb, inst.vram_total_mb),
                Style::default().fg(theme.fg),
            ),
        ]));
    }

    // 5. args
    let args_joined = inst
        .launch_args
        .iter()
        .take(4)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    let args_display = if args_joined.is_empty() {
        "(none)".to_string()
    } else {
        trunc(&args_joined, inner_w.saturating_sub(6).max(1))
    };
    lines.push(Line::from(vec![
        Span::styled("args: ", Style::default().fg(theme.muted)),
        Span::styled(args_display, Style::default().fg(theme.muted)),
    ]));

    // 6. env count
    lines.push(Line::from(vec![
        Span::styled("env: ", Style::default().fg(theme.muted)),
        Span::styled(
            format!("{} vars", inst.env_vars.len()),
            Style::default().fg(theme.muted),
        ),
    ]));

    // 7. log file (optional)
    if let Some(log) = inst.log_file.as_deref() {
        lines.push(Line::from(vec![
            Span::styled("log: ", Style::default().fg(theme.muted)),
            Span::styled(
                trunc(log, inner_w.saturating_sub(5).max(1)),
                Style::default().fg(theme.muted),
            ),
        ]));
    }

    f.render_widget(Paragraph::new(lines), inner);
}

fn compact_line<'a>(inst: &'a Instance, theme: &Theme, max_w: usize) -> Line<'a> {
    let (status_color, _) = status_meta(inst.status, theme);
    let port = inst.port.map_or_else(|| "-".into(), |p| p.to_string());
    let gpus = if inst.gpu_ids.is_empty() {
        "-".to_string()
    } else {
        inst.gpu_ids.join(",")
    };
    let raw = format!(
        "{} · {} · :{} · tp{} · gpus {}",
        inst.container_name, inst.model_name, port, inst.tensor_parallel_size, gpus
    );
    Line::from(Span::styled(
        trunc(&raw, max_w),
        Style::default().fg(status_color),
    ))
}

/// Detail modal: summary + launch_args + env_vars + log footer.
///
/// Resolve a click at `(x, y)` inside the Instances tab body. Returns a
/// `KeyAction` to dispatch, or `None` when the click misses everything
/// actionable.
///
/// Re-runs the same Layout split as `draw` so card rects line up exactly with
/// what the user sees. Clicking the already-selected card opens the detail
/// modal (acts as a double-click affordance); clicking any other card moves
/// the selection cursor by the delta to that card.
pub fn hit_test(area: Rect, x: u16, y: u16, state: &AppState) -> Option<KeyAction> {
    if state.instances.is_empty() {
        return None;
    }
    if x < area.x || x >= area.x + area.width || y < area.y || y >= area.y + area.height {
        return None;
    }

    let instances = sorted_instances(&state.instances);
    let cols = pick_cols(area.width);
    let rows = instances.len().div_ceil(cols);
    if rows == 0 || cols == 0 {
        return None;
    }

    let sel = clamp_sel(state.instance_sel, instances.len());

    let row_constraints: Vec<Constraint> = (0..rows)
        .map(|_| Constraint::Ratio(1, rows as u32))
        .collect();
    let row_slots = Layout::default()
        .direction(Direction::Vertical)
        .constraints(row_constraints)
        .split(area);

    for (row_idx, row_slot) in row_slots.iter().enumerate() {
        let col_constraints: Vec<Constraint> = (0..cols)
            .map(|_| Constraint::Ratio(1, cols as u32))
            .collect();
        let col_slots = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(col_constraints)
            .split(*row_slot);
        for (col_idx, cell) in col_slots.iter().enumerate() {
            let idx = row_idx * cols + col_idx;
            if idx >= instances.len() {
                continue;
            }
            if point_in_rect(*cell, x, y) {
                if idx == sel {
                    return Some(KeyAction::OpenDetail);
                }
                let delta = idx.cast_signed() - sel.cast_signed();
                return Some(KeyAction::Move(delta));
            }
        }
    }
    None
}

/// Pure point-in-rect check using half-open coordinates (right/bottom edges
/// are exclusive), matching ratatui's own rect semantics.
const fn point_in_rect(r: Rect, x: u16, y: u16) -> bool {
    x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height
}

pub fn draw_detail(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let popup = centered_rect(85, 85, 120, 36, area);

    if state.instances.is_empty() {
        let inner = draw_popup_frame(f, popup, " Instance · (no selection) ", theme);
        let p = Paragraph::new(Line::from(Span::styled(
            "no instances to show",
            Style::default().fg(theme.muted),
        )));
        f.render_widget(p, inner);
        return;
    }

    let instances = sorted_instances(&state.instances);
    let sel = clamp_sel(state.instance_sel, instances.len());
    let Some(inst) = instances.get(sel) else {
        let inner = draw_popup_frame(f, popup, " Instance · (no selection) ", theme);
        let p = Paragraph::new(Line::from(Span::styled(
            "no selection",
            Style::default().fg(theme.muted),
        )));
        f.render_widget(p, inner);
        return;
    };

    let title = format!(" Instance · {} ", inst.container_name);
    let inner = draw_popup_frame(f, popup, &title, theme);
    if inner.height == 0 || inner.width == 0 {
        return;
    }

    // Vertical: summary (3) | body (min) | footer (1)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(inner);

    render_summary(f, chunks[0], inst, theme);
    render_body(f, chunks[1], inst, theme);
    render_footer(f, chunks[2], inst, theme);
}

fn render_summary(f: &mut Frame, area: Rect, inst: &Instance, theme: &Theme) {
    let (status_color, status_text) = status_meta(inst.status, theme);
    let muted = Style::default().fg(theme.muted);
    let fg = Style::default().fg(theme.fg);

    let id_w = (area.width as usize).saturating_sub(16).max(8);
    let container_id = trunc(&inst.container_id, id_w);
    let partition = inst.partition_info.as_deref().unwrap_or("-");
    let quant = inst.quantization.as_deref().unwrap_or("-");
    let port = inst.port.map_or_else(|| "-".into(), |p| p.to_string());
    let gpus = if inst.gpu_ids.is_empty() {
        "-".to_string()
    } else {
        inst.gpu_ids.join(",")
    };

    let lines = vec![
        Line::from(vec![
            Span::styled(
                format!(" {status_text} "),
                Style::default()
                    .fg(theme.bg)
                    .bg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled("id: ", muted),
            Span::styled(container_id, fg),
            Span::raw("  "),
            Span::styled("port: ", muted),
            Span::styled(port, fg),
            Span::raw("  "),
            Span::styled("tp: ", muted),
            Span::styled(inst.tensor_parallel_size.to_string(), fg),
        ]),
        Line::from(vec![
            Span::styled("model: ", muted),
            Span::styled(inst.model_name.clone(), fg),
            Span::raw("  "),
            Span::styled("gpus: ", muted),
            Span::styled(gpus, Style::default().fg(theme.accent)),
            Span::raw("  "),
            Span::styled("tok/W: ", muted),
            Span::styled(
                format::tokens_per_watt(inst.tokens_per_watt),
                Style::default().fg(theme.accent),
            ),
            Span::raw("  "),
            Span::styled("gen: ", muted),
            Span::styled(format::tps_opt(inst.gen_tps), fg),
        ]),
        Line::from(vec![
            Span::styled("partition: ", muted),
            Span::styled(partition.to_string(), fg),
            Span::raw("  "),
            Span::styled("quantization: ", muted),
            Span::styled(quant.to_string(), fg),
            Span::raw("  "),
            Span::styled("vram: ", muted),
            Span::styled(format::mib_pair(inst.vram_used_mb, inst.vram_total_mb), fg),
        ]),
    ];

    let p = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn render_body(f: &mut Frame, area: Rect, inst: &Instance, theme: &Theme) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
        .split(area);

    // launch_args (left)
    let args_block = Block::default()
        .borders(Borders::ALL)
        .title(" launch_args ")
        .border_style(theme.border_style())
        .title_style(theme.title_style());
    let args_inner = args_block.inner(chunks[0]);
    f.render_widget(args_block, chunks[0]);

    let args_lines: Vec<Line> = if inst.launch_args.is_empty() {
        vec![Line::from(Span::styled(
            "(none)",
            Style::default().fg(theme.muted),
        ))]
    } else {
        inst.launch_args
            .iter()
            .map(|a| Line::from(Span::styled(a.clone(), Style::default().fg(theme.fg))))
            .collect()
    };
    f.render_widget(
        Paragraph::new(args_lines).wrap(Wrap { trim: false }),
        args_inner,
    );

    // env_vars (right). BTreeMap iterates sorted by key.
    let env_block = Block::default()
        .borders(Borders::ALL)
        .title(" env_vars ")
        .border_style(theme.border_style())
        .title_style(theme.title_style());
    let env_inner = env_block.inner(chunks[1]);
    f.render_widget(env_block, chunks[1]);

    let env_lines: Vec<Line> = if inst.env_vars.is_empty() {
        vec![Line::from(Span::styled(
            "(none)",
            Style::default().fg(theme.muted),
        ))]
    } else {
        inst.env_vars
            .iter()
            .map(|(k, v)| {
                Line::from(vec![
                    Span::styled(k.clone(), Style::default().fg(theme.accent)),
                    Span::styled("=", Style::default().fg(theme.muted)),
                    Span::styled(v.clone(), Style::default().fg(theme.fg)),
                ])
            })
            .collect()
    };
    f.render_widget(
        Paragraph::new(env_lines).wrap(Wrap { trim: false }),
        env_inner,
    );
}

fn render_footer(f: &mut Frame, area: Rect, inst: &Instance, theme: &Theme) {
    let log = inst.log_file.as_deref().unwrap_or("-");
    let p = Paragraph::new(Line::from(vec![
        Span::styled("log: ", Style::default().fg(theme.muted)),
        Span::styled(log.to_string(), Style::default().fg(theme.muted)),
    ]));
    f.render_widget(p, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};

    fn mk_inst(name: &str) -> Instance {
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
            tokens_per_watt: None,
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
    fn pick_cols_scales_with_width() {
        assert_eq!(pick_cols(60), 1);
        assert_eq!(pick_cols(99), 1);
        assert_eq!(pick_cols(100), 2);
        assert_eq!(pick_cols(159), 2);
        assert_eq!(pick_cols(160), 3);
        assert_eq!(pick_cols(300), 3);
    }

    #[test]
    fn status_meta_maps_each_variant() {
        let theme = Theme::default_dark();
        assert_eq!(status_meta(InstanceStatus::Running, &theme).1, "RUNNING");
        assert_eq!(status_meta(InstanceStatus::Starting, &theme).1, "STARTING");
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

    #[test]
    fn point_in_rect_uses_half_open_semantics() {
        let r = Rect::new(10, 5, 4, 3);
        assert!(point_in_rect(r, 10, 5));
        assert!(point_in_rect(r, 13, 7));
        // Right and bottom edges are exclusive.
        assert!(!point_in_rect(r, 14, 5));
        assert!(!point_in_rect(r, 10, 8));
        // Outside.
        assert!(!point_in_rect(r, 9, 5));
        assert!(!point_in_rect(r, 10, 4));
    }

    fn mk_state(instances: HashMap<String, Instance>, sel: usize) -> AppState {
        AppState {
            connect: "test".into(),
            conn: ConnState::Initial,
            latest: None,
            history: std::collections::VecDeque::new(),
            bench_rows: std::collections::VecDeque::new(),
            instances,
            active_tab: crate::app::ActiveTab::Instances,
            modal: crate::app::Modal::None,
            instance_sel: sel,
            bench_sel: 0,
            gpu_sel: 0,
            gpu_scroll: 0,
            theme_name: "default-dark".into(),
            theme: Theme::default_dark(),
            theme_picker_sel: 0,
            bench_detail_scroll: 0,
            chat: Vec::new(),
            chat_input: String::new(),
            chat_sending: false,
            chat_dispatch: false,
            chat_focused: false,
            chat_scroll: 0,
            chat_llm: None,
            chat_consent: crate::app::ChatConsent::Unavailable,
            chat_detect_offer: None,
            chat_detecting: false,
            chat_detect_dispatch: false,
            chat_detect_msg: None,
            chat_persist_dispatch: false,
            replay: None,
            last_body_area: None,
            last_tab_bar_area: None,
            jobs: rocm_dash_core::state::State::default(),
            services: None,
            serve_wizard: None,
            engine_manager: None,
            doctor_manager: None,
            update_manager: None,
            install_manager: None,
            logs_view: None,
            runtime_manager: None,
            onboarding: None,
            automations_manager: None,
            command_screen: None,
            config_manager: None,
            model_recipes: Vec::new(),
            runtimes: Vec::new(),
            automations: Vec::new(),
        }
    }

    fn mk_state_with(names: &[&str], sel: usize) -> AppState {
        mk_state(map_with(names), sel)
    }

    #[test]
    fn hit_test_returns_none_when_empty() {
        let s = mk_state(HashMap::new(), 0);
        let area = Rect::new(0, 0, 80, 20);
        assert_eq!(hit_test(area, 5, 5, &s), None);
    }

    #[test]
    fn hit_test_returns_none_when_outside_area() {
        let s = mk_state_with(&["a", "b"], 0);
        let area = Rect::new(10, 5, 80, 20);
        // Above area.
        assert_eq!(hit_test(area, 50, 4, &s), None);
        // Left of area.
        assert_eq!(hit_test(area, 9, 10, &s), None);
        // Right edge exclusive.
        assert_eq!(hit_test(area, 90, 10, &s), None);
        // Bottom edge exclusive.
        assert_eq!(hit_test(area, 50, 25, &s), None);
    }

    #[test]
    fn hit_test_on_selected_card_opens_detail() {
        // Narrow width forces 1 col; two cards stack vertically.
        // sel = 0 → click the first (top) card.
        let s = mk_state_with(&["alpha", "bravo"], 0);
        let area = Rect::new(0, 0, 60, 20); // width 60 → cols=1, rows=2
        let action = hit_test(area, 10, 2, &s);
        assert_eq!(action, Some(KeyAction::OpenDetail));
    }

    #[test]
    fn hit_test_on_other_card_returns_move_with_delta() {
        // Two cards stacked vertically; sel=0 → clicking bottom card moves +1.
        let s = mk_state_with(&["alpha", "bravo"], 0);
        let area = Rect::new(0, 0, 60, 20); // cols=1, rows=2 → each row ~10 tall
        let action = hit_test(area, 10, 15, &s);
        assert_eq!(action, Some(KeyAction::Move(1)));
    }

    #[test]
    fn hit_test_returns_negative_delta_when_clicking_earlier_card() {
        // sel=1 (bravo) → clicking alpha at top yields delta -1.
        let s = mk_state_with(&["alpha", "bravo"], 1);
        let area = Rect::new(0, 0, 60, 20);
        let action = hit_test(area, 10, 2, &s);
        assert_eq!(action, Some(KeyAction::Move(-1)));
    }

    #[test]
    fn hit_test_grid_layout_2_cols_picks_correct_card() {
        // width 120 → cols=2; 4 instances → rows=2.
        let s = mk_state_with(&["a", "b", "c", "d"], 0);
        let area = Rect::new(0, 0, 120, 20);
        // Top-right card is index 1 → delta +1 from sel=0.
        let action = hit_test(area, 90, 2, &s);
        assert_eq!(action, Some(KeyAction::Move(1)));
        // Bottom-left card is index 2 → delta +2.
        let action = hit_test(area, 10, 15, &s);
        assert_eq!(action, Some(KeyAction::Move(2)));
        // Bottom-right card is index 3 → delta +3.
        let action = hit_test(area, 90, 15, &s);
        assert_eq!(action, Some(KeyAction::Move(3)));
    }

    fn mk_inst_kv(id: &str, name: &str, kv: Option<f32>) -> Instance {
        Instance {
            container_id: id.into(),
            container_name: name.into(),
            kv_cache_usage_pct: kv,
            ..Default::default()
        }
    }

    fn mk_snap(insts: Vec<Instance>) -> rocm_dash_core::metrics::Snapshot {
        rocm_dash_core::metrics::Snapshot {
            instances: insts,
            ..Default::default()
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn heatmap_rows_track_kv_cache_per_snapshot() {
        let i_a = mk_inst_kv("a", "alpha", Some(10.0));
        let i_b = mk_inst_kv("b", "beta", Some(20.0));
        let history: std::collections::VecDeque<_> = vec![
            mk_snap(vec![
                mk_inst_kv("a", "alpha", Some(5.0)),
                mk_inst_kv("b", "beta", Some(15.0)),
            ]),
            mk_snap(vec![
                mk_inst_kv("a", "alpha", Some(50.0)),
                mk_inst_kv("b", "beta", Some(80.0)),
            ]),
        ]
        .into_iter()
        .collect();
        let live = vec![&i_a, &i_b];
        let rows = build_kv_heatmap_rows(&history, &live);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].label, "alpha");
        assert_eq!(rows[0].data, vec![5.0, 50.0]);
        assert_eq!(rows[1].label, "beta");
        assert_eq!(rows[1].data, vec![15.0, 80.0]);
        // Max is fixed at 100 — kv-cache is already a percentage.
        assert_eq!(rows[0].max, 100.0);
    }

    #[test]
    fn heatmap_row_missing_instance_in_snap_yields_zero() {
        // Instance b only existed in the second snapshot.
        let i_a = mk_inst_kv("a", "alpha", Some(0.0));
        let i_b = mk_inst_kv("b", "beta", Some(0.0));
        let history: std::collections::VecDeque<_> = vec![
            mk_snap(vec![mk_inst_kv("a", "alpha", Some(10.0))]),
            mk_snap(vec![
                mk_inst_kv("a", "alpha", Some(20.0)),
                mk_inst_kv("b", "beta", Some(30.0)),
            ]),
        ]
        .into_iter()
        .collect();
        let live = vec![&i_a, &i_b];
        let rows = build_kv_heatmap_rows(&history, &live);
        assert_eq!(rows[0].data, vec![10.0, 20.0]);
        // b is missing from snap[0] → 0.0 padding.
        assert_eq!(rows[1].data, vec![0.0, 30.0]);
    }

    #[test]
    fn heatmap_row_none_kv_cache_yields_zero() {
        let i_a = mk_inst_kv("a", "alpha", None);
        let history: std::collections::VecDeque<_> =
            vec![mk_snap(vec![mk_inst_kv("a", "alpha", None)])]
                .into_iter()
                .collect();
        let live = vec![&i_a];
        let rows = build_kv_heatmap_rows(&history, &live);
        assert_eq!(rows[0].data, vec![0.0]);
    }

    #[test]
    fn label_width_caps_to_useful_range() {
        let short_id = mk_inst_kv("a", "x", None);
        let long_id = mk_inst_kv("b", "this-is-a-very-long-container-name", None);
        assert_eq!(label_width_for(&[&short_id]), 8);
        assert_eq!(label_width_for(&[&long_id]), 20);
    }

    #[test]
    fn heatmap_height_caps_to_half_panel() {
        assert_eq!(compute_heatmap_height(4, 30), 7); // 4+3
        assert_eq!(compute_heatmap_height(20, 16), 8); // capped at half
        assert_eq!(compute_heatmap_height(1, 30), 5); // floor at 5
    }

    /// Flatten a rendered TestBackend buffer into one newline-joined string so
    /// substring assertions can confirm what reached the screen.
    fn buffer_text(term: &ratatui::Terminal<ratatui::backend::TestBackend>) -> String {
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
    fn render_shows_quantization_and_vram_for_populated_instance() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        // Populate the previously-dead fields: quantization + per-instance VRAM.
        let mut inst = mk_inst("vllm");
        inst.quantization = Some("fp8".into());
        inst.vram_used_mb = 49152;
        inst.vram_total_mb = 196_608; // mib_pair → "48.0 / 192.0 GiB"
        let vram = format::mib_pair(inst.vram_used_mb, inst.vram_total_mb);
        assert_eq!(vram, "48.0 / 192.0 GiB"); // sanity on the expected string

        let mut m = HashMap::new();
        m.insert(inst.container_id.clone(), inst);
        let state = mk_state(m, 0);

        // Card grid: the VRAM line fires only because vram_total_mb > 0.
        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| draw(f, f.area(), &state, &state.theme))
            .unwrap();
        let grid = buffer_text(&term);
        assert!(
            grid.contains(&vram),
            "card grid must render the used / total MiB VRAM string; got:\n{grid}"
        );

        // Detail modal: shows the quantization value and the VRAM pair.
        let mut term = Terminal::new(TestBackend::new(160, 48)).unwrap();
        term.draw(|f| draw_detail(f, f.area(), &state, &state.theme))
            .unwrap();
        let detail = buffer_text(&term);
        assert!(
            detail.contains("fp8"),
            "detail modal must render the quantization value; got:\n{detail}"
        );
        assert!(
            detail.contains(&vram),
            "detail modal must render the used / total MiB VRAM string; got:\n{detail}"
        );
    }
}
