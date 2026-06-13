//! Serve wizard overlay (Phase 3 Wave 1).
//!
//! The headline operational screen rebuilt on the Wave-0 primitives: a compact
//! form that builds a `rocm serve … --managed` invocation and runs it **through
//! the approval gate and the job-bridge** — never inline, never with a legacy
//! `std::thread::spawn` + `try_recv`. A served model launched here surfaces in
//! the services manager and the dashboard's live `gen_tps` (the D7 wire is
//! already in place via `rocm serve --managed`).
//!
//! The model field can be typed directly (a recipe name, alias, or path) or
//! filled from the reusable Wave-0 [`FolderBrowser`] for a local model path
//! (`Tab` on the Model field). The approval *decision* is the user's, captured
//! by the render+event seam; the CLI owns the actual launch — the read-only
//! chat invariant is untouched.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use rocm_dash_core::state::{SideEffect, State, StateEvent};

use crate::ui::approval::{
    ApprovalChoice, ApprovalRequest, ApprovalVerdict, approval_key, draw_approval,
};
use crate::ui::exec::{exe_label, resolve_exe};
use crate::ui::folder_browser::{FolderBrowser, FolderOutcome, draw_folder_browser};
use crate::ui::job_console::draw_job_console;
use crate::ui::modal::{centered_rect, draw_popup_frame};
use crate::ui::model_picker::{ModelPicker, ModelRecipeSummary, PickerOutcome, draw_model_picker};
use crate::ui::theme::Theme;

/// Engine inventory — names mirror `apps/rocm` `engine_inventory()`. Kept
/// TUI-local (a stable, small list) so this layer needs no `rocm-core` dep.
pub const ENGINES: &[&str] = &["lemonade", "pytorch", "llama.cpp", "vllm", "sglang", "atom"];

/// Device-policy choices. Index 0 omits `--device` entirely (engine default);
/// the rest mirror `rocm-core`'s validated `gpu_required|gpu_preferred|cpu_only`.
pub const DEVICES: &[&str] = &[
    "(engine default)",
    "gpu_required",
    "gpu_preferred",
    "cpu_only",
];

/// Mirrors `rocm-core::DEFAULT_LOCAL_HOST` / `DEFAULT_LOCAL_PORT` (TUI-local to
/// avoid the dep; the CLI re-applies its own defaults if these are cleared).
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: &str = "11435";

/// The form fields, in vertical order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Model,
    Engine,
    Device,
    Host,
    Port,
    Mode,
    Launch,
}

/// Field order; `state.field` indexes this.
pub const FIELDS: &[Field] = &[
    Field::Model,
    Field::Engine,
    Field::Device,
    Field::Host,
    Field::Port,
    Field::Mode,
    Field::Launch,
];

/// An approved-but-not-yet-launched serve invocation.
#[derive(Debug, Clone)]
pub struct PendingServe {
    /// Resolved `rocm` binary path (captured at approval time so a later
    /// `current_exe()` failure can't silently drop an approved launch).
    pub cmd: String,
    /// The argv after the binary (`["serve", model, "--engine", …]`).
    pub args: Vec<String>,
    pub request: ApprovalRequest,
    pub choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the wizard is closed.
#[derive(Debug, Clone)]
pub struct ServeWizardState {
    pub field: usize,
    pub model: String,
    pub engine_idx: usize,
    pub device_idx: usize,
    pub host: String,
    pub port: String,
    pub managed: bool,
    /// Local-path picker (Wave-0 primitive); `Some` while browsing.
    pub browser: Option<FolderBrowser>,
    /// Model-recipe picker sub-step; `Some` while choosing a recipe.
    pub picker: Option<ModelPicker>,
    /// Approval modal; `Some` while a launch is gated.
    pub approval: Option<PendingServe>,
    /// In-flight (or just-finished) launch job id.
    pub active_job: Option<String>,
    /// Transient validation message (e.g. empty model / bad port).
    pub message: Option<String>,
}

impl Default for ServeWizardState {
    fn default() -> Self {
        Self {
            field: 0,
            model: String::new(),
            engine_idx: 0,
            device_idx: 0,
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT.to_string(),
            managed: true,
            browser: None,
            picker: None,
            approval: None,
            active_job: None,
            message: None,
        }
    }
}

impl ServeWizardState {
    fn current_field(&self) -> Field {
        FIELDS[self.field.min(FIELDS.len() - 1)]
    }

    fn move_field(&mut self, delta: isize) {
        let max = FIELDS.len() as isize - 1;
        self.field = (self.field as isize + delta).clamp(0, max) as usize;
    }

    fn cycle(&mut self, delta: isize) {
        match self.current_field() {
            Field::Engine => self.engine_idx = cycle_idx(self.engine_idx, ENGINES.len(), delta),
            Field::Device => self.device_idx = cycle_idx(self.device_idx, DEVICES.len(), delta),
            Field::Mode => self.managed = !self.managed,
            _ => {}
        }
    }

    fn type_char(&mut self, c: char) {
        match self.current_field() {
            Field::Model => self.model.push(c),
            Field::Host => self.host.push(c),
            // Port accepts digits only — never builds an unparseable `--port`.
            Field::Port if c.is_ascii_digit() => self.port.push(c),
            _ => {}
        }
    }

    fn backspace(&mut self) {
        match self.current_field() {
            Field::Model => {
                self.model.pop();
            }
            Field::Host => {
                self.host.pop();
            }
            Field::Port => {
                self.port.pop();
            }
            _ => {}
        }
    }

    /// Build the `rocm` argv for the current form, or an error message.
    fn build_args(&self) -> Result<Vec<String>, String> {
        let model = self.model.trim();
        if model.is_empty() {
            return Err("model is required".to_string());
        }
        let mut args = vec!["serve".to_string(), model.to_string()];
        args.push("--engine".to_string());
        args.push(ENGINES[self.engine_idx.min(ENGINES.len() - 1)].to_string());
        // Index 0 = engine default → omit --device.
        if self.device_idx > 0 {
            args.push("--device".to_string());
            args.push(DEVICES[self.device_idx.min(DEVICES.len() - 1)].to_string());
        }
        let host = self.host.trim();
        if !host.is_empty() {
            args.push("--host".to_string());
            args.push(host.to_string());
        }
        let port = self.port.trim();
        if !port.is_empty() {
            // u16 accepts 0, but 0 is not a bindable listen port — reject it so
            // the error surfaces in the form, not as a downstream bind failure.
            match port.parse::<u16>() {
                Ok(p) if p > 0 => {
                    args.push("--port".to_string());
                    args.push(p.to_string());
                }
                _ => return Err(format!("port `{port}` is not a valid 1–65535 value")),
            }
        }
        // Managed (default) hands supervision to the daemon → it shows up in the
        // services manager + dashboard gen_tps. Foreground runs in this job.
        if self.managed {
            args.push("--managed".to_string());
        } else {
            args.push("--foreground".to_string());
        }
        Ok(args)
    }
}

fn cycle_idx(cur: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        return 0;
    }
    let n = len as isize;
    (((cur as isize + delta) % n + n) % n) as usize
}

/// Handle a key while the wizard is open. Mirrors the services-manager seam:
/// mutates the overlay + job model in place and returns reducer side effects
/// (e.g. `SpawnJob`) for the event loop to drive through the job-bridge.
pub fn on_key(
    wizard: &mut Option<ServeWizardState>,
    jobs: &mut State,
    recipes: &[ModelRecipeSummary],
    key: KeyEvent,
) -> Vec<SideEffect> {
    let Some(w) = wizard.as_mut() else {
        return Vec::new();
    };

    // 1) Model-recipe picker sub-step has focus.
    if let Some(picker) = w.picker.as_mut() {
        match picker.on_key(key.code, recipes) {
            PickerOutcome::Chosen(summary) => {
                w.model = summary.id;
                // Pre-select the recipe's preferred engine when it is one this
                // wizard lists; otherwise leave the engine choice untouched.
                if let Some(eng) = summary.preferred_engine
                    && let Some(idx) = ENGINES.iter().position(|e| *e == eng)
                {
                    w.engine_idx = idx;
                }
                w.picker = None;
            }
            PickerOutcome::Cancelled => w.picker = None,
            PickerOutcome::None => {}
        }
        return Vec::new();
    }

    // 2) Folder browser (local model path) has focus.
    if let Some(fb) = w.browser.as_mut() {
        match fb.on_key(key.code) {
            FolderOutcome::Chosen(path) => {
                w.model = path.to_string_lossy().into_owned();
                w.browser = None;
            }
            FolderOutcome::Cancelled => w.browser = None,
            FolderOutcome::None | FolderOutcome::Navigated => {}
        }
        return Vec::new();
    }

    // 2) Approval modal has focus.
    if let Some(pending) = w.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.choice);
        pending.choice = choice;
        match verdict {
            Some(ApprovalVerdict::Approve) => {
                if let Some(pending) = w.approval.take() {
                    return spawn_serve(w, jobs, pending);
                }
            }
            Some(ApprovalVerdict::Deny) | Some(ApprovalVerdict::Cancel) => w.approval = None,
            None => {}
        }
        return Vec::new();
    }

    // 3) A launch job is showing in the console.
    if let Some(job_id) = w.active_job.clone() {
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return jobs.apply(StateEvent::CancelJob(job_id));
            }
            // `q` always closes the whole overlay so the user is never trapped
            // while a job runs (it keeps running in the background).
            KeyCode::Char('q') => *wizard = None,
            KeyCode::Esc | KeyCode::Enter
                if jobs.job(&job_id).map(|j| j.is_terminal()).unwrap_or(true) =>
            {
                w.active_job = None;
            }
            _ => {}
        }
        return Vec::new();
    }

    // 4) Form editing.
    match key.code {
        KeyCode::Esc => *wizard = None,
        KeyCode::Up => w.move_field(-1),
        KeyCode::Down => w.move_field(1),
        KeyCode::Left => w.cycle(-1),
        KeyCode::Right => w.cycle(1),
        KeyCode::Char(' ') if w.current_field() == Field::Mode => w.cycle(1),
        // Tab on the Model field opens the local-path picker (Wave-0 primitive).
        KeyCode::Tab if w.current_field() == Field::Model => {
            let start = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
            w.browser = Some(FolderBrowser::new("Pick a local model path", start));
        }
        KeyCode::Enter => {
            if w.current_field() == Field::Launch {
                request_launch(w);
            } else if w.current_field() == Field::Model && !recipes.is_empty() {
                // On the Model field, Enter opens the recipe picker (the
                // model_picker sub-step); free-text typing + Tab-browse remain.
                // Seed the filter with anything already typed so the picker
                // opens pre-narrowed (e.g. typed "qwen" → Qwen recipes).
                w.picker = Some(ModelPicker {
                    query: w.model.trim().to_string(),
                    selected: 0,
                });
            } else {
                w.move_field(1);
            }
        }
        KeyCode::Backspace => w.backspace(),
        KeyCode::Char(c) => w.type_char(c),
        _ => {}
    }
    Vec::new()
}

/// Validate the form and stage an approval (no job runs until approved).
fn request_launch(w: &mut ServeWizardState) {
    match w.build_args() {
        Ok(args) => {
            let cmd = resolve_exe();
            let cmdline = format!("{} {}", exe_label(&cmd), args.join(" "));
            let request = ApprovalRequest::new(
                format!("serve “{}”", w.model.trim()),
                vec![
                    cmdline,
                    String::new(),
                    "This launches a local model server through the ROCm CLI.".to_string(),
                    if w.managed {
                        "Managed: it will appear in the services manager and dashboard.".to_string()
                    } else {
                        "Foreground: it runs in this job console until stopped.".to_string()
                    },
                ],
            );
            w.message = None;
            w.approval = Some(PendingServe {
                cmd,
                args,
                request,
                choice: ApprovalChoice::default(),
            });
        }
        Err(msg) => w.message = Some(msg),
    }
}

/// Launch the approved serve invocation as a background job.
fn spawn_serve(
    w: &mut ServeWizardState,
    jobs: &mut State,
    pending: PendingServe,
) -> Vec<SideEffect> {
    // A stable id keyed by the model so re-launches replace the prior console.
    let model_key: String = w
        .model
        .trim()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let id = format!("serve-{model_key}");
    let fx = jobs.apply(StateEvent::StartJob {
        id: id.clone(),
        cmd: pending.cmd,
        args: pending.args,
    });
    // The reducer is idempotent: a `StartJob` for an id that is already running
    // (not terminal) no-ops and returns no effects. If that happens, do NOT
    // point `active_job` at the stale job and claim success — surface it and
    // leave the form so the user can wait, cancel, or rename.
    if fx.is_empty() {
        w.message = Some(format!("a job for “{}” is already running", w.model.trim()));
        return fx;
    }
    w.active_job = Some(id);
    fx
}

/// Render the overlay (form, or the folder browser, or the approval modal, or
/// the job console — in priority order).
pub fn draw_serve_wizard(
    f: &mut Frame,
    area: Rect,
    w: &ServeWizardState,
    jobs: &State,
    recipes: &[ModelRecipeSummary],
    theme: &Theme,
) {
    // The job console takes over while a launch is in flight / finished.
    if let Some(job_id) = &w.active_job
        && let Some(job) = jobs.job(job_id)
    {
        draw_job_console(f, area, job, 0, theme);
        return;
    }

    let popup = centered_rect(72, 78, 96, 26, area);
    let inner = draw_popup_frame(f, popup, "Serve a model", theme);
    if inner.height == 0 {
        return;
    }

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

    let has_recipes = !recipes.is_empty();
    let lines: Vec<Line> = FIELDS
        .iter()
        .enumerate()
        .map(|(i, field)| field_line(*field, i == w.field, w, has_recipes, theme))
        .collect();
    f.render_widget(Paragraph::new(lines), rows[0]);

    let msg = w.message.as_deref().unwrap_or("");
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            msg.to_string(),
            Style::default().fg(theme.err),
        ))),
        rows[1],
    );

    let hint = if recipes.is_empty() {
        "↑↓ field · ←→ change · Tab browse (model) · Enter next/launch · Esc close"
    } else {
        "↑↓ field · ←→ change · Enter pick (model)/next/launch · Tab browse · Esc close"
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            hint,
            Style::default().fg(theme.muted),
        ))),
        rows[2],
    );

    // Picker / folder browser / approval sit on top of the form when active.
    if let Some(picker) = &w.picker {
        draw_model_picker(f, area, picker, recipes, theme);
    }
    if let Some(fb) = &w.browser {
        draw_folder_browser(f, area, fb, theme);
    }
    if let Some(pending) = &w.approval {
        draw_approval(f, area, &pending.request, pending.choice, theme);
    }
}

fn field_line<'a>(
    field: Field,
    selected: bool,
    w: &'a ServeWizardState,
    has_recipes: bool,
    theme: &Theme,
) -> Line<'a> {
    let model_placeholder = if has_recipes {
        "(Enter to pick a recipe · type a name · Tab to browse)"
    } else {
        "(type a name / path, or Tab to browse)"
    };
    let (label, value): (&str, String) = match field {
        Field::Model => ("Model", display_value(&w.model, model_placeholder)),
        Field::Engine => (
            "Engine",
            ENGINES[w.engine_idx.min(ENGINES.len() - 1)].to_string(),
        ),
        Field::Device => (
            "Device",
            DEVICES[w.device_idx.min(DEVICES.len() - 1)].to_string(),
        ),
        Field::Host => ("Host", display_value(&w.host, "(engine default)")),
        Field::Port => ("Port", display_value(&w.port, "(engine default)")),
        Field::Mode => (
            "Mode",
            if w.managed {
                "managed".to_string()
            } else {
                "foreground".to_string()
            },
        ),
        Field::Launch => ("", String::new()),
    };

    if field == Field::Launch {
        let style = if selected {
            Style::default()
                .bg(theme.ok)
                .fg(theme.bg)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.ok)
        };
        return Line::from(Span::styled("  [ Launch ]  ", style));
    }

    let marker = if selected { "▶ " } else { "  " };
    let label_style = if selected {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.muted)
    };
    let value_style = if selected {
        Style::default().fg(theme.fg).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.fg)
    };
    Line::from(vec![
        Span::styled(marker, label_style),
        Span::styled(format!("{label:<8}"), label_style),
        Span::styled(value, value_style),
    ])
}

fn display_value(v: &str, placeholder: &'static str) -> String {
    if v.is_empty() {
        placeholder.to_string()
    } else {
        v.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    fn key(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    fn typed(s: &str) -> Vec<KeyEvent> {
        s.chars().map(|c| key(KeyCode::Char(c))).collect()
    }

    #[test]
    fn cycle_idx_wraps_both_directions() {
        assert_eq!(cycle_idx(0, 3, -1), 2);
        assert_eq!(cycle_idx(2, 3, 1), 0);
        assert_eq!(cycle_idx(0, 0, 1), 0);
    }

    #[test]
    fn default_form_targets_managed_lemonade() {
        let w = ServeWizardState::default();
        assert!(w.managed);
        assert_eq!(ENGINES[w.engine_idx], "lemonade");
        assert_eq!(w.device_idx, 0); // engine default → no --device
        assert_eq!(w.host, "127.0.0.1");
        assert_eq!(w.port, "11435");
    }

    #[test]
    fn build_args_requires_a_model() {
        let w = ServeWizardState::default();
        assert_eq!(w.build_args().unwrap_err(), "model is required");
    }

    #[test]
    fn build_args_emits_managed_serve_with_defaults() {
        let w = ServeWizardState {
            model: "qwen".into(),
            ..Default::default()
        };
        let args = w.build_args().unwrap();
        assert_eq!(
            args,
            vec![
                "serve",
                "qwen",
                "--engine",
                "lemonade",
                "--host",
                "127.0.0.1",
                "--port",
                "11435",
                "--managed",
            ]
        );
    }

    #[test]
    fn build_args_includes_device_when_not_default_and_foreground() {
        let w = ServeWizardState {
            model: "glm".into(),
            device_idx: 1, // gpu_required
            managed: false,
            ..Default::default()
        };
        let args = w.build_args().unwrap();
        assert!(args.windows(2).any(|p| p == ["--device", "gpu_required"]));
        assert!(args.contains(&"--foreground".to_string()));
        assert!(!args.contains(&"--managed".to_string()));
    }

    #[test]
    fn build_args_rejects_bad_port() {
        let w = ServeWizardState {
            model: "m".into(),
            port: "99999".into(),
            ..Default::default()
        };
        assert!(w.build_args().unwrap_err().contains("port"));
    }

    #[test]
    fn build_args_rejects_port_zero() {
        // u16 parses 0, but it is not a bindable listen port — must be rejected
        // in the form, matching the "1–65535" message.
        let w = ServeWizardState {
            model: "m".into(),
            port: "0".into(),
            ..Default::default()
        };
        assert!(w.build_args().unwrap_err().contains("port"));
    }

    #[test]
    fn port_field_accepts_digits_only() {
        let mut w = ServeWizardState::default();
        w.port.clear();
        w.field = FIELDS.iter().position(|f| *f == Field::Port).unwrap();
        for k in typed("80a0") {
            // route through type_char via the field
            if let KeyCode::Char(c) = k.code {
                w.type_char(c);
            }
        }
        assert_eq!(w.port, "800");
    }

    #[test]
    fn launch_requires_approval_then_spawns_job() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        // Fill the model by typing on the Model field (field 0 by default).
        for k in typed("qwen") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        assert_eq!(wiz.as_ref().unwrap().model, "qwen");
        // Jump to Launch and press Enter → approval staged, NO job yet.
        let launch_idx = FIELDS.iter().position(|f| *f == Field::Launch).unwrap();
        wiz.as_mut().unwrap().field = launch_idx;
        let fx = on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        assert!(fx.is_empty(), "launch must not run before approval");
        assert!(wiz.as_ref().unwrap().approval.is_some());
        assert!(jobs.jobs.is_empty());

        // Approve → exactly one SpawnJob, job registered, console active.
        let fx = on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('y')));
        assert_eq!(fx.len(), 1);
        assert!(matches!(fx[0], SideEffect::SpawnJob { .. }));
        let w = wiz.as_ref().unwrap();
        assert!(w.approval.is_none());
        assert_eq!(w.active_job.as_deref(), Some("serve-qwen"));
        assert_eq!(jobs.jobs.len(), 1);
    }

    #[test]
    fn empty_model_launch_sets_message_not_approval() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        let launch_idx = FIELDS.iter().position(|f| *f == Field::Launch).unwrap();
        wiz.as_mut().unwrap().field = launch_idx;
        let fx = on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        assert!(fx.is_empty());
        let w = wiz.as_ref().unwrap();
        assert!(w.approval.is_none());
        assert_eq!(w.message.as_deref(), Some("model is required"));
    }

    #[test]
    fn deny_cancels_without_spawning() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        wiz.as_mut().unwrap().model = "m".into();
        wiz.as_mut().unwrap().field = FIELDS.iter().position(|f| *f == Field::Launch).unwrap();
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        let fx = on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('n')));
        assert!(fx.is_empty());
        assert!(wiz.as_ref().unwrap().approval.is_none());
        assert!(jobs.jobs.is_empty());
    }

    #[test]
    fn esc_closes_when_idle() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Esc));
        assert!(wiz.is_none());
    }

    #[test]
    fn q_escapes_overlay_while_job_runs() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        wiz.as_mut().unwrap().model = "m".into();
        wiz.as_mut().unwrap().field = FIELDS.iter().position(|f| *f == Field::Launch).unwrap();
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('y')));
        assert!(wiz.as_ref().unwrap().active_job.is_some());
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('q')));
        assert!(wiz.is_none(), "q must close the overlay even mid-job");
    }

    #[test]
    fn esc_is_ignored_while_job_is_running_then_dismisses_when_terminal() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        wiz.as_mut().unwrap().model = "m".into();
        wiz.as_mut().unwrap().field = FIELDS.iter().position(|f| *f == Field::Launch).unwrap();
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('y')));
        let job_id = wiz.as_ref().unwrap().active_job.clone().unwrap();
        // Job is Running: Esc must NOT dismiss the console or close the overlay.
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Esc));
        assert_eq!(wiz.as_ref().unwrap().active_job.as_ref(), Some(&job_id));
        assert!(wiz.is_some());
        // Once terminal, Esc dismisses the console back to the form.
        jobs.apply(StateEvent::JobDone {
            id: job_id,
            code: 0,
        });
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Esc));
        assert!(wiz.as_ref().unwrap().active_job.is_none());
        assert!(
            wiz.is_some(),
            "dismissing the console keeps the wizard open"
        );
    }

    #[test]
    fn relaunch_while_prior_job_running_surfaces_message_not_stale_console() {
        // The reducer no-ops a StartJob for a still-running id. spawn_serve must
        // NOT claim success (set active_job) when no SpawnJob was emitted.
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        let launch = FIELDS.iter().position(|f| *f == Field::Launch).unwrap();

        // First launch of "qwen": real spawn.
        for k in typed("qwen") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        wiz.as_mut().unwrap().field = launch;
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('y')));
        assert_eq!(
            wiz.as_ref().unwrap().active_job.as_deref(),
            Some("serve-qwen")
        );

        // Simulate the user closing + reopening the overlay (fresh state) while
        // the "serve-qwen" job is still Running in the shared job model.
        let mut wiz2 = Some(ServeWizardState::default());
        for k in typed("qwen") {
            on_key(&mut wiz2, &mut jobs, &[], k);
        }
        wiz2.as_mut().unwrap().field = launch;
        on_key(&mut wiz2, &mut jobs, &[], key(KeyCode::Enter));
        let fx = on_key(&mut wiz2, &mut jobs, &[], key(KeyCode::Char('y')));
        // No new SpawnJob, no stale console, an informative message instead.
        assert!(fx.is_empty(), "no double-spawn for a running id");
        let w = wiz2.as_ref().unwrap();
        assert!(w.active_job.is_none(), "must not point at the stale job");
        assert!(
            w.message
                .as_deref()
                .unwrap_or("")
                .contains("already running")
        );
        assert_eq!(jobs.jobs.len(), 1, "still just the one job");
    }

    #[test]
    fn tab_on_model_opens_folder_browser() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Tab));
        assert!(wiz.as_ref().unwrap().browser.is_some());
        // Esc inside the browser closes the browser, not the overlay.
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Esc));
        assert!(wiz.as_ref().unwrap().browser.is_none());
        assert!(wiz.is_some());
    }

    #[test]
    fn left_right_cycles_engine() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        wiz.as_mut().unwrap().field = FIELDS.iter().position(|f| *f == Field::Engine).unwrap();
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Right));
        assert_eq!(wiz.as_ref().unwrap().engine_idx, 1);
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Left));
        assert_eq!(wiz.as_ref().unwrap().engine_idx, 0);
    }

    #[test]
    fn enter_on_model_opens_picker_and_choice_fills_model_and_engine() {
        let recipes = vec![ModelRecipeSummary {
            id: "GLM-4".into(),
            aliases: vec!["glm".into()],
            task: "chat".into(),
            preferred_engine: Some("vllm".into()),
        }];
        let mut wiz = Some(ServeWizardState::default()); // field 0 = Model
        let mut jobs = State::default();
        // Enter on Model opens the picker when recipes exist.
        on_key(&mut wiz, &mut jobs, &recipes, key(KeyCode::Enter));
        assert!(wiz.as_ref().unwrap().picker.is_some());
        // Enter in the picker chooses the (only) recipe → fills model + engine.
        on_key(&mut wiz, &mut jobs, &recipes, key(KeyCode::Enter));
        let w = wiz.as_ref().unwrap();
        assert!(w.picker.is_none());
        assert_eq!(w.model, "GLM-4");
        assert_eq!(
            ENGINES[w.engine_idx], "vllm",
            "preferred engine pre-selected"
        );
    }

    #[test]
    fn enter_on_model_advances_when_no_recipes() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        assert!(wiz.as_ref().unwrap().picker.is_none());
        assert_eq!(wiz.as_ref().unwrap().field, 1, "Enter advances to Engine");
    }

    #[test]
    fn recipe_with_unknown_preferred_engine_leaves_engine_idx() {
        // Pins the silent-fallback contract: a preferred_engine the wizard does
        // not list must NOT crash and must leave the engine choice untouched
        // (model still filled). Guards future ENGINES vs rocm-core divergence.
        let recipes = vec![ModelRecipeSummary {
            id: "some-model".into(),
            aliases: vec![],
            task: "chat".into(),
            preferred_engine: Some("not-in-engines-list".into()),
        }];
        let mut wiz = Some(ServeWizardState::default()); // engine_idx 0 = lemonade
        let mut jobs = State::default();
        on_key(&mut wiz, &mut jobs, &recipes, key(KeyCode::Enter)); // open picker
        on_key(&mut wiz, &mut jobs, &recipes, key(KeyCode::Enter)); // choose first
        let w = wiz.as_ref().unwrap();
        assert_eq!(w.model, "some-model");
        assert_eq!(
            w.engine_idx, 0,
            "unknown preferred engine leaves the choice"
        );
    }

    #[test]
    fn picker_opens_pre_filtered_by_typed_model_text() {
        let recipes = vec![
            ModelRecipeSummary {
                id: "Qwen3-4B".into(),
                aliases: vec!["qwen".into()],
                task: "chat".into(),
                preferred_engine: None,
            },
            ModelRecipeSummary {
                id: "GLM-4".into(),
                aliases: vec!["glm".into()],
                task: "chat".into(),
                preferred_engine: None,
            },
        ];
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        // Type "qwen" on the Model field, then Enter to open the picker.
        for k in typed("qwen") {
            on_key(&mut wiz, &mut jobs, &recipes, k);
        }
        on_key(&mut wiz, &mut jobs, &recipes, key(KeyCode::Enter));
        let picker = wiz.as_ref().unwrap().picker.as_ref().unwrap();
        assert_eq!(picker.query, "qwen");
        assert_eq!(picker.filtered(&recipes).len(), 1, "pre-narrowed to Qwen");
        // Enter chooses the single match.
        on_key(&mut wiz, &mut jobs, &recipes, key(KeyCode::Enter));
        assert_eq!(wiz.as_ref().unwrap().model, "Qwen3-4B");
    }

    #[test]
    fn picker_esc_returns_to_form_without_changing_model() {
        let recipes = vec![ModelRecipeSummary {
            id: "GLM-4".into(),
            aliases: vec![],
            task: "chat".into(),
            preferred_engine: None,
        }];
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        on_key(&mut wiz, &mut jobs, &recipes, key(KeyCode::Enter));
        on_key(&mut wiz, &mut jobs, &recipes, key(KeyCode::Esc));
        let w = wiz.as_ref().unwrap();
        assert!(w.picker.is_none());
        assert!(w.model.is_empty());
        assert!(wiz.is_some(), "picker Esc keeps the wizard open");
    }

    fn render(w: &ServeWizardState, jobs: &State) -> String {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(120, 30);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw_serve_wizard(f, f.area(), w, jobs, &[], &theme))
            .unwrap();
        let buf = term.backend().buffer().clone();
        buf.content().iter().map(|c| c.symbol()).collect()
    }

    #[test]
    fn snapshot_renders_form_fields() {
        let w = ServeWizardState::default();
        let out = render(&w, &State::default());
        assert!(out.contains("Serve a model"), "titled overlay");
        assert!(out.contains("Model"), "model field");
        assert!(out.contains("lemonade"), "default engine shown");
        assert!(out.contains("Launch"), "launch action");
    }

    #[test]
    fn snapshot_shows_approval_modal_on_launch() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        wiz.as_mut().unwrap().model = "qwen".into();
        wiz.as_mut().unwrap().field = FIELDS.iter().position(|f| *f == Field::Launch).unwrap();
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        let out = render(wiz.as_ref().unwrap(), &jobs);
        assert!(out.contains("Review:"), "approval modal shown");
        assert!(out.contains("serve"), "describes the gated launch");
        assert!(out.contains("Approve"), "approve button present");
    }
}
