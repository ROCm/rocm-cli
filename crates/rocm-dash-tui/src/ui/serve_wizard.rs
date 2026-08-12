// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Serve wizard overlay (Phase 3 Wave 1).
//!
//! The headline operational screen rebuilt on the Wave-0 primitives: a compact
//! **model-first** form that builds a `rocm serve …` invocation and runs it
//! **through the approval gate and the job-bridge** — never inline, never with
//! a legacy `std::thread::spawn` + `try_recv`. A served model launched here
//! surfaces in the services manager and the dashboard's live `gen_tps`.
//!
//! Progressive disclosure: the default form is Model → `Advanced settings` →
//! Launch. Everything else (engine, device policy, host, port, mode) is
//! automatic and lives behind the inline `Advanced settings` row — expanded in
//! place, never as a second modal. Because the defaults are genuinely
//! automatic, the default invocation is exactly `rocm serve MODEL`: the CLI
//! stays the single resolution authority for engine choice, the GPU-required
//! policy, the loopback host, the port, and managed mode.
//!
//! The model field can be typed directly (a recipe name, alias, or path) or
//! filled from the reusable Wave-0 [`FolderBrowser`] for a local model path
//! (`Tab` on the Model field). The approval *decision* is the user's, captured
//! by the render+event seam; the CLI owns the actual launch — the read-only
//! chat invariant is untouched.

use crossterm::event::{KeyCode, KeyEvent};
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
use crate::ui::job_console::{ConsoleOutcome, on_console_key};
use crate::ui::model_picker::{ModelPicker, ModelRecipeSummary, PickerOutcome, draw_model_picker};
use crate::ui::panel::{self, BoxRole};
use crate::ui::theme::Theme;

/// Engine inventory — index 0 is the automatic choice.
///
/// Automatic emits no `--engine` and lets the CLI resolve one; the rest mirror
/// `apps/rocm` `engine_inventory()`. Kept TUI-local (a stable, small list) so
/// this layer needs no `rocm-core` dep.
pub const ENGINES: &[&str] = &["automatic", "lemonade", "vllm"];

/// Index into [`ENGINES`] that means "let the CLI decide".
pub const ENGINE_AUTO: usize = 0;

/// The device policy this wizard offers: read-only, GPU-required, automatic.
///
/// ROCm never falls back to CPU, so there is nothing here to choose — omitting
/// `--device` already resolves to `gpu_required` in the CLI.
pub const DEVICE_POLICY: &str = "GPU required (automatic)";

/// Mirrors `rocm-core::DEFAULT_LOCAL_HOST` / `DEFAULT_LOCAL_PORT` (TUI-local to
/// avoid the dep). The default host is omitted from argv; the default port text
/// is only the starting point for a *Custom* port.
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: &str = "11435";

/// The shared phrase for an automatic port.
///
/// The wizard cannot truthfully promise a concrete port before the CLI leases
/// one, so the approval card says exactly this and the launch output prints the
/// resolved endpoint.
pub const AUTO_PORT_NOTE: &str = "Port: automatic; endpoint shown after launch";

/// The exact cross-field guidance for a custom host left on an automatic port.
///
/// Automatic selection is only supported on the canonical loopback host.
pub const CUSTOM_HOST_NEEDS_PORT: &str = "Custom hosts require a custom port; set Port to Custom.";

/// Loopback hosts the CLI accepts without `--allow-public-bind`.
///
/// Dash has no public-bind confirmation or endpoint-key UI, so a public bind
/// must be requested deliberately from the CLI instead.
pub const LOOPBACK_HOSTS: &[&str] = &["127.0.0.1", "localhost", "::1", "[::1]"];

/// The guidance shown when a public / non-loopback host is typed into Dash.
pub const PUBLIC_HOST_NEEDS_CLI: &str = "Dash serves on loopback only; for a public bind run \
     `rocm serve --host … --port … --allow-public-bind` from the CLI.";

/// Whether `host` is one of the loopback spellings Dash may bind.
#[must_use]
pub fn is_loopback_host(host: &str) -> bool {
    LOOPBACK_HOSTS
        .iter()
        .any(|known| known.eq_ignore_ascii_case(host))
}

/// The form fields. Not every field is visible at once — see
/// [`ServeWizardState::visible_fields`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Model,
    /// Inline progressive-disclosure row (`Advanced settings ▸/▾`).
    Advanced,
    Engine,
    Device,
    Host,
    Port,
    Mode,
    Launch,
}

/// Full (expanded) field order.
pub const FIELDS: &[Field] = &[
    Field::Model,
    Field::Advanced,
    Field::Engine,
    Field::Device,
    Field::Host,
    Field::Port,
    Field::Mode,
    Field::Launch,
];

/// Collapsed (default) field order — model-first, then disclosure, then launch.
pub const BASIC_FIELDS: &[Field] = &[Field::Model, Field::Advanced, Field::Launch];

/// Port policy: automatic (the CLI leases a free local port) or an explicit
/// custom port typed by the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PortMode {
    #[default]
    Auto,
    Custom,
}

/// Inline single-line text editor state for the Host / Custom-port rows.
///
/// On entry the existing value is *selected*: the first printable character
/// replaces it wholesale (the common "retype it" case) while Left/Right,
/// Backspace, and further typing fall back to ordinary cursor editing.
#[derive(Debug, Clone)]
pub struct InlineEditor {
    pub field: Field,
    /// Value to restore on Escape.
    pub original: String,
    pub value: String,
    /// Cursor position, in characters, in `0..=value.chars().count()`.
    pub cursor: usize,
    /// Whether the initial value is still fully selected.
    pub selected: bool,
}

impl InlineEditor {
    fn new(field: Field, value: &str) -> Self {
        Self {
            field,
            original: value.to_string(),
            value: value.to_string(),
            cursor: value.chars().count(),
            selected: true,
        }
    }

    fn len(&self) -> usize {
        self.value.chars().count()
    }

    fn byte_at(&self, cursor: usize) -> usize {
        self.value
            .char_indices()
            .nth(cursor)
            .map_or(self.value.len(), |(b, _)| b)
    }

    fn insert(&mut self, c: char) {
        if self.selected {
            self.value.clear();
            self.cursor = 0;
            self.selected = false;
        }
        let at = self.byte_at(self.cursor);
        self.value.insert(at, c);
        self.cursor += 1;
    }

    fn backspace(&mut self) {
        if self.selected {
            self.value.clear();
            self.cursor = 0;
            self.selected = false;
            return;
        }
        if self.cursor == 0 {
            return;
        }
        let from = self.byte_at(self.cursor - 1);
        let to = self.byte_at(self.cursor);
        self.value.replace_range(from..to, "");
        self.cursor -= 1;
    }

    fn move_cursor(&mut self, delta: isize) {
        self.selected = false;
        let max = self.len().cast_signed();
        self.cursor = (self.cursor.cast_signed() + delta).clamp(0, max) as usize;
    }
}

/// An approved-but-not-yet-launched serve invocation.
#[derive(Debug, Clone)]
pub struct PendingServe {
    /// Resolved `rocm` binary path (captured at approval time so a later
    /// `current_exe()` failure can't silently drop an approved launch).
    pub cmd: String,
    /// The argv after the binary (`["serve", model, …]`).
    pub args: Vec<String>,
    pub request: ApprovalRequest,
    pub choice: ApprovalChoice,
}

/// Overlay state. `None` on `AppState` means the wizard is closed.
#[derive(Debug, Clone)]
pub struct ServeWizardState {
    /// Index into [`ServeWizardState::visible_fields`].
    pub field: usize,
    pub model: String,
    pub engine_idx: usize,
    pub host: String,
    pub port_mode: PortMode,
    /// The custom port text (only meaningful when `port_mode` is `Custom`).
    pub port: String,
    pub managed: bool,
    /// Whether the inline `Advanced settings` rows are showing.
    pub advanced_expanded: bool,
    /// Inline text editor; `Some` while editing Host or the custom port.
    pub editor: Option<InlineEditor>,
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
            engine_idx: ENGINE_AUTO,
            host: DEFAULT_HOST.to_string(),
            port_mode: PortMode::Auto,
            port: DEFAULT_PORT.to_string(),
            managed: true,
            advanced_expanded: false,
            editor: None,
            browser: None,
            picker: None,
            approval: None,
            active_job: None,
            message: None,
        }
    }
}

impl ServeWizardState {
    /// The rows the user can actually see and reach right now.
    #[must_use]
    pub const fn visible_fields(&self) -> &'static [Field] {
        if self.advanced_expanded {
            FIELDS
        } else {
            BASIC_FIELDS
        }
    }

    /// The focused row (clamped — collapsing can never strand focus).
    #[must_use]
    pub fn current_field(&self) -> Field {
        let vis = self.visible_fields();
        vis[self.field.min(vis.len() - 1)]
    }

    fn move_field(&mut self, delta: isize) {
        let vis = self.visible_fields();
        let cur = self.field.min(vis.len() - 1);
        let max = vis.len().cast_signed() - 1;
        self.field = (cur.cast_signed() + delta).clamp(0, max) as usize;
    }

    /// Focus the given field, expanding Advanced when it is hidden.
    fn focus(&mut self, field: Field) {
        if !self.visible_fields().contains(&field) {
            self.advanced_expanded = true;
        }
        if let Some(idx) = self.visible_fields().iter().position(|f| *f == field) {
            self.field = idx;
        }
    }

    fn cycle(&mut self, delta: isize) {
        match self.current_field() {
            // Advanced is a choice row too: ← collapses, → expands. Only
            // reachable while Advanced owns focus, and Advanced sits at the
            // same index in both orders, so focus is preserved either way.
            Field::Advanced => self.advanced_expanded = delta > 0,
            Field::Engine => self.engine_idx = cycle_idx(self.engine_idx, ENGINES.len(), delta),
            Field::Port => {
                self.port_mode = match self.port_mode {
                    PortMode::Auto => PortMode::Custom,
                    PortMode::Custom => PortMode::Auto,
                };
            }
            Field::Mode => self.managed = !self.managed,
            _ => {}
        }
    }

    /// Direct typing only applies to the Model row; Host and the custom port
    /// are edited through the explicit inline editor.
    fn type_char(&mut self, c: char) {
        if self.current_field() == Field::Model {
            self.model.push(c);
        }
    }

    fn backspace(&mut self) {
        if self.current_field() == Field::Model {
            self.model.pop();
        }
    }

    /// Whether any advanced setting deviates from the automatic defaults.
    #[must_use]
    pub fn is_customized(&self) -> bool {
        self.engine_idx != ENGINE_AUTO
            || self.host.trim() != DEFAULT_HOST
            || self.port_mode != PortMode::Auto
            || !self.managed
    }

    /// The `Advanced settings` summary text — the one place a *collapsed* form
    /// still tells the truth about overrides, including a broken hidden one.
    #[must_use]
    pub fn advanced_summary(&self) -> &'static str {
        if !self.is_customized() {
            return "Automatic";
        }
        if self.port_mode == PortMode::Custom && parse_port(&self.port).is_none() {
            "Customized · port needs attention"
        } else {
            "Customized"
        }
    }

    /// Validate the whole form. On success returns the concrete port to emit
    /// (`None` for automatic); on failure the offending field plus a plain fix.
    fn validate(&self) -> Result<Option<u16>, (Field, String)> {
        if self.model.trim().is_empty() {
            return Err((Field::Model, "model is required".to_string()));
        }
        let host = self.host.trim();
        if host.is_empty() {
            return Err((
                Field::Host,
                format!("host is required; type an address or restore {DEFAULT_HOST}"),
            ));
        }
        // Automatic port selection is only supported on the canonical loopback
        // host — anything else must name its own port. Never invent one.
        if host != DEFAULT_HOST && self.port_mode == PortMode::Auto {
            return Err((Field::Port, CUSTOM_HOST_NEEDS_PORT.to_string()));
        }
        // Dash has no `--allow-public-bind` confirmation and no endpoint-key
        // UI, so it must never stage a public bind — even with an explicit
        // port. That deliberate step belongs to the CLI.
        if !is_loopback_host(host) {
            return Err((Field::Host, PUBLIC_HOST_NEEDS_CLI.to_string()));
        }
        if self.port_mode == PortMode::Custom {
            let Some(port) = parse_port(&self.port) else {
                let shown = self.port.trim();
                return Err((
                    Field::Port,
                    format!("port `{shown}` is not a valid 1–65535 value"),
                ));
            };
            return Ok(Some(port));
        }
        Ok(None)
    }

    /// Build the `rocm` argv for the current form, or the offending field and
    /// an error message. Only explicit overrides are emitted: the automatic
    /// form is exactly `serve MODEL`.
    fn build_args(&self) -> Result<Vec<String>, (Field, String)> {
        let port = self.validate()?;
        let model = self.model.trim();
        let mut args = vec!["serve".to_string(), model.to_string()];
        // Automatic engine → the CLI resolves it.
        if self.engine_idx != ENGINE_AUTO {
            args.push("--engine".to_string());
            args.push(ENGINES[self.engine_idx.min(ENGINES.len() - 1)].to_string());
        }
        // Device policy is automatic and GPU-required; omitting `--device`
        // already means `gpu_required` in the CLI.
        let host = self.host.trim();
        if host != DEFAULT_HOST {
            args.push("--host".to_string());
            args.push(host.to_string());
        }
        if let Some(port) = port {
            args.push("--port".to_string());
            args.push(port.to_string());
        }
        // Managed (the default) hands supervision to the daemon → it shows up
        // in the services manager + dashboard gen_tps, and needs no flag.
        if !self.managed {
            args.push("--foreground".to_string());
        }
        Ok(args)
    }
}

/// Parse a custom port: digits only, 1–65535. `0` parses as `u16` but is not a
/// bindable listen port, so it is rejected here rather than downstream.
fn parse_port(raw: &str) -> Option<u16> {
    match raw.trim().parse::<u16>() {
        Ok(p) if p > 0 => Some(p),
        _ => None,
    }
}

const fn cycle_idx(cur: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        return 0;
    }
    let n = len.cast_signed();
    (((cur.cast_signed() + delta) % n + n) % n) as usize
}

/// Handle a key while the wizard is open.
///
/// Mirrors the services-manager seam: mutates the overlay + job model in place and returns reducer side effects
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
                // Only the model is filled. A recipe's preferred engine is NOT
                // forced onto the form: the CLI stays the resolution authority
                // and the wizard keeps emitting an automatic engine.
                w.model = summary.id;
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

    // 3) Approval modal has focus.
    if let Some(pending) = w.approval.as_mut() {
        let (choice, verdict) = approval_key(key.code, pending.choice);
        pending.choice = choice;
        match verdict {
            Some(ApprovalVerdict::Approve) => {
                if let Some(pending) = w.approval.take() {
                    return spawn_serve(w, jobs, pending);
                }
            }
            Some(ApprovalVerdict::Deny | ApprovalVerdict::Cancel) => w.approval = None,
            None => {}
        }
        return Vec::new();
    }

    // 4) A launch job is showing in the console.
    if let Some(job_id) = w.active_job.clone() {
        match on_console_key(&job_id, jobs, key) {
            ConsoleOutcome::Cancelled(fx) => return fx,
            ConsoleOutcome::Closed => *wizard = None,
            ConsoleOutcome::Dismissed => w.active_job = None,
            ConsoleOutcome::Unhandled => {}
        }
        return Vec::new();
    }

    // 5) Inline text editor (Host / custom Port) owns the keyboard. It is not a
    //    modal: the form stays painted underneath, and form navigation keys
    //    cannot reach hidden rows from here.
    if w.editor.is_some() {
        editor_key(w, key.code);
        return Vec::new();
    }

    // 6) Form editing.
    match key.code {
        KeyCode::Esc => *wizard = None,
        KeyCode::Up => w.move_field(-1),
        KeyCode::Down => w.move_field(1),
        KeyCode::Left => w.cycle(-1),
        KeyCode::Right => w.cycle(1),
        KeyCode::Char(' ') if w.current_field() == Field::Mode => w.cycle(1),
        KeyCode::Char(' ') if w.current_field() == Field::Advanced => {
            w.advanced_expanded = !w.advanced_expanded;
        }
        // Tab on the Model field opens the local-path picker (Wave-0 primitive).
        KeyCode::Tab if w.current_field() == Field::Model => {
            let start = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
            w.browser = Some(FolderBrowser::new("Pick a local model path", start));
        }
        KeyCode::Enter => match w.current_field() {
            Field::Launch => request_launch(w),
            // Expand/collapse only ever happens while Advanced owns focus, so
            // focus lands back on Advanced in both directions.
            Field::Advanced => w.advanced_expanded = !w.advanced_expanded,
            Field::Host => w.editor = Some(InlineEditor::new(Field::Host, &w.host)),
            Field::Port if w.port_mode == PortMode::Custom => {
                w.editor = Some(InlineEditor::new(Field::Port, &w.port));
            }
            Field::Model if !recipes.is_empty() => {
                // On the Model field, Enter opens the recipe picker (the
                // model_picker sub-step); free-text typing + Tab-browse remain.
                // Seed the filter with anything already typed so the picker
                // opens pre-narrowed (e.g. typed "qwen" → Qwen recipes).
                w.picker = Some(ModelPicker {
                    query: w.model.trim().to_string(),
                    selected: 0,
                });
            }
            _ => w.move_field(1),
        },
        KeyCode::Backspace => w.backspace(),
        KeyCode::Char(c) => w.type_char(c),
        _ => {}
    }
    Vec::new()
}

/// Drive the inline editor. Enter accepts, Escape restores, and nothing here
/// can open another overlay.
fn editor_key(w: &mut ServeWizardState, code: KeyCode) {
    if w.editor.is_none() {
        return;
    }
    match code {
        KeyCode::Esc => {
            // Restore the original — an abandoned edit never mutates the form.
            if let Some(ed) = w.editor.take() {
                match ed.field {
                    Field::Host => w.host = ed.original,
                    Field::Port => w.port = ed.original,
                    _ => {}
                }
            }
            refresh_message(w);
        }
        KeyCode::Left | KeyCode::Right => {
            let delta = if code == KeyCode::Left { -1 } else { 1 };
            if let Some(ed) = w.editor.as_mut() {
                ed.move_cursor(delta);
            }
        }
        KeyCode::Backspace => {
            if let Some(ed) = w.editor.as_mut() {
                ed.backspace();
            }
        }
        KeyCode::Enter => {
            let Some((field, value)) = w.editor.as_ref().map(|e| (e.field, e.value.clone())) else {
                return;
            };
            match field {
                Field::Host => {
                    let trimmed = value.trim();
                    if trimmed.is_empty() {
                        // Keep the editor open with a plain fix rather than
                        // silently accepting an unusable host.
                        w.message = Some(format!(
                            "host cannot be empty; type an address or press Esc to restore {DEFAULT_HOST}"
                        ));
                        return;
                    }
                    w.host = trimmed.to_string();
                }
                Field::Port => {
                    if value.trim().is_empty() {
                        // Mirror the Host rule: an empty custom port is not a
                        // value, so keep the editor open with a plain fix.
                        w.message = Some(
                            "port cannot be empty; type a 1–65535 value or press Esc to cancel"
                                .to_string(),
                        );
                        return;
                    }
                    // Digits-only is enforced on input; range validation happens
                    // at review time so an in-progress value stays visible.
                    w.port = value;
                }
                _ => {}
            }
            refresh_message(w);
            w.editor = None;
        }
        KeyCode::Char(c) => {
            if let Some(ed) = w.editor.as_mut() {
                if ed.field == Field::Port && !c.is_ascii_digit() {
                    return;
                }
                ed.insert(c);
            }
        }
        _ => {}
    }
}

/// Re-evaluate an *outstanding* validation complaint against the current form:
/// a genuine fix clears it, a still-broken form re-states the reason. Silent
/// when nothing has been complained about yet, so ordinary edits stay quiet.
fn refresh_message(w: &mut ServeWizardState) {
    if w.message.is_some() {
        w.message = w.validate().err().map(|(_, msg)| msg);
    }
}

/// Validate the form and stage an approval (no job runs until approved).
fn request_launch(w: &mut ServeWizardState) {
    match w.build_args() {
        Ok(args) => {
            let cmd = resolve_exe();
            let cmdline = format!("{} {}", exe_label(&cmd), args.join(" "));
            let mut body = vec![
                cmdline,
                String::new(),
                "This launches a local model server through the ROCm CLI.".to_string(),
            ];
            // Only claim an automatic port once the combination is valid — the
            // concrete endpoint is printed by the CLI after it leases one.
            if w.port_mode == PortMode::Auto {
                body.push(AUTO_PORT_NOTE.to_string());
            }
            body.push(if w.managed {
                "Managed: it will appear in the services manager and dashboard.".to_string()
            } else {
                "Foreground: it runs in this job console until stopped.".to_string()
            });
            let request = ApprovalRequest::new(format!("serve “{}”", w.model.trim()), body);
            w.message = None;
            w.approval = Some(PendingServe {
                cmd,
                args,
                request,
                choice: ApprovalChoice::default(),
            });
        }
        Err((field, msg)) => {
            // Move focus to the problem — expanding Advanced when the offending
            // row is hidden — so the fix is always visible, never guessed at.
            w.focus(field);
            w.message = Some(msg);
        }
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
    _jobs: &State,
    recipes: &[ModelRecipeSummary],
    theme: &Theme,
) {
    let inner = panel::bento(
        f,
        area,
        Some("Serve a model"),
        BoxRole::Primary,
        false,
        theme,
    );
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
    let vis = w.visible_fields();
    let focused = w.field.min(vis.len() - 1);
    let lines: Vec<Line> = vis
        .iter()
        .enumerate()
        .map(|(i, field)| field_line(*field, i == focused, w, has_recipes, theme))
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

    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            footer_hint(w, has_recipes),
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

/// Footer help, derived from focus and edit state. Every hint names the row's
/// role in words (`choice`, `editable`, `read only`) so meaning never depends
/// on colour or glyphs alone.
fn footer_hint(w: &ServeWizardState, has_recipes: bool) -> &'static str {
    if let Some(ed) = &w.editor {
        return match ed.field {
            Field::Port => {
                "editing port · ←→ cursor · Backspace delete · Enter accept · Esc cancel"
            }
            _ => "editing host · ←→ cursor · Backspace delete · Enter accept · Esc cancel",
        };
    }
    match w.current_field() {
        Field::Model if has_recipes => {
            "editable · Enter pick a recipe · type a name · Tab browse a path · Esc close"
        }
        Field::Model => "editable · type a name or path · Tab browse a path · Esc close",
        Field::Advanced if w.advanced_expanded => {
            "Enter or ← collapse advanced settings · defaults stay automatic"
        }
        Field::Advanced => "Enter or → expand advanced settings · defaults stay automatic",
        Field::Engine => "choice · ←→ pick an engine · automatic lets the CLI decide",
        Field::Device => "read only · GPU required · ROCm never falls back to CPU",
        Field::Host => "editable · Enter to edit the host · loopback only in Dash",
        Field::Port if w.port_mode == PortMode::Custom => {
            "choice · ←→ back to automatic · Enter to edit the port"
        }
        Field::Port => "choice · ←→ switch to custom · automatic picks a free local port",
        Field::Mode => "choice · ←→ managed or foreground",
        Field::Launch => "Enter to review the exact command · nothing runs before you approve",
    }
}

/// Render one row. Choice values carry chevrons, editable values brackets, and
/// the automatic device policy is plain read-only text.
fn field_line<'a>(
    field: Field,
    selected: bool,
    w: &'a ServeWizardState,
    has_recipes: bool,
    theme: &Theme,
) -> Line<'a> {
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

    if field == Field::Advanced {
        let chevron = if w.advanced_expanded { "▾" } else { "▸" };
        return Line::from(vec![
            Span::styled(marker, label_style),
            Span::styled("Advanced settings ", label_style),
            Span::styled(chevron, label_style),
            Span::styled(format!(" {}", w.advanced_summary()), value_style),
        ]);
    }

    let model_placeholder = if has_recipes {
        "(Enter to pick a recipe · type a name · Tab to browse)"
    } else {
        "(type a name / path, or Tab to browse)"
    };

    let label = match field {
        Field::Model => "Model",
        Field::Engine => "Engine",
        Field::Device => "Device",
        Field::Host => "Host",
        Field::Port => "Port",
        Field::Mode => "Mode",
        Field::Advanced | Field::Launch => unreachable!(),
    };

    let mut spans = vec![
        Span::styled(marker, label_style),
        Span::styled(format!("{label:<8}"), label_style),
    ];

    // The editor, when open, replaces its own row's value with a live caret.
    let editing = w.editor.as_ref().filter(|e| e.field == field);

    match field {
        Field::Model => spans.push(Span::styled(
            bracketed(&w.model, model_placeholder),
            value_style,
        )),
        Field::Engine => spans.push(Span::styled(
            chevroned(ENGINES[w.engine_idx.min(ENGINES.len() - 1)]),
            value_style,
        )),
        // Read-only: no chevrons, no brackets — nothing to change here.
        Field::Device => spans.push(Span::styled(DEVICE_POLICY, value_style)),
        Field::Host => {
            if let Some(ed) = editing {
                spans.extend(editor_spans(ed, value_style, theme));
            } else {
                spans.push(Span::styled(bracketed(&w.host, "(unset)"), value_style));
            }
        }
        Field::Port => {
            let mode = match w.port_mode {
                PortMode::Auto => "automatic",
                PortMode::Custom => "custom",
            };
            spans.push(Span::styled(chevroned(mode), value_style));
            if w.port_mode == PortMode::Custom {
                spans.push(Span::styled("  ", value_style));
                if let Some(ed) = editing {
                    spans.extend(editor_spans(ed, value_style, theme));
                } else {
                    spans.push(Span::styled(bracketed(&w.port, "(unset)"), value_style));
                }
            }
        }
        Field::Mode => spans.push(Span::styled(
            chevroned(if w.managed { "managed" } else { "foreground" }),
            value_style,
        )),
        Field::Advanced | Field::Launch => unreachable!(),
    }

    Line::from(spans)
}

/// Editable value with a visible caret (and a reversed run while the initial
/// value is still selected), rendered so the text stays contiguous.
fn editor_spans<'a>(ed: &'a InlineEditor, base: Style, theme: &Theme) -> Vec<Span<'a>> {
    let caret = Style::default()
        .bg(theme.accent)
        .fg(theme.bg)
        .add_modifier(Modifier::BOLD);
    let mut spans = vec![Span::styled("[", base)];
    if ed.selected && !ed.value.is_empty() {
        spans.push(Span::styled(ed.value.as_str(), caret));
    } else {
        let chars: Vec<char> = ed.value.chars().collect();
        let cut = ed.cursor.min(chars.len());
        let before: String = chars[..cut].iter().collect();
        if !before.is_empty() {
            spans.push(Span::styled(before, base));
        }
        if cut < chars.len() {
            spans.push(Span::styled(chars[cut].to_string(), caret));
            let after: String = chars[cut + 1..].iter().collect();
            if !after.is_empty() {
                spans.push(Span::styled(after, base));
            }
        } else {
            spans.push(Span::styled(" ", caret));
        }
    }
    spans.push(Span::styled("]", base));
    spans
}

fn bracketed(v: &str, placeholder: &'static str) -> String {
    if v.is_empty() {
        placeholder.to_string()
    } else {
        format!("[{v}]")
    }
}

fn chevroned(v: &str) -> String {
    format!("‹ {v} ›")
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

    fn feed(wiz: &mut Option<ServeWizardState>, jobs: &mut State, keys: &[KeyCode]) {
        for c in keys {
            on_key(wiz, jobs, &[], key(*c));
        }
    }

    /// Put focus on a row, expanding Advanced when that row is hidden.
    fn focus_field(w: &mut ServeWizardState, field: Field) {
        w.focus(field);
    }

    #[test]
    fn cycle_idx_wraps_both_directions() {
        assert_eq!(cycle_idx(0, 3, -1), 2);
        assert_eq!(cycle_idx(2, 3, 1), 0);
        assert_eq!(cycle_idx(0, 0, 1), 0);
    }

    // ---------------------------------------------------------------- defaults

    #[test]
    fn default_form_is_fully_automatic_and_collapsed() {
        let w = ServeWizardState::default();
        assert!(!w.advanced_expanded);
        assert_eq!(w.visible_fields(), BASIC_FIELDS);
        assert_eq!(w.engine_idx, ENGINE_AUTO);
        assert_eq!(ENGINES[w.engine_idx], "automatic");
        assert_eq!(w.host, "127.0.0.1");
        assert_eq!(w.port_mode, PortMode::Auto);
        assert!(w.managed);
        assert_eq!(w.advanced_summary(), "Automatic");
        assert!(!w.is_customized());
    }

    // ------------------------------------------------------------------- argv

    #[test]
    fn build_args_requires_a_model() {
        let w = ServeWizardState::default();
        let (field, msg) = w.build_args().unwrap_err();
        assert_eq!(field, Field::Model);
        assert_eq!(msg, "model is required");
    }

    #[test]
    fn default_argv_is_bare_serve_model() {
        let w = ServeWizardState {
            model: "qwen".into(),
            ..Default::default()
        };
        assert_eq!(w.build_args().unwrap(), vec!["serve", "qwen"]);
    }

    #[test]
    fn automatic_defaults_omit_engine_device_host_port_and_managed() {
        let w = ServeWizardState {
            model: "qwen".into(),
            ..Default::default()
        };
        let args = w.build_args().unwrap();
        for flag in ["--engine", "--device", "--host", "--port", "--managed"] {
            assert!(!args.contains(&flag.to_string()), "{flag} must be omitted");
        }
    }

    #[test]
    fn explicit_overrides_emit_only_their_own_flags() {
        let w = ServeWizardState {
            model: "glm".into(),
            engine_idx: 2, // vllm
            host: "localhost".into(),
            port_mode: PortMode::Custom,
            port: "8000".into(),
            managed: false,
            ..Default::default()
        };
        assert_eq!(
            w.build_args().unwrap(),
            vec![
                "serve",
                "glm",
                "--engine",
                "vllm",
                "--host",
                "localhost",
                "--port",
                "8000",
                "--foreground",
            ]
        );
    }

    #[test]
    fn custom_port_on_default_host_emits_port_only() {
        let w = ServeWizardState {
            model: "m".into(),
            port_mode: PortMode::Custom,
            port: "11500".into(),
            ..Default::default()
        };
        assert_eq!(
            w.build_args().unwrap(),
            vec!["serve", "m", "--port", "11500"]
        );
    }

    #[test]
    fn build_args_rejects_out_of_range_and_zero_ports() {
        for bad in ["99999", "0", "", "  "] {
            let w = ServeWizardState {
                model: "m".into(),
                port_mode: PortMode::Custom,
                port: bad.into(),
                ..Default::default()
            };
            let (field, msg) = w.build_args().unwrap_err();
            assert_eq!(field, Field::Port, "{bad:?}");
            assert!(msg.contains("1–65535"), "{bad:?}: {msg}");
        }
    }

    #[test]
    fn empty_host_is_rejected_with_a_plain_fix() {
        let w = ServeWizardState {
            model: "m".into(),
            host: "   ".into(),
            ..Default::default()
        };
        let (field, msg) = w.build_args().unwrap_err();
        assert_eq!(field, Field::Host);
        assert!(msg.contains("host is required"), "{msg}");
    }

    #[test]
    fn noncanonical_host_with_auto_port_is_blocked_with_the_contract_message() {
        for host in ["localhost", "::1", "0.0.0.0", "192.168.1.5"] {
            let w = ServeWizardState {
                model: "m".into(),
                host: host.into(),
                port_mode: PortMode::Auto,
                ..Default::default()
            };
            let (field, msg) = w.build_args().unwrap_err();
            assert_eq!(field, Field::Port, "{host}");
            assert_eq!(msg, CUSTOM_HOST_NEEDS_PORT, "{host}");
        }
    }

    #[test]
    fn loopback_host_with_custom_port_is_allowed() {
        for host in ["localhost", "::1", "[::1]", "LocalHost"] {
            let w = ServeWizardState {
                model: "m".into(),
                host: host.into(),
                port_mode: PortMode::Custom,
                port: "9000".into(),
                ..Default::default()
            };
            assert_eq!(
                w.build_args().unwrap(),
                vec!["serve", "m", "--host", host, "--port", "9000"],
                "{host}"
            );
        }
    }

    #[test]
    fn public_host_never_reaches_approval_even_with_a_custom_port() {
        // Dash has no --allow-public-bind confirmation and no endpoint-key UI,
        // so a public bind must be requested from the CLI instead.
        for host in ["0.0.0.0", "192.168.1.5", "::", "example.invalid"] {
            let w = ServeWizardState {
                model: "m".into(),
                host: host.into(),
                port_mode: PortMode::Custom,
                port: "8000".into(),
                ..Default::default()
            };
            let (field, msg) = w.build_args().unwrap_err();
            assert_eq!(field, Field::Host, "{host}");
            assert_eq!(msg, PUBLIC_HOST_NEEDS_CLI, "{host}");
        }
    }

    #[test]
    fn public_host_launch_stages_no_approval() {
        let mut wiz = Some(ServeWizardState {
            model: "m".into(),
            host: "0.0.0.0".into(),
            port_mode: PortMode::Custom,
            port: "8000".into(),
            ..Default::default()
        });
        let mut jobs = State::default();
        wiz.as_mut().unwrap().focus(Field::Launch);
        let fx = on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        assert!(fx.is_empty());
        let w = wiz.as_ref().unwrap();
        assert!(w.approval.is_none(), "a public bind never reaches approval");
        assert!(w.advanced_expanded, "the offending row is revealed");
        assert_eq!(w.current_field(), Field::Host);
        assert_eq!(w.message.as_deref(), Some(PUBLIC_HOST_NEEDS_CLI));
        assert!(jobs.jobs.is_empty());
    }

    #[test]
    fn is_loopback_host_matches_the_cli_accepted_spellings() {
        for ok in ["127.0.0.1", "localhost", "::1", "[::1]", "LOCALHOST"] {
            assert!(is_loopback_host(ok), "{ok}");
        }
        for bad in ["0.0.0.0", "127.0.0.2", "::", "192.168.1.5", ""] {
            assert!(!is_loopback_host(bad), "{bad}");
        }
    }

    // -------------------------------------------------------- disclosure/focus

    #[test]
    fn advanced_expands_and_collapses_only_while_focused_and_keeps_focus() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        // Model has focus: Right must not expand anything.
        feed(&mut wiz, &mut jobs, &[KeyCode::Right]);
        assert!(!wiz.as_ref().unwrap().advanced_expanded);

        // Down → Advanced, Enter expands, focus stays on Advanced.
        feed(&mut wiz, &mut jobs, &[KeyCode::Down, KeyCode::Enter]);
        {
            let w = wiz.as_ref().unwrap();
            assert!(w.advanced_expanded);
            assert_eq!(w.visible_fields(), FIELDS);
            assert_eq!(w.current_field(), Field::Advanced);
        }
        // Enter collapses again, still on Advanced.
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        {
            let w = wiz.as_ref().unwrap();
            assert!(!w.advanced_expanded);
            assert_eq!(w.current_field(), Field::Advanced);
        }
        // ← collapses / → expands from the same row.
        feed(&mut wiz, &mut jobs, &[KeyCode::Right]);
        assert!(wiz.as_ref().unwrap().advanced_expanded);
        feed(&mut wiz, &mut jobs, &[KeyCode::Left]);
        assert!(!wiz.as_ref().unwrap().advanced_expanded);
        assert_eq!(wiz.as_ref().unwrap().current_field(), Field::Advanced);
    }

    #[test]
    fn expanded_focus_order_is_model_advanced_engine_device_host_port_mode_launch() {
        let mut w = ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        };
        let mut seen = Vec::new();
        for _ in 0..FIELDS.len() {
            seen.push(w.current_field());
            w.move_field(1);
        }
        assert_eq!(seen, FIELDS.to_vec());
        // Clamped at the end.
        assert_eq!(w.current_field(), Field::Launch);
    }

    #[test]
    fn collapsing_preserves_overrides_and_reports_customized() {
        let mut w = ServeWizardState {
            advanced_expanded: true,
            engine_idx: 1,
            port_mode: PortMode::Custom,
            port: "8000".into(),
            ..Default::default()
        };
        w.advanced_expanded = false;
        assert_eq!(w.engine_idx, 1);
        assert_eq!(w.port, "8000");
        assert_eq!(w.advanced_summary(), "Customized");
    }

    #[test]
    fn invalid_hidden_custom_port_says_it_needs_attention() {
        let w = ServeWizardState {
            model: "m".into(),
            port_mode: PortMode::Custom,
            port: "abc".into(),
            ..Default::default()
        };
        assert!(!w.advanced_expanded);
        assert_eq!(w.advanced_summary(), "Customized · port needs attention");
        assert!(w.build_args().is_err(), "launch stays blocked");
    }

    #[test]
    fn out_of_range_hidden_port_also_needs_attention() {
        let w = ServeWizardState {
            port_mode: PortMode::Custom,
            port: "70000".into(),
            ..Default::default()
        };
        assert_eq!(w.advanced_summary(), "Customized · port needs attention");
    }

    #[test]
    fn invalid_combination_expands_advanced_and_focuses_the_problem() {
        let mut wiz = Some(ServeWizardState {
            model: "m".into(),
            host: "localhost".into(),
            ..Default::default()
        });
        let mut jobs = State::default();
        wiz.as_mut().unwrap().field = 2; // Launch (collapsed order)
        let fx = on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        assert!(fx.is_empty(), "nothing may run");
        let w = wiz.as_ref().unwrap();
        assert!(w.approval.is_none(), "invalid combos never reach approval");
        assert!(w.advanced_expanded, "the hidden problem is revealed");
        assert_eq!(w.current_field(), Field::Port);
        assert_eq!(w.message.as_deref(), Some(CUSTOM_HOST_NEEDS_PORT));
        assert_eq!(w.host, "localhost", "both values preserved");
        assert_eq!(w.port_mode, PortMode::Auto);
    }

    // --------------------------------------------------------------- controls

    #[test]
    fn left_right_cycles_engine_including_automatic() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Engine);
        feed(&mut wiz, &mut jobs, &[KeyCode::Right]);
        assert_eq!(ENGINES[wiz.as_ref().unwrap().engine_idx], "lemonade");
        feed(&mut wiz, &mut jobs, &[KeyCode::Right]);
        assert_eq!(ENGINES[wiz.as_ref().unwrap().engine_idx], "vllm");
        feed(&mut wiz, &mut jobs, &[KeyCode::Right]);
        assert_eq!(wiz.as_ref().unwrap().engine_idx, ENGINE_AUTO, "wraps");
    }

    #[test]
    fn left_right_toggles_port_mode_only_when_focused() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Device);
        feed(&mut wiz, &mut jobs, &[KeyCode::Right]);
        assert_eq!(
            wiz.as_ref().unwrap().port_mode,
            PortMode::Auto,
            "Device is read-only and cannot change the port"
        );
        focus_field(wiz.as_mut().unwrap(), Field::Port);
        feed(&mut wiz, &mut jobs, &[KeyCode::Right]);
        assert_eq!(wiz.as_ref().unwrap().port_mode, PortMode::Custom);
        feed(&mut wiz, &mut jobs, &[KeyCode::Left]);
        assert_eq!(wiz.as_ref().unwrap().port_mode, PortMode::Auto);
    }

    #[test]
    fn mode_toggles_between_managed_and_foreground() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Mode);
        feed(&mut wiz, &mut jobs, &[KeyCode::Char(' ')]);
        assert!(!wiz.as_ref().unwrap().managed);
        feed(&mut wiz, &mut jobs, &[KeyCode::Left]);
        assert!(wiz.as_ref().unwrap().managed);
    }

    #[test]
    fn typing_only_edits_the_model_row() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Host);
        for k in typed("zzz") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        assert_eq!(
            wiz.as_ref().unwrap().host,
            "127.0.0.1",
            "host only changes through the inline editor"
        );
        focus_field(wiz.as_mut().unwrap(), Field::Model);
        for k in typed("qwen") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        assert_eq!(wiz.as_ref().unwrap().model, "qwen");
        feed(&mut wiz, &mut jobs, &[KeyCode::Backspace]);
        assert_eq!(wiz.as_ref().unwrap().model, "qwe");
    }

    // ----------------------------------------------------------- inline editor

    #[test]
    fn host_editor_selects_on_entry_and_first_char_replaces() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Host);
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        {
            let ed = wiz.as_ref().unwrap().editor.as_ref().unwrap();
            assert_eq!(ed.field, Field::Host);
            assert!(ed.selected);
            assert_eq!(ed.value, "127.0.0.1");
        }
        for k in typed("0.0.0.0") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        assert_eq!(
            wiz.as_ref().unwrap().editor.as_ref().unwrap().value,
            "0.0.0.0"
        );
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        let w = wiz.as_ref().unwrap();
        assert!(w.editor.is_none());
        assert_eq!(w.host, "0.0.0.0");
    }

    #[test]
    fn editor_cursor_insert_and_backspace() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            host: "abc".into(),
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Host);
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter, KeyCode::Left]);
        // Left cleared the selection and moved the cursor to before 'c'.
        {
            let ed = wiz.as_ref().unwrap().editor.as_ref().unwrap();
            assert!(!ed.selected);
            assert_eq!(ed.cursor, 2);
        }
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('X')));
        assert_eq!(wiz.as_ref().unwrap().editor.as_ref().unwrap().value, "abXc");
        feed(&mut wiz, &mut jobs, &[KeyCode::Backspace]);
        assert_eq!(wiz.as_ref().unwrap().editor.as_ref().unwrap().value, "abc");
        feed(&mut wiz, &mut jobs, &[KeyCode::Right, KeyCode::Right]);
        assert_eq!(wiz.as_ref().unwrap().editor.as_ref().unwrap().cursor, 3);
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        assert_eq!(wiz.as_ref().unwrap().host, "abc");
    }

    #[test]
    fn editor_escape_restores_the_original_value() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Host);
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        for k in typed("nonsense") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        feed(&mut wiz, &mut jobs, &[KeyCode::Esc]);
        let w = wiz.as_ref().unwrap();
        assert!(w.editor.is_none(), "Esc closes the editor");
        assert!(wiz.is_some(), "Esc in the editor never closes the overlay");
        assert_eq!(w.host, "127.0.0.1", "original restored");
    }

    #[test]
    fn host_editor_trims_and_rejects_blank() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Host);
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        for k in typed("  ") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        {
            let w = wiz.as_ref().unwrap();
            assert!(w.editor.is_some(), "blank host keeps the editor open");
            assert_eq!(w.host, "127.0.0.1", "form untouched");
            assert!(
                w.message
                    .as_deref()
                    .unwrap()
                    .contains("host cannot be empty")
            );
        }
        for k in typed(" 10.0.0.7 ") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        let w = wiz.as_ref().unwrap();
        assert!(w.editor.is_none());
        assert_eq!(w.host, "10.0.0.7", "trimmed on accept");
    }

    #[test]
    fn port_editor_accepts_digits_only() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            port_mode: PortMode::Custom,
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Port);
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        for k in typed("80a0!") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        assert_eq!(wiz.as_ref().unwrap().editor.as_ref().unwrap().value, "800");
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        assert_eq!(wiz.as_ref().unwrap().port, "800");
    }

    #[test]
    fn port_editor_rejects_an_empty_value_like_host_does() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            model: "m".into(),
            port_mode: PortMode::Custom,
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Port);
        // Enter selects "11435"; Backspace clears the whole selection.
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter, KeyCode::Backspace]);
        assert_eq!(wiz.as_ref().unwrap().editor.as_ref().unwrap().value, "");
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        {
            let w = wiz.as_ref().unwrap();
            assert!(w.editor.is_some(), "empty port keeps the editor open");
            assert_eq!(w.port, "11435", "form untouched");
            let msg = w.message.as_deref().unwrap();
            assert!(msg.contains("port cannot be empty"), "{msg}");
            assert!(msg.contains("1–65535"), "{msg}");
        }
        // Typing a real value accepts normally and clears the complaint.
        for k in typed("9001") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        let w = wiz.as_ref().unwrap();
        assert!(w.editor.is_none());
        assert_eq!(w.port, "9001");
        assert!(w.message.is_none(), "a fixed form stops complaining");
    }

    #[test]
    fn escaping_an_editor_recomputes_an_outstanding_complaint() {
        // A blank-host complaint must not survive the Esc that restores a
        // perfectly good host…
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            model: "m".into(),
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Host);
        feed(
            &mut wiz,
            &mut jobs,
            &[KeyCode::Enter, KeyCode::Backspace, KeyCode::Enter],
        );
        assert!(wiz.as_ref().unwrap().message.is_some());
        feed(&mut wiz, &mut jobs, &[KeyCode::Esc]);
        let w = wiz.as_ref().unwrap();
        assert_eq!(w.host, "127.0.0.1");
        assert!(w.message.is_none(), "stale complaint cleared by the fix");

        // …but a still-broken form keeps saying why.
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            model: "m".into(),
            host: "localhost".into(),
            ..Default::default()
        });
        wiz.as_mut().unwrap().message = Some(CUSTOM_HOST_NEEDS_PORT.to_string());
        focus_field(wiz.as_mut().unwrap(), Field::Host);
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter, KeyCode::Esc]);
        let w = wiz.as_ref().unwrap();
        assert_eq!(w.host, "localhost");
        assert_eq!(w.message.as_deref(), Some(CUSTOM_HOST_NEEDS_PORT));
    }

    #[test]
    fn enter_on_auto_port_advances_instead_of_editing() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Port);
        feed(&mut wiz, &mut jobs, &[KeyCode::Enter]);
        let w = wiz.as_ref().unwrap();
        assert!(w.editor.is_none());
        assert_eq!(w.current_field(), Field::Mode);
    }

    #[test]
    fn editor_navigation_keys_cannot_move_form_focus() {
        let mut wiz = Some(ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        });
        let mut jobs = State::default();
        focus_field(wiz.as_mut().unwrap(), Field::Host);
        let before = wiz.as_ref().unwrap().field;
        feed(
            &mut wiz,
            &mut jobs,
            &[KeyCode::Enter, KeyCode::Up, KeyCode::Down],
        );
        let w = wiz.as_ref().unwrap();
        assert!(w.editor.is_some());
        assert_eq!(w.field, before, "form navigation is inert while editing");
    }

    // -------------------------------------------------------------- approval

    #[test]
    fn launch_requires_approval_then_spawns_job() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        for k in typed("qwen") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        assert_eq!(wiz.as_ref().unwrap().model, "qwen");
        wiz.as_mut().unwrap().focus(Field::Launch);
        let fx = on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        assert!(fx.is_empty(), "launch must not run before approval");
        assert!(wiz.as_ref().unwrap().approval.is_some());
        assert!(jobs.jobs.is_empty());

        let fx = on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('y')));
        assert_eq!(fx.len(), 1);
        assert!(matches!(fx[0], SideEffect::SpawnJob { .. }));
        let w = wiz.as_ref().unwrap();
        assert!(w.approval.is_none());
        assert_eq!(w.active_job.as_deref(), Some("serve-qwen"));
        assert_eq!(jobs.jobs.len(), 1);
    }

    #[test]
    fn approval_shows_exact_argv_and_the_automatic_port_phrase() {
        let mut w = ServeWizardState {
            model: "qwen".into(),
            ..Default::default()
        };
        request_launch(&mut w);
        let pending = w.approval.as_ref().unwrap();
        assert_eq!(pending.args, vec!["serve", "qwen"]);
        let body = pending.request.body.join("\n");
        assert!(body.contains("serve qwen"), "{body}");
        assert!(body.contains(AUTO_PORT_NOTE), "{body}");
    }

    #[test]
    fn approval_omits_the_automatic_phrase_for_a_custom_port() {
        let mut w = ServeWizardState {
            model: "qwen".into(),
            port_mode: PortMode::Custom,
            port: "8000".into(),
            ..Default::default()
        };
        request_launch(&mut w);
        let pending = w.approval.as_ref().unwrap();
        assert!(pending.args.windows(2).any(|p| p == ["--port", "8000"]));
        assert!(!pending.request.body.join("\n").contains(AUTO_PORT_NOTE));
    }

    #[test]
    fn empty_model_launch_sets_message_not_approval() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        wiz.as_mut().unwrap().focus(Field::Launch);
        let fx = on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        assert!(fx.is_empty());
        let w = wiz.as_ref().unwrap();
        assert!(w.approval.is_none());
        assert_eq!(w.message.as_deref(), Some("model is required"));
        assert_eq!(
            w.current_field(),
            Field::Model,
            "focus lands on the problem"
        );
    }

    #[test]
    fn deny_cancels_without_spawning() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        wiz.as_mut().unwrap().model = "m".into();
        wiz.as_mut().unwrap().focus(Field::Launch);
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
        wiz.as_mut().unwrap().focus(Field::Launch);
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('y')));
        assert!(wiz.as_ref().unwrap().active_job.is_some());
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('q')));
        assert!(wiz.is_none(), "q must close the overlay even mid-job");
    }

    #[test]
    fn esc_closes_overlay_while_running_then_dismisses_when_terminal() {
        // Running: Esc leaves the whole overlay (the launch keeps running in the
        // background) — the user is never trapped during the readiness wait.
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        wiz.as_mut().unwrap().model = "m".into();
        wiz.as_mut().unwrap().focus(Field::Launch);
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('y')));
        assert!(wiz.as_ref().unwrap().active_job.is_some());
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Esc));
        assert!(wiz.is_none(), "Esc closes the overlay even mid-job");

        // Terminal: Esc dismisses the console back to the form (wizard stays open).
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        wiz.as_mut().unwrap().model = "m".into();
        wiz.as_mut().unwrap().focus(Field::Launch);
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Char('y')));
        let job_id = wiz.as_ref().unwrap().active_job.clone().unwrap();
        jobs.apply(StateEvent::JobDone {
            id: job_id,
            code: 0,
        });
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Esc));
        assert!(wiz.as_ref().unwrap().active_job.is_none());
        assert!(
            wiz.is_some(),
            "dismissing a finished console keeps the wizard open"
        );
    }

    #[test]
    fn relaunch_while_prior_job_running_surfaces_message_not_stale_console() {
        // The reducer no-ops a StartJob for a still-running id. spawn_serve must
        // NOT claim success (set active_job) when no SpawnJob was emitted.
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();

        // First launch of "qwen": real spawn.
        for k in typed("qwen") {
            on_key(&mut wiz, &mut jobs, &[], k);
        }
        wiz.as_mut().unwrap().focus(Field::Launch);
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
        wiz2.as_mut().unwrap().focus(Field::Launch);
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

    // ------------------------------------------------------- picker / browser

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
    fn enter_on_model_opens_picker_and_choice_fills_model_only() {
        let recipes = vec![ModelRecipeSummary {
            id: "GLM-4".into(),
            aliases: vec!["glm".into()],
            task: "chat".into(),
            preferred_engine: Some("vllm".into()),
        }];
        let mut wiz = Some(ServeWizardState::default()); // field 0 = Model
        let mut jobs = State::default();
        on_key(&mut wiz, &mut jobs, &recipes, key(KeyCode::Enter));
        assert!(wiz.as_ref().unwrap().picker.is_some());
        on_key(&mut wiz, &mut jobs, &recipes, key(KeyCode::Enter));
        let w = wiz.as_ref().unwrap();
        assert!(w.picker.is_none());
        assert_eq!(w.model, "GLM-4");
        assert_eq!(
            w.engine_idx, ENGINE_AUTO,
            "a recipe never forces an engine override — the CLI resolves it"
        );
        assert_eq!(w.build_args().unwrap(), vec!["serve", "GLM-4"]);
    }

    #[test]
    fn enter_on_model_advances_when_no_recipes() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        assert!(wiz.as_ref().unwrap().picker.is_none());
        assert_eq!(
            wiz.as_ref().unwrap().current_field(),
            Field::Advanced,
            "Enter advances to the disclosure row"
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
        for k in typed("qwen") {
            on_key(&mut wiz, &mut jobs, &recipes, k);
        }
        on_key(&mut wiz, &mut jobs, &recipes, key(KeyCode::Enter));
        let picker = wiz.as_ref().unwrap().picker.as_ref().unwrap();
        assert_eq!(picker.query, "qwen");
        assert_eq!(picker.filtered(&recipes).len(), 1, "pre-narrowed to Qwen");
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

    // ---------------------------------------------------------------- render

    fn render(w: &ServeWizardState, jobs: &State) -> String {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        let theme = Theme::from_name("default-dark");
        let backend = TestBackend::new(120, 30);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw_serve_wizard(f, f.area(), w, jobs, &[], &theme))
            .unwrap();
        let buf = term.backend().buffer().clone();
        buf.content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    /// A rendered form row starts with the 2-cell focus marker and an
    /// 8-cell-padded label, so `Mode` is `"Mode    "` and can never be matched
    /// by the `Model` row. Bare substrings are not precise enough here.
    fn row_label(label: &str) -> String {
        format!("{label:<8}")
    }

    #[test]
    fn collapsed_render_shows_only_model_advanced_and_launch() {
        let w = ServeWizardState::default();
        let out = render(&w, &State::default());
        assert!(out.contains("Serve a model"), "titled overlay");
        assert!(out.contains(&row_label("Model")), "model row");
        assert!(out.contains("Advanced settings ▸"), "collapsed disclosure");
        assert!(out.contains("Automatic"), "automatic summary");
        assert!(out.contains("[ Launch ]"), "launch action");
        for hidden in ["Engine", "Device", "Host", "Port", "Mode"] {
            assert!(
                !out.contains(&row_label(hidden)),
                "{hidden} row must stay hidden: {out:?}"
            );
        }
    }

    #[test]
    fn expanded_render_inlines_the_advanced_rows_without_a_second_modal() {
        let w = ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        };
        let out = render(&w, &State::default());
        assert!(out.contains("Advanced settings ▾"), "expanded disclosure");
        for shown in ["Model", "Engine", "Device", "Host", "Port", "Mode"] {
            assert!(
                out.contains(&row_label(shown)),
                "{shown} row missing: {out:?}"
            );
        }
        assert!(out.contains("[ Launch ]"), "launch action");
        // Role affordances: chevrons for choices, brackets for editable text,
        // plain read-only text for the GPU-required policy.
        assert!(out.contains("‹ automatic ›"), "choice chevrons: {out:?}");
        assert!(out.contains("[127.0.0.1]"), "editable brackets: {out:?}");
        assert!(
            out.contains(DEVICE_POLICY),
            "read-only device text: {out:?}"
        );
        assert!(!out.contains("Review:"), "no second modal");
    }

    #[test]
    fn collapsed_render_reports_a_broken_hidden_port() {
        let w = ServeWizardState {
            model: "m".into(),
            port_mode: PortMode::Custom,
            port: "abc".into(),
            ..Default::default()
        };
        let out = render(&w, &State::default());
        assert!(out.contains("port needs attention"), "{out:?}");
    }

    #[test]
    fn footer_names_the_role_of_the_focused_row() {
        let mut w = ServeWizardState {
            advanced_expanded: true,
            ..Default::default()
        };
        w.focus(Field::Engine);
        assert!(render(&w, &State::default()).contains("choice"));
        w.focus(Field::Device);
        assert!(render(&w, &State::default()).contains("read only"));
        w.focus(Field::Host);
        assert!(render(&w, &State::default()).contains("editable"));
        // Port names the direction it would move in, not a generic pair.
        w.focus(Field::Port);
        assert!(render(&w, &State::default()).contains("choice · ←→ switch to custom"));
        w.port_mode = PortMode::Custom;
        assert!(
            render(&w, &State::default())
                .contains("choice · ←→ back to automatic · Enter to edit the port")
        );
        w.focus(Field::Host);
        let host = w.host.clone();
        w.editor = Some(InlineEditor::new(Field::Host, &host));
        assert!(render(&w, &State::default()).contains("Esc cancel"));
    }

    #[test]
    fn no_generic_change_footer_remains() {
        let w = ServeWizardState::default();
        assert!(!render(&w, &State::default()).contains("←→ change"));
    }

    #[test]
    fn snapshot_shows_approval_modal_on_launch() {
        let mut wiz = Some(ServeWizardState::default());
        let mut jobs = State::default();
        wiz.as_mut().unwrap().model = "qwen".into();
        wiz.as_mut().unwrap().focus(Field::Launch);
        on_key(&mut wiz, &mut jobs, &[], key(KeyCode::Enter));
        let out = render(wiz.as_ref().unwrap(), &jobs);
        assert!(out.contains("Review:"), "approval modal shown");
        assert!(out.contains("serve"), "describes the gated launch");
        assert!(out.contains("Approve"), "approve button present");
        assert!(out.contains("endpoint shown after launch"), "{out:?}");
    }
}
