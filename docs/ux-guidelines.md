# ROCm CLI UX Guidelines

These guidelines are project constraints for user-facing ROCm CLI flows.

## Audience

- Assume most users are non-technical Windows users.
- Treat Linux users as non-technical unless a workflow explicitly targets
  developer/debug usage.
- Use simple English. Avoid runtime, wheel, environment, adapter, and command
  jargon in first-run UI unless there is no simpler accurate wording.

## First-Time Setup

- First-time setup must be a dedicated setup prompt/screen before the main TUI.
- Do not require users to know or type slash commands to complete setup.
- Do not use an LLM assistant for bootstrap/setup. Setup should be
  deterministic: choose folder, review install, run install, show success, then
  continue to the main TUI.
- First-time setup should focus on one job: installing the TheRock ROCm Python
  wheels into a managed Python venv.
- Prompt the user for the venv install location with a simple default.
- Show visible progress/log output for what is currently being installed.
- Bootstrap portable/local userland tools where feasible. Do not ask users to
  manually install Python, curl, Cargo, or similar prerequisites when rocm-cli
  can download and manage a local copy.
- Any mutating action must ask for explicit approval before it runs.
- No CPU fallback is allowed for GPU-required paths.

## Persistence

- First-time setup choices must persist across runs.
- Store user-facing setup settings in JSON under the user's home `.rocm`
  directory, for example `~/.rocm/config.json`.
- Keep the persisted format readable and recoverable.

## Permissions

- Default mode asks before changing local state.
- Full access mode is explicit opt-in, explained in plain English, and
  resettable at any time with `/permissions`.
- Full access skips rocm-cli confirmation prompts only. It does not bypass OS
  elevation, driver requirements, missing GPU support, or no-fallback rules.

## Main TUI

- The normal TUI should be useful after setup, not a setup command cheat sheet.
- The default feel should be a friendly control room for laymen, not a dry log
  viewer. Use plain labels, color, motion/progress, and focused review cards to
  make the next action obvious.
- Verbosity is a hard no. First-view command output should be minimal and
  plain-English. The only intentionally verbose surface is the foreground
  install/progress card while pip or another installer is actively running.
- The Home dashboard should use real arrow-key action rows plus a plain-English
  detail pane. Do not render Home as a transcript-like block with a handmade
  prompt marker.
- Use overlapping modal-style cards whenever a user is making a focused
  decision or watching a contained operation, such as approvals, setup/install
  progress, log details, model tool-call review, command help, clear
  confirmation, quit confirmation, and short error/fix prompts. Keep ordinary
  browsing/list screens as stable panes so arrow-key navigation remains
  predictable.
- A visible modal-style card owns keyboard focus. While an install/progress
  card is open, arrows, Tab, Enter, and Esc must not operate hidden rows,
  slash completions, or prompt actions behind the card.
- Color should carry meaning: cyan for the active focus, green for ready/safe,
  yellow/orange for work in progress or caution, and AMD red only for real
  errors or destructive danger.
- Do not show a persistent Activity pane by default.
- Do not show a prompt/composer on screens where no assistant or server session
  exists. Use a main menu with clickable/arrow-key rows until the user starts or
  opens a chat session.
- Treat slash commands as entry points into navigable screens, not as wrappers
  around non-TUI command output.
- Typed slash commands with arguments should prefill the same guided screen
  whenever possible. For example, `/serve MODEL ...` should open the serve
  wizard with Start reachable by Enter, and `/model qwen` should highlight the
  matching model recipe in the model picker, not require prompt editing before
  the user can continue.
- Prefer arrow-key lists, Enter actions, Esc back, and clear footer hints over
  requiring users to type subcommands.
- When a screen has item actions such as install, remove, choose, stop,
  restart, send, edit, or back, expose those as selectable rows.
  Use F5 for ordinary refresh instead of adding a Refresh row, except on
  screens where checking for updates is the main task.
  Letter shortcuts may exist, but they must not be required or be the main
  instruction.
- Short-lived cards such as Help, Clear, and Quit must not replace the current
  screen. They should overlap it, consume arrow keys while open, close with
  Esc, and return the user to the same row they were on.
- Keep advanced setup/import/adopt actions behind a clearly selectable
  `Advanced options` row when the first-view task is ordinary setup or
  selection. The advanced screen must still be arrow-key navigable.
- Do not present rollback as a normal ROCm Installs action. Users should switch
  ROCm versions by choosing an installed item from the list; any legacy typed
  rollback path should explain that simpler picker flow.
- First-time setup folder selection must be arrow-friendly. Left/Right on the
  folder row should cycle easy folder choices; Enter may still open manual text
  entry for users who need a custom path.
- Folder pickers must also be mouse-friendly. Clicking a folder should select
  or open it directly, and the chosen path should be shown as a full path.
- Do not show internal launcher/cache/tool folders as attractive install
  choices. Keep the default install path simple and user-owned.
- Commands that send prompts or start provider work, such as `/chat <prompt>`,
  should show a review/send screen first unless the user is already in an
  explicit send action.
- Keep non-TUI commands available for scripting, automation, and debugging, but
  do not dump their raw text into the TUI as the primary experience.
- Put current operation state in the footer/status area. Show live command
  output in the active screen when it is part of the user's current task, and
  put history in explicit log views.
- Only show install/service logs in the foreground progress or details card for
  the active operation. Avoid duplicate background log text behind a modal.
- Do not let Back/Esc hide the screen that owns a mutating running command such
  as install, update, engine setup, service lifecycle, or automation approval.
  Keep live progress visible until the command finishes, then let the user
  leave.
- Passive background checks, such as Doctor refresh, may be closed while they
  finish as long as they do not block later setup actions or dump output into
  the transcript.
- Doctor must not interrupt an active install, update, service action, chat,
  plan, or approval. Keep the current screen visible and explain that Doctor
  can run after the active work finishes.
- Usage/error text in the TUI should be a plain fix in the relevant screen.
  Reserve raw command syntax for explicit help or non-TUI command output.
- The `?` shortcut is contextual. It must not open command help while setup,
  install progress, approval cards, or chat input own the screen.
- Service lists in the TUI should show living services only by default. Failed
  or stopped history belongs behind explicit logs/details or a non-TUI
  `--all` style command.

## Command Navigability Baseline

All advertised slash commands should either perform a clear immediate action
(`/clear`, `/quit`, `/exit`) or open exactly one focused TUI surface. They must
not append raw command reports to the transcript as the primary experience.
Hidden compatibility aliases should follow the same navigability rules when
typed, even when they are intentionally left out of completions and help.

Current navigable surfaces include `/home`, `/doctor`, `/setup`, `/permissions`,
`/runtimes`, `/engine`, `/model`, `/plan`, `/config`, `/automations`,
`/reviews`, `/approve`, `/reject`, `/edit`, `/install`, `/services`, `/logs`,
`/gpu`, `/update`, `/daemon`, `/chat`, `/provider`, `/uninstall`, `/comfyui`,
and `/serve`.

Acceptance for each command surface: opening the command shows a stable screen,
Up/Down and Tab/Shift+Tab move the current choice when there is more than one
row, Enter performs the primary action or opens details, PageUp/PageDown scroll
long details where that surface has a detail pane, Esc returns to Home when a
screen was opened from the dashboard, and mutating actions explain plainly what
will change before approval.

Command navigability follow-up notes:

- No locally tracked TUI command navigability gaps are open after the
  provider-key, Logs/service detail, all-command navigability, model-picker,
  install-completion, Shift+Tab reverse-navigation, completion simplification,
  default-runtime validation, progress-label, engine-detail, setup-cancel,
  body-visible error, and async Doctor background-blocking follow-ups.
- Keep primary panes free of raw backend labels after new command surfaces are
  added; raw command output belongs in Logs or explicit help/debug views.
- Hide advanced file locations in TUI first views by default and expose them
  through an explicit selectable row such as `Show file locations`.
- Keep backend labels such as wheels, tarballs, runtime keys, registries, and
  daemon policy out of first-visible TUI screens unless the user opens an
  explicit advanced/debug view.
