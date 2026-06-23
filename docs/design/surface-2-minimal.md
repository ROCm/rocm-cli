# Surface 2 — Bare `rocm` minimal launcher

**What it is.** The new front door. Typing `rocm` with no arguments opens a
compact, single-screen **intent menu** — not the full dashboard. It answers one
question, *"What would you like to do?"*, with a deliberately narrow set of rows
drawn straight from the PRD / 1-pager / AAI demos. Pick a row, press Enter, get
walked to the outcome. The full dash is always one row away.

**Design inspiration.** This surface follows the `ccstatusline` configuration TUI
pattern in [`plans/experiment/rocm-minimal-tui-pattern-idea.png`](../../plans/experiment/rocm-minimal-tui-pattern-idea.png):
a left-aligned, single screen with **a live status-preview strip on top**, an
icon+label menu below it, one clearly-focused row (`▸` + accent), sections
separated by whitespace, unavailable rows greyed with an inline reason, and a
friendly footer tip. We keep that skeleton and dress it in the rocm glyph/color
language. The status strip is where rocm's "live instrument" identity shows up
even on the smallest surface — it shows real GPU/serving telemetry, not a static
banner.

**Why it exists.** The 1-pager's thesis is *intent → outcome*: "the user journey
starts with a developer stating what they want to accomplish." The minimal
launcher is the deterministic, non-LLM expression of that for a non-technical
user — exactly the "dedicated prompt/screen before the main TUI"
(ux-guidelines:15) that the audit says is missing (F104 never auto-triggers).

**Scope discipline.** This surface is intentionally *narrow*. It is not the dash.
Its rows are the AAI demo verbs and nothing else:

| Row | Maps to | Status |
|---|---|---|
| **Serve a model** | serve wizard (F195) | live |
| **Set up this system** | setup/install wizard (F104/F001) | live |
| **Diagnose & fix** | doctor surface (F184) | live |
| **Chat** | chat session (F185) | live |
| **Optimize a model** | *planned — hyperloom skill* | **aspirational** (shown `soon`, dimmed) |
| **Open full dashboard →** | `rocm dash` | live (the escalation door) |

> **Honesty note.** Only **Optimize** is aspirational here — there is no
> `optimize`/tuning command in the codebase today (the bench tab *displays*
> externally-produced results; it doesn't run or tune). It renders dimmed with a
> `soon` badge per the "honesty over magic" rule, never as a live action.
> An earlier draft also listed *Generate an image*; it was removed because the
> shipped capability is ComfyUI **lifecycle management** (`rocm comfyui …`), not
> an in-launcher prompt→image flow.

---

## 2.1 Returning user — GPU already set up, a model running

The status strip up top is *live* (the `ccstatusline` move): GPU, utilization,
VRAM, temperature on line 1; serving state and throughput on line 2. Cyan `▸`
marks the focused row. Icons give each verb a glanceable identity. Sections are
separated by whitespace, not boxes — calmer and closer to the reference.

```
  rocm.ai                                                  v0.9 · ROCm 6.2  ▔▔▔▔
  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁

   GPU  Radeon 8060S (Strix Halo)  │  Util 34%  │  VRAM 18 / 128 GB  │  51°C
   ●   Serving  Qwen3-72B  on :8000  │  42 t/s  │  ✓ healthy

  What would you like to do?

    ▸  ◆  Serve a model          run another model on your GPU
       ⚙  Set up this system     install / update ROCm
       ⚕  Diagnose & fix         check GPU, driver & ROCm
       ◷  Chat                   talk to Qwen3-72B or an API model
       ⚡  Optimize a model   soon   planned — hyperloom     (dimmed)

       ▣  Open full dashboard  →    live instruments & every action

  ↑↓  move      Enter  select      d  dashboard      :  palette      q  quit
```

> `▔▔` / `▁▁` denote the subtle warm top-gradient from the reference, rendered
> with half-block glyphs in `accent_2`; AMD red is *not* used here (no danger).

- **Live status strip** = rocm's instrument identity at minimal scale. Util/VRAM
  are colored by the gradient (green→amber→red) the dash already uses
  (`ui/gradient.rs`). The `●` serving dot is green when healthy.
- **Icon column** gives each verb a stable glyph (`◆ serve · ⚙ setup · ⚕ doctor ·
  ◷ chat · ⚡ optimize · ▣ dash`), echoing the reference's icon menu.
- **Aspirational rows are dimmed with a `soon` badge** — *Optimize* (planned
  hyperloom skill) is shown but never presented as live.
- **Context-aware copy** — because a model is running, *Chat* names the live
  model (`Qwen3-72B`) instead of generic text.
- `:` opens the **command palette** (same overlay as the dash, §3.4 of
  surface-3) so a power user gets the full surface list without leaving the menu.
- No prompt/composer is shown — there is no chat session yet (ux-guidelines:71).

### 2.1b Greyed rows with an inline reason (from the reference)

When a verb isn't available yet, it stays visible but greyed, with the reason
inline — exactly the reference's `Configure Status Line (install first)`. This
teaches instead of hiding (rescues the blank-state problem F114/F126):

```
   GPU  Radeon 8060S (Strix Halo)  │  Util 2%  │  VRAM 0 / 128 GB  │  44°C
   ◌   No model running

  What would you like to do?

    ▸  ◆  Serve a model          run a model on your GPU
       ⚙  Set up this system     install / update ROCm
       ⚕  Diagnose & fix         check GPU, driver & ROCm
       ·  Chat                   add a provider or serve a model (greyed)
       ⚡  Optimize a model   soon   planned — hyperloom        (dimmed)
       ▣  Open full dashboard  →
```

- Two kinds of dimming: a **greyed** row names the unlock step in parens
  (*Chat — "add a provider…"*, available once you act); an **aspirational** row
  carries a `soon` badge (*Optimize*, not built yet).
- The status dot flips to `◌` (idle) and the strip honestly reads
  `No model running` — never a fake number.

---

## 2.2 First run — deterministic setup auto-triggers (fixes F104)

If `~/.rocm/config.json` shows setup never completed, the launcher opens **into**
the setup card before showing the menu. This is the single highest-leverage
discoverability fix in the audit — and it lands on the exact first-run audience.
Setup is deterministic, **not** an LLM (ux-guidelines:17): choose folder → review
→ install → success → continue.

```
            ╭──────────────────────────────────────────────────────────╮
            │  Welcome to rocm.ai                              Step 1/3  │
            │ ──────────────────────────────────────────────────────────│
            │                                                            │
            │   Let's get your AMD GPU ready for local AI.               │
            │   This installs ROCm into a folder you choose. ~3 minutes. │
            │                                                            │
            │   Detected:  AMD Radeon 8060S (Strix Halo) · 128 GB        │
            │                                                            │
            │   Install location                                         │
            │    ◂  C:\Users\you\.rocm           ▸   (recommended)       │
            │                                                            │
            │       Advanced options  ▾   channel · version · import     │
            │                                                            │
            │ ──────────────────────────────────────────────────────────│
            │   This will download files. Nothing runs without your OK.  │
            ╰──────────────────────────────────────────────────────────╯

              ◂ ▸  change folder      Enter  review install      Esc  later
```

- **Folder row is arrow-friendly** (`◂ ▸` cycles easy choices; Enter opens manual
  entry) — ux-guidelines:97-99.
- **Advanced** disclosure hides channel/format/version-pin/import
  (ux-guidelines:91, 164-166).
- The bottom strip is the plain-English approval contract (ux-guidelines:27):
  *"Nothing runs without your OK."*

Step 2 is the review+approval card; step 3 streams live install progress in a
foreground card that **owns keyboard focus** (ux-guidelines:64-65) and resolves
into a celebratory success card:

```
            ╭──────────────────────────────────────────────────────────╮
            │  ✓  ROCm is installed and your GPU is ready.               │
            │                                                            │
            │     ▸  Serve a model now                                   │
            │        Open full dashboard  →                             │
            ╰──────────────────────────────────────────────────────────╯
```

This ties success straight to the next action (ux-guidelines:50; strategy W2.3)
and persists `setup_completed` so it never re-triggers (F104's deferred
fast-follow).

---

## 2.3 An intent row expands in place — `Serve a model`

Selecting a row does **not** dump command output. It opens that row's guided
screen as an overlapping card (ux-guidelines:73-79). `Serve` prefills the serve
wizard; Enter starts it.

```
            ╭──────────────────────────────────────────────────────────╮
            │  Serve a model                                       Esc ◂ │
            │ ──────────────────────────────────────────────────────────│
            │   Choose a model that fits your 128 GB:                     │
            │                                                            │
            │    ▸  Qwen3-72B          ✓ fits · recommended · ~minutes   │
            │       Llama-3.3-70B      ✓ fits                            │
            │       DeepSeek-R1-32B    ✓ fits · fastest                  │
            │       Browse all models  →                                 │
            │                                                            │
            │   Engine:  Lemonade  (auto-selected for Strix Halo)        │
            │ ──────────────────────────────────────────────────────────│
            ╰──────────────────────────────────────────────────────────╯

              ↑↓  choose      Enter  start serving      a  advanced engine
```

- VRAM-fit is shown as a plain `✓ fits`, not a raw byte estimate (rescues the
  VRAM-attribution honesty concern F133/F162 at the layman layer).
- Engine selection is automatic with an `a advanced` escape — Lemonade steering
  per strategy §5.

---

## Acceptance notes

- Bare `rocm` opens this card, never the full dash, never raw `--help`.
- First run auto-opens setup; setup is deterministic and approval-gated; choice
  persists (F104).
- Rows are arrow-navigable; every primary action reachable in ≤2 keystrokes.
- No backend jargon in any first-view row; advanced detail is one disclosure away.
- `Open full dashboard →` and `d` both escalate to Surface 3.
