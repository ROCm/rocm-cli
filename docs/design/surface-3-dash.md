# Surface 3 — `rocm dash` full control room

**What it is.** The live instrument panel. A new **Home landing** opens by
default; **outlined folder tabs** switch between Home, **Action** (optimized for
doing), **Observe** (optimized for watching), and **Chat**. A command palette
(`:`) and a help-as-list (`?`) overlay every screen.

**What changes vs. today.** Today the dash is five flat, inline-highlighted tabs
(Overview/Hardware/Instances/Bench/Chat, `ui/tabs/mod.rs:20-26`) over a 60/40
text split (`ui/tabs/overview.rs`), and the operational power (serve, install,
engines, doctor, providers) lives in overlays with **no tab home**. We:

1. add a **Home landing** with bento tiles (glance + next action),
2. give tabs **real outlines** (the explicit ask),
3. collapse the 5 flat tabs into **Action** + **Observe**, and
4. make the command palette the universal launcher.

```
   5 flat tabs              →     1 landing + 3 intent tabs
   Overview ┐                     ┌ Home    (glance + next action)
   Hardware │                     │ Action  (serve/install/engines/doctor/…)
   Instances│  ── folds into ──►  │ Observe (GPU/instances/bench/telemetry)
   Bench    │                     │ Chat    (agent + /plan chip)
   Chat    ─┘                     └
```

---

## 3.1 The outlined tab bar

Active tab: `accent` (cyan) outline, leading `●`, **bottom edge open** into the
panel below. Inactive tabs: `muted` closed boxes resting on the panel rim. The
digit hint (`1`–`4`) stays for muscle memory; the palette/help chips live on the
right.

```
   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐
   │ ● Home   │  │  Action  │  │ Observe  │  │  Chat  │        Esc menu  t theme  ? help
 ┌─┘  1       └──┴── 2 ──────┴──┴── 3 ─────┴──┴── 4 ───┴──────────────────────────┐
 │                                                                                 │
 │   …active tab's panel…                                                          │
 └─────────────────────────────────────────────────────────────────────────────┘
```

Reuses `compute_chip_layout` (`ui/tabs/mod.rs`) for hit-testing so mouse clicks
keep working; only the rendering gains box-drawing.

---

## 3.2 Home — the new landing (bento tiles, glance + next action)

The default view. A **bento grid** (design-quality bar: hierarchy, depth,
grid-breaking composition) — *not* the old uniform text split. One hero tile, a
context-aware next-action tile, glanceable instruments, services, and a
notifications strip. Every tile is arrow-navigable; Enter opens the relevant
Action/Observe surface. This is the "friendly control room, not a log viewer"
mandate (ux-guidelines:48-50).

```
 ┌─ rocm.ai ────────────────────────────────── connected · rocm daemon 0.9 ──┐
 │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐                          │
 │ │ ● Home   │ │  Action  │ │ Observe  │ │  Chat  │      Esc menu  t theme  ? help │
 ├─┘          └─┴──────────┴─┴──────────┴─┴────────┴──────────────────────────┤
 │                                                                            │
 │  ╔══ AMD Radeon 8060S · Strix Halo ═══════════╗  ╭─ Next step ───────────╮ │
 │  ║                                             ║  │                       │ │
 │  ║   GPU UTIL   ▰▰▰▰▰▰▱▱▱▱  62%                ║  │  ● Qwen3-72B is live  │ │
 │  ║   ⎓ 24.1 tokens / watt        ▁▂▃▅▆▇▆▅▃▂    ║  │    on :8000           │ │
 │  ║                                             ║  │                       │ │
 │  ║   VRAM  ▰▰▰▰▰▰▰▰▱▱  82 / 128 GB             ║  │   ▸ Open Chat   →     │ │
 │  ║   TEMP  51°C        POWER  86 W             ║  │   View in Observe →   │ │
 │  ╚═════════════════════════════════════════════╝  ╰───────────────────────╯ │
 │                                                                            │
 │  ╭─ Running ─ 1 ──────────────╮  ╭─ Health ───────╮  ╭─ Updates ─────────╮ │
 │  │ ● Qwen3-72B   vLLM   :8000 │  │ ✓ GPU          │  │ ⇲ ROCm 6.3 ready  │ │
 │  │   42 t/s · TP 1 · 1 GPU    │  │ ✓ Driver       │  │   review & approve │ │
 │  │                            │  │ ✓ ROCm 6.2     │  │                   │ │
 │  │ ▸ Serve another  →         │  │ ▸ Run doctor → │  │ ▸ View update  →  │ │
 │  ╰────────────────────────────╯  ╰────────────────╯  ╰───────────────────╯ │
 │                                                                            │
 └────────────────────────────────────────────────────────────────────────────┘
   ↑↓←→ move tile   Enter open   1–4 tabs   w serve   t  theme   ?  help   q quit
```

Tile inventory (each maps to existing features, now *visible*):

| Tile | Content | Surfaces / features |
|---|---|---|
| **Hero GPU** | model, util gauge, **tokens-per-watt** hero, VRAM/temp/power | promotes F134 (buried) + F255/F257 gauges (strategy W2.1) |
| **Next step** | context-aware single best action, pre-selected | ties success → next action (ux-guidelines:50) |
| **Running** | live services only, count + throughput | F196 `/services`, living-only per ux-guidelines:128-130 |
| **Health** | GPU/driver/ROCm green checks, `Run doctor →` | F184 doctor, rescued onto the landing |
| **Updates** | approval-gated ROCm upgrade proposal | F-update watcher; AAI Demo 3 "upgrade proposals in the TUI" |

Empty-state Home (no model, fresh setup) swaps the hero's right column for a big
`▸ Serve your first model →` card (fixes the blank-pane finding F114/F126):

```
 │  ╔══ AMD Radeon 8060S · ready ════════════════╗  ╭─ Get started ─────────╮ │
 │  ║   GPU UTIL  ▱▱▱▱▱▱▱▱▱▱   2%   ◌ idle        ║  │  No model running yet │ │
 │  ║   VRAM  ▱▱▱▱▱▱▱▱▱▱  0 / 128 GB              ║  │                       │ │
 │  ╚═════════════════════════════════════════════╝  │  ▸ Serve a model  →   │ │
```

---

## 3.3 Action tab — optimized for *doing*

Every guided, mutating workflow as a **visible tile**, so the 79 hidden gems get
a named home instead of a memorized slash string. Left column = action tiles
(arrow-navigable); right column = a plain-English detail/preview pane for the
focused tile (ux-guidelines:55, "Home dashboard should use real arrow-key action
rows plus a plain-English detail pane").

```
 ├─┐          ┌─┘ ● Action └─┐          ┌──────────┐ ┌────────┐────────────────┤
 │ └──────────┘              └──────────┴──────────┴─┴────────┴                 │
 │                                                                            │
 │  Actions                              │  ◆ Serve a model                    │
 │                                       │  ─────────────────────────────────  │
 │   ▸ ◆  Serve a model                  │  Bring a model up as a local        │
 │     ⚙  Set up / Install ROCm          │  OpenAI-compatible endpoint.        │
 │     ⌬  Engines                        │                                     │
 │     ⚕  Diagnose & fix  (doctor)       │  Recommended for your GPU:          │
 │     ⇲  Check for updates              │    Qwen3-72B   ✓ fits · ~minutes    │
 │     ⮌  Manage providers & keys        │    Engine: Lemonade (auto)          │
 │     ⚡  Optimize a model    soon       │                                     │
 │     ⊘  Uninstall            (greyed)  │  ▸ Start serving        Enter       │
 │                                       │    Advanced engine opts   a         │
 │  Background helpers ▾                  │                                     │
 │  soon = planned, not yet built        │  Mutating actions ask before they   │
 │                                       │  run. Default mode: ask.            │
 └────────────────────────────────────────────────────────────────────────────┘
   ↑↓ choose   Enter start   a advanced   t  theme   ?  help   Esc Home
```

Notes:

- **Each tile prefills its guided screen** (ux-guidelines:75-79). `Enter` on
  *Serve* opens the serve wizard with Start reachable by Enter — never a typed
  command. Typed `/serve qwen` from the palette lands on the same screen.
- **Rescues in one surface:** F195 serve, F193 install, F194 engines, F184
  doctor, update, F208/F231/F166/F167 providers, F199 uninstall. The whole
  F193–F208 cluster, visible.
- **Optimize is aspirational** — shown dimmed with a `soon` badge (planned
  hyperloom skill); there is no optimize/tuning command today. *Generate an
  image* was dropped: the shipped `rocm comfyui` is server lifecycle management,
  not an in-dash prompt→image action.
- **"Background helpers"** (not "Automations" — jargon fix F056) is an `Advanced`
  disclosure so first view stays calm (ux-guidelines:91).
- **Uninstall greyed unless applicable**, and dry-run-by-default when opened
  (F199); destructive confirm uses AMD red.
- The detail pane restates the approval contract in plain English (F172/F173).

### Approval modal (mutating action) — overlaps, owns focus

When a tile triggers a mutation, the approval card overlaps and owns the keyboard
(ux-guidelines:64-65; F203 re-enters approval even on a proposal):

```
        ╭─ Approve install ────────────────────────────────────────╮
        │  rocm will install ROCm 6.2 wheels into                   │
        │    C:\Users\you\.rocm\venv                                │
        │                                                           │
        │  This downloads ~2.4 GB and creates a managed Python env. │
        │  Nothing else on your system is changed.                  │
        │                                                           │
        │     ▸ Approve   (Enter)        Reject   (Esc)             │
        ╰───────────────────────────────────────────────────────────╯
```

---

## 3.4 Command palette (`:`) — the universal launcher (rescues F263)

One key (`:`) opens a fuzzy-filterable list of **navigable destinations**, each a
plain-English label + one-line description. This is the TUI-legal command
cheat-sheet: it lists places to *select*, not strings to *type*
(ux-guidelines:80-83). It overlaps any screen and returns you exactly where you
were.

```
        ╭─ Go to… ──────────────────────────────────────────────────╮
        │  : serv▌                                                   │
        │ ───────────────────────────────────────────────────────── │
        │   ▸ ◆  Serve a model            start a local endpoint     │
        │     ⌬  Engines                  install / pick an engine   │
        │     ⚕  Services                 stop or inspect running    │
        │ ───────────────────────────────────────────────────────── │
        │   recent:  Doctor · Serve Qwen3-72B · Theme            │
        ╰───────────────────────────────────────────────────────────╯
          type to filter   ↑↓ move   Enter open   Esc close
```

Covers the 25 navigable surfaces enumerated at ux-guidelines:140-144. Reuses the
`Modal`/popup-frame primitive (`ui/modal.rs`). This single mechanism rescues
F196/F199/F200/F202/F206/F207/F208/F198/F193/F194/F195 at once.

---

## 3.5 Observe tab — optimized for *watching*

The Overview + Hardware + Instances + Bench content, recomposed as a **glanceable
instrument board** rather than four flat tabs of tables. Top band = live
instruments (the existing sparkline F255 / gradient gauge F257 / KV-cache heatmap
F256, promoted to headline size); below = the running-instances table and recent
bench rollup, each with item actions exposed as rows.

```
 ├─┐          ┌──────────┐ ┌─┘ ● Observe └─┐ ┌────────┐──────────────────────────┤
 │ └──────────┘          └─┘               └─┘        └                          │
 │                                                                            │
 │  ┌─ GPU 0 · MI300X · gfx942 ───────────────────────────────────────────┐  │
 │  │ UTIL  ▁▂▃▅▆▇▇▆▅▆▇▇▆▅▃▂▃▅▆▇  62%      VRAM ▰▰▰▰▰▰▰▰▱▱ 82/128 GB        │  │
 │  │ TEMP  51°C   POWER 86 W   CLK 1.7GHz   ⎓ 24.1 tok/W                    │  │
 │  └────────────────────────────────────────────────────────────────────────┘  │
 │                                                                            │
 │  ┌─ Instances · 1 ──────────────────────╮  ┌─ Bench · last run ─────────┐  │
 │  │ name        model       port   t/s    │  │ cell    model    gTPS  ✓/✗ │  │
 │  │ ● qwen-72b  Qwen3-72B   8000   42  ●  │  │ tg128   Qwen3-72B 41.8  ✓  │  │
 │  │                                       │  │ pp512   Qwen3-72B  3.1k ✓  │  │
 │  │ ▸ Stop   Restart   Logs   Detail      │  │ ▸ Run benchmark  →         │  │
 │  ╰───────────────────────────────────────╯  └────────────────────────────┘  │
 │                                                                            │
 │  legend:  ● running   ◌ stopped   ↺ rollback        attribution: per-process │
 └────────────────────────────────────────────────────────────────────────────┘
   ↑↓ select   Enter detail   s stop   l logs   F5 refresh   t  theme   ? help
```

Notes:

- **Tokens-per-watt (`⎓`) is promoted** to a headline instrument (F134, buried
  today; strategy W2.1).
- **Item actions are rows, not hidden keys** — Stop / Restart / Logs / Detail
  (ux-guidelines:82-83), rescuing F196 service actions.
- **Honesty:** an **attribution badge** states `per-process` vs `estimated`
  (fixes F133/F245 — device-summed VRAM never masquerades as per-model); the
  glyph legend line is on-screen (F217); unpopulated fields show `N/A`, not `0`
  (F124/F132); a stub collector renders `metrics not yet available` rather than
  blank (F126/F114).
- **Demo-data banner** (amber, persistent) appears here and on Home whenever the
  live daemon is unavailable — the must-fix from the audit (F151):

```
 │  ⚠  Demo data — not your live GPU.  Live telemetry isn't available on this   │
 │     machine yet.  Showing a recorded session so you can explore.             │
```

---

## 3.6 Chat tab — the agent, with `/plan` made visible (F207)

Unchanged in spirit (the strongest area, polish 4.07) but the most novice-
friendly capability — plain-English planning — becomes a **visible action chip**
instead of a memorized `/plan` string (REPORT finding #3). Typing `/` opens an
arrow-navigable completion menu with plain-English descriptions; Enter prefills
the guided screen, never dumps a report (strategy §3.5).

```
 ├─┐          ┌──────────┐ ┌──────────┐ ┌─┘ ● Chat └─┐──────────────────────────┤
 │ └──────────┘          └─┘          └─┘            └                          │
 │                                                                            │
 │   you   serve qwen and tell me when it's ready                              │
 │                                                                            │
 │   ●     I'll start Qwen3-72B with vLLM. This will download weights and      │
 │         bring up an endpoint on :8000. Approve?                             │
 │         ╭─ proposed ──────────────────────────────╮                        │
 │         │ ▸ Approve   Reject   Edit                │                        │
 │         ╰─────────────────────────────────────────╯                        │
 │                                                                            │
 │ ──────────────────────────────────────────────────────────────────────────│
 │  [ ✦ Plan this ]  [ ◆ Serve ]  [ ⚕ Doctor ]        provider: claude  ●     │
 │  > ▌                                                                        │
 └────────────────────────────────────────────────────────────────────────────┘
   Enter send   / commands   ✦ plan   Shift+Tab approve   t  theme   ? help
```

- **`✦ Plan this`** action chip surfaces F207 as a button (strategy §3.5).
- Mutating tool calls render as **review cards** with Approve/Reject/Edit
  (F172/F173/F203) — the double-gate the audit praises.
- The composer appears **only here**, where a session exists (ux-guidelines:71).
- Provider shown with a live dot; switching reverts on failure (F208).

---

## 3.7 Help-as-list (`?`) — contextual, navigable (converts F182)

`?` overlaps the current screen with *this tab's* actions as a navigable list —
each row launches its surface — instead of a static 30-command catalog. Contextual
per ux-guidelines:126 (must not open during setup/install/approval/chat-input).

```
        ╭─ What can I do here?  (Observe) ──────────────────────────╮
        │   ▸  Stop a running model            s                     │
        │      View logs                       l                     │
        │      Open instance detail            Enter                 │
        │      Run a benchmark                 →                     │
        │      Refresh now                     F5                    │
        │ ───────────────────────────────────────────────────────── │
        │      Go to any surface…              :                     │
        ╰───────────────────────────────────────────────────────────╯
          ↑↓ move   Enter do it   Esc back to where you were
```

> **`?` vs. Esc → Help.** `?` is the *contextual* per-tab action list above.
> The global **keyboard reference** lives under the Esc menu (§3.8) — a static
> two-column cheat-sheet of every shortcut.

---

## 3.8 Global menu (`Esc`) — logo, Options, Help, Quit

`Esc` opens a btop-style **main menu** over a grey-dimmed backdrop: a big
gradient block "ROCM" logo and a vertical list. `↓` cycles; `Enter` selects.

```
        ╔══════════════════════════════════════════════════════════╗
        ║        ██████   █████   ██████  ██   ██                   ║
        ║        ██   ██ ██   ██ ██       ███ ███   (gradient)      ║
        ║        ██████  ██   ██ ██       ██ █ ██                   ║
        ║        ██   ██ ██   ██ ██       ██   ██                   ║
        ║        ██   ██  █████   ██████  ██   ██                   ║
        ║              AMD ROCm · local AI control room             ║
        ║                                                          ║
        ║                    ▸ Options                              ║
        ║                      Help                                 ║
        ║                      Quit                                 ║
        ║            ↑↓ move   Enter select   Esc close             ║
        ╚══════════════════════════════════════════════════════════╝
```

### Options — tabbed settings (General · CPU · GPU · Engines)

Outlined tabs (reusing the §3.1 tab system). Rows are toggles (`● / ○`), cycles
(`◂ ▸`), or actions (`→`). A starter set, all easy to implement:

- **General** — Theme · Start screen (Home / Minimal) · Refresh interval ·
  Confirm before changes (Ask / Full access) · Store telemetry on this PC only ·
  Soft bell on long-job complete · Reduce motion · Show file locations.
- **CPU** — Show per-core bars · History window (30/60/120s) · Aggregate sparkline.
- **GPU** — VRAM units (GB/GiB/%) · Temperature unit (°C/°F) · Per-GPU sparklines ·
  Attribution badge · Highlight GPU over N%.
- **Engines** — Default engine (Lemonade/vLLM/llama.cpp) · Auto-select for
  hardware · Loopback-only serve · Warm-up notice.

### Help — two-column keyboard reference

A static cheat-sheet grouped by **Navigate · Overlays · Actions · Chat · Global**,
keys rendered as chips. This is the global counterpart to the contextual `?`
(§3.7).

Rendered: `dash-menu.svg`, `dash-options.svg`, `dash-help.svg` in
[mockups/](mockups/README.md).

---

## Acceptance notes

- Home is the default tab; tabs render as outlined folder tabs with the active
  one opening into the panel.
- Action exposes every guided/mutating workflow as a visible, arrow-navigable
  tile with a plain-English detail pane; mutations are approval-gated.
- Observe promotes live instruments (incl. tokens-per-watt), exposes item actions
  as rows, labels demo data, shows attribution confidence, and never renders a
  silent blank.
- `:` palette and `?` help-as-list overlay every screen and return focus to the
  prior row.
- Every advertised surface is reachable in ≤2 keystrokes from any tab (strategy
  success metric).
