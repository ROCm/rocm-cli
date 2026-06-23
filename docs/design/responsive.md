# Responsive dash — two sizes, two layouts

A TUI's "screen size" is really **columns × rows**, which scales with both the
physical display and the font. The question "how would we design for a 13″ laptop
vs. a 27″ external" maps cleanly onto two terminal-size classes:

| Class | Physical | Typical maximized terminal | Layout |
|---|---|---|---|
| **Compact** | 13–14″ laptop | ~120–160 cols × 28–40 rows | the current views |
| **Wide** | 15″+ laptop / 24–27″+ external | ~180–280 cols × 45–75 rows | the new views here |

The dash picks a layout from the live terminal size (ratatui already reports it).
Suggested breakpoint: **Wide when `cols ≥ 180` AND `rows ≥ 45`**, else Compact.
(One threshold per axis; a half-met terminal stays Compact so nothing clips.)

---

## The core inversion: sequential → simultaneous

Compact and Wide are not "the same screen, scaled." They invert the interaction
model, because the binding constraint changes:

| | **Compact (laptop)** | **Wide (external)** |
|---|---|---|
| Binding constraint | space | attention |
| Tabs | switch the **whole** screen | drive only the **center**; rails persist |
| Telemetry | one tab (Observe) | a **persistent left rail**, always on |
| Assistant | one tab (Chat) | a **persistent right dock**, always on |
| Master ↔ detail | one at a time (Enter / overlay) | **both visible** side by side |
| Overlays (palette, help, approval) | take the screen | float as cards over a still-visible dash |
| GPUs | one headline GPU | a **GPU wall** (all of them) |
| History depth | last ~30 samples | last ~60–120 samples, taller traces |

The compact layout hides to fit; the wide layout **stops hiding**. On a 27″ you
should never have to leave your live GPU telemetry to read the assistant, or tab
away from the model list to see the model you selected.

### What each extra dimension buys

- **More columns →** put panes *beside* each other instead of behind tabs. This
  is the triptych below: telemetry ▏ workspace ▏ assistant.
- **More rows →** depth: every GPU on a node as its own instrument card, longer
  sparkline history, more instance/bench rows, taller gauges, inline detail.

---

## The Wide layout: a triptych "command center"

```
 ┌ rocm.ai ───────────────────────── connected · node-mi300x-8 ····· : palette  ? help ┐
 │ ┌─ GPUs · 8 ──────┐ ┌─ Home ┐┌Action┐┌Observe┐┌Chat┐         ┌─ Assistant ─────────┐ │
 │ │ GPU0  ▰▰▰▰▱ 62% │ ├───────┘└──────┴┴───────┴┴────┴───────┐ │ ● next: open Chat   │ │
 │ │ util ▂▃▅▆▇▆▅    │ │                                       │ │ ─────────────────── │ │
 │ │ VRAM ▰▰▰▰▰▱ 71% │ │   …tab-driven workspace…              │ │ you  serve qwen     │ │
 │ │ GPU1  ▰▰▰▱▱ 48% │ │                                       │ │ ●    starting…      │ │
 │ │ util ▁▂▃▅▆▅▃    │ │   (Home tiles / Action master-detail  │ │                     │ │
 │ │ VRAM ▰▰▰▱▱ 55% │ │    / Observe table+detail)            │ │ [✦ Plan] [◆ Serve]  │ │
 │ │ … GPU2..7 …     │ │                                       │ │ > ▌                 │ │
 │ └─────────────────┘ └───────────────────────────────────────┘ └─────────────────────┘ │
 └──────────────────────────────────────────────────────────────────────────────────────┘
   left rail: PERSISTENT telemetry   center: tab-driven   right dock: PERSISTENT assistant
```

**Invariants of the triptych:**

1. **Left rail = telemetry, always.** The GPU instrument wall is the product's
   identity (the 1-pager's "instrument panel for your GPU"); on a wide screen it
   never gets hidden behind a tab.
2. **Center = the workspace, tab-driven.** The outlined tabs still exist, but on
   Wide they only swap the *center* column — not the whole screen. Inside the
   center, master and detail sit side by side (no Enter-to-reveal).
3. **Right dock = the assistant by default — but contextual.** Natural-language
   help is "always available" (1-pager), so it lives beside the work, not behind
   a Chat tab. The dock can also adapt to the focused task: a **live logs
   stream** while inspecting a service on Observe, or a **context rail** (what
   the agent can see — services, GPU state, recent tools, skills) when Chat owns
   the center. Telemetry-left / workspace-center / dock-right is the invariant;
   what fills the dock follows the task.

This is the honest answer to "do it differently": Compact is a **stack you page
through**; Wide is a **cockpit you scan**.

---

## Per-surface reflow

| Surface | Compact | Wide |
|---|---|---|
| **Home** | bento tiles, one GPU hero | GPU wall (left) + status bento (center) + assistant (right); wide tokens/watt hero strip |
| **Observe** | tabs of tables; Enter for detail | GPU wall (all GPUs) + instances **master-detail** in one view + bench, all visible |
| **Action** | list → Enter → detail pane | list **and** detail side by side; the serve wizard shows model list + VRAM-fit + engine + advanced at once |
| **Chat** | its own tab | the persistent right dock (and can expand to center on the Chat tab with a context rail) |
| **Palette / Help / Approval** | own the screen | float as cards over the still-live triptych |

---

## Rendered wide views

See **[mockups/](mockups/README.md)** — the `*-wide` scenes are rendered at
220 × 54 (a believable maximized 27″ terminal); real terminals are often larger,
which simply adds more GPU cards and more history, never more chrome.

- `dash-wide-home.svg` — the flagship triptych
- `dash-wide-observe.svg` — 8-GPU wall + instances master-detail
- `dash-wide-action.svg` — list + live serve wizard side by side
- `dash-wide-chat.svg` — conversation in center + agent **context rail**
- `dash-wide-observe-logs.svg` — the contextual dock as a **live logs stream**

Regenerate everything (compact + wide) with the one command in
[mockups/README.md](mockups/README.md).
