# Surface 1 — Single-command CLI (`rocm <command>`)

**What it is.** One verb in, one plain-English answer out. No TUI, no cursor, no
state. This is the scriptable / air-gapped / CI surface the 1-pager calls out
("local, secure, optional air-gapped path") and the surface ux-guidelines:107-108
says to keep for "scripting, automation, and debugging."

**What changes.** Not the mechanism — the *messaging*. The audit's biggest CLI
problem is **dead-end rejections**: platform-limited paths that refuse without a
next step (F002, F005, F040–F042, F275/F280/F288) and missing-auth errors with no
fix (F085). Every mock below ends in a runnable next step, in plain English, with
backend jargon held back.

Design rules for this surface:

- **Status line first, detail under it.** A layman can stop reading after line 1.
- **Color = meaning** (same tokens as the TUI): `✓` green ok, `⚠` amber caution,
  `✗` red error. Plain text still readable with color stripped (pipes/CI).
- **Every non-ok state names the exact command that fixes it.**
- **No wheel/venv/gfx jargon in the headline** — it lives under an indented
  detail or a `Details:` block (ux-guidelines:10-11, 164-166).

---

## 1.1 `rocm doctor` — the diagnostic spine

The PRD's Appendix-C promise: *"`rocm doctor` returns plain-language status with
fix commands inline."* Today the data is captured (OS, arch, gfx target, ROCm
family) but the audit flags the *interpretation* layer as the value-add.

### Healthy system (Linux / MI300X)

```
  rocm doctor

  ✓  Your system is ready for local AI.

     GPU            AMD Instinct MI300X  (gfx942) · 192 GB
     Driver         amdgpu 6.8 · ROCm 6.2 detected
     Python env     managed · /home/dev/.rocm/venv
     Engines        vLLM ✓   llama.cpp ✓   Lemonade ✓

     Next:  rocm serve qwen3-72b      start a model
            rocm dash                 open the dashboard
```

### Broken system (Strix Halo / Windows) — actionable, not a dead end

```
  rocm doctor

  ⚠  Almost ready — 1 thing needs attention.

     GPU            AMD Radeon 8060S  (Strix Halo · gfx1151) · 128 GB unified
     Driver         ✓  managed by Windows — nothing to install
     ROCm wheels    ✗  not installed for your GPU yet
     Engine         Lemonade  (recommended for this device)

     Fix this:
       rocm install            install the right ROCm for gfx1151  (guided)

     Then:
       rocm serve qwen         Lemonade will be selected automatically

     Details (advanced) ▾   gfx1151 · TheRock wheel channel · venv path
```

What this fixes:

- **F005** — instead of rejecting "DKMS driver install unsupported," it says
  *"managed by Windows — nothing to install"* (a `✓`, not an error).
- **F002 / F275 / F280 / F288** — platform-limited engines no longer dead-end;
  the device is steered to **Lemonade (F038)**, the friendliest implemented
  engine for Strix Halo.
- Backend labels (gfx, wheel channel, venv) collapse under an `advanced`
  disclosure, honoring ux-guidelines:164-166.

---

## 1.2 `rocm serve <model>` — single-shot serve

Mirrors the AAI **Demo 1** ("Serve Qwen3-72B on my AMD GPU"). One command brings
an OpenAI-compatible endpoint live; progress streams as plain lines.

```
  rocm serve qwen3-72b

  ●  Selecting engine……  vLLM  (best fit for MI300X · gfx942)
  ●  Checking VRAM……     ✓  72B fits in 192 GB
  ●  Pulling weights……   ✓  cached
  ●  Starting endpoint……  ✓  http://127.0.0.1:8000  (loopback only)

  ✓  Qwen3-72B is serving.

     Chat now:   rocm chat                 talk to it in your terminal
     Inspect:    rocm dash                 watch it live in the dashboard
     Stop:       rocm serve --stop qwen3-72b
```

Missing-auth variant (fixes **F085** — recoverable error with the exact fix):

```
  rocm chat

  ⚠  No chat provider is set up yet.

     Pick one:
       rocm config provider openai      use an OpenAI key
       rocm config provider claude      use an Anthropic key
       rocm serve qwen && rocm chat     or chat with a local model — no key needed

     Where to get a key:  https://platform.openai.com/api-keys
```

---

## 1.3 Discoverability bridge — bare `rocm` is mentioned, never forced

Running `rocm` with no command does **not** dump help text (ux-guidelines:108,
124-125). It opens the minimal launcher (Surface 2). But every single-command
output points back to richer surfaces, so a power user always knows the door
exists:

```
     Tip:  run  rocm        for a guided menu
           run  rocm dash   for the live dashboard
```

`rocm --help` still prints the full command reference for those who ask for it
explicitly — that is the one place raw command syntax belongs
(ux-guidelines:125).

---

## Acceptance notes

- Output is plain-text-clean when piped (color is additive, never required).
- No headline contains the words *wheel*, *venv*, *gfx*, *adapter*, *tarball*
  (they may appear under an explicit `Details (advanced)` block).
- Every `⚠` / `✗` line is immediately followed by a runnable `Fix this:` block.
- The Windows/Strix-Halo path never ends in a rejection without a next step.
