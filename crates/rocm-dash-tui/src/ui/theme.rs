//! Semantic theme.
//!
//! Two construction paths:
//! 1. Bespoke constructors (`default_dark`, `default_light`) that don't fit a
//!    16-color ANSI mapping cleanly.
//! 2. `from_palette(&Palette16)` for canonical terminal palettes. Hex values
//!    sourced from each project's published palette. Theme catalogue
//!    inspired by [ansicolor.com](https://ansicolor.com/).
//!
//! Pattern borrowed from ctux (see `../../../wiki/sources/ctux.md`).

use ratatui::style::{Color, Modifier, Style};

#[derive(Debug, Clone, Copy)]
pub struct Theme {
    pub bg: Color,
    pub surface: Color,
    pub surface_2: Color,
    pub fg: Color,
    pub muted: Color,
    pub accent: Color,
    pub accent_2: Color,
    pub ok: Color,
    pub warn: Color,
    pub err: Color,
    pub border: Color,
}

#[derive(Debug, Clone, Copy)]
pub enum StatusTone {
    Neutral,
    Muted,
    Info,
    Accent,
    Warning,
    Success,
    Error,
    Alert,
}

/// Canonical 16-color ANSI palette plus default background/foreground.
/// Used as the source-of-truth shape for imported palettes; `Theme::from_palette`
/// reduces it to our 11-slot semantic palette.
#[derive(Debug, Clone, Copy)]
pub struct Palette16 {
    pub bg: Color,
    pub fg: Color,
    pub black: Color,
    pub red: Color,
    pub green: Color,
    pub yellow: Color,
    pub blue: Color,
    pub magenta: Color,
    pub cyan: Color,
    pub white: Color,
    pub br_black: Color,
    pub br_red: Color,
    pub br_green: Color,
    pub br_yellow: Color,
    pub br_blue: Color,
    pub br_magenta: Color,
    pub br_cyan: Color,
    pub br_white: Color,
}

/// Registry entry. (name, constructor). First entry is the default.
type ThemeCtor = fn() -> Theme;

const REGISTRY: &[(&str, ThemeCtor)] = &[
    ("default-dark", Theme::default_dark),
    ("default-light", Theme::default_light),
    ("solarized-dark", Theme::solarized_dark),
    ("tokyo-night", Theme::tokyo_night),
    ("tokyo-night-light", Theme::tokyo_night_light),
    ("dracula", Theme::dracula),
    ("gruvbox-dark", Theme::gruvbox_dark),
    ("nord", Theme::nord),
    ("one-dark", Theme::one_dark),
    ("monokai", Theme::monokai),
    ("catppuccin-mocha", Theme::catppuccin_mocha),
    ("catppuccin-latte", Theme::catppuccin_latte),
    ("ayu-dark", Theme::ayu_dark),
    ("ayu-light", Theme::ayu_light),
    ("github-dark", Theme::github_dark),
];

/// All registered theme names, in cycle order.
pub fn theme_names() -> Vec<&'static str> {
    REGISTRY.iter().map(|(n, _)| *n).collect()
}

impl Theme {
    // --- bespoke palettes (don't fit a clean ANSI mapping) -----------------

    pub const fn default_dark() -> Self {
        Self {
            bg: rgb(0x13, 0x14, 0x16),
            surface: rgb(0x1d, 0x1f, 0x21),
            surface_2: rgb(0x27, 0x27, 0x2a),
            fg: rgb(0xea, 0xeb, 0xec),
            muted: rgb(0xb4, 0xb9, 0xbc),
            accent: rgb(0x00, 0xc2, 0xde),
            accent_2: rgb(0x00, 0x7d, 0xb8),
            ok: rgb(0x1a, 0xa0, 0x1a),
            warn: rgb(0xf5, 0x9e, 0x0b),
            err: rgb(0xed, 0x1c, 0x24),
            border: rgb(0x65, 0x6b, 0x72),
        }
    }

    pub const fn default_light() -> Self {
        Self {
            bg: rgb(0xfa, 0xfa, 0xfb),
            surface: rgb(0xf2, 0xf3, 0xf5),
            surface_2: rgb(0xe4, 0xe7, 0xea),
            fg: rgb(0x17, 0x1a, 0x1d),
            muted: rgb(0x55, 0x5a, 0x60),
            accent: rgb(0x00, 0x76, 0x9b),
            accent_2: rgb(0x00, 0x4f, 0x71),
            ok: rgb(0x14, 0x80, 0x14),
            warn: rgb(0xb2, 0x6b, 0x00),
            err: rgb(0xc4, 0x18, 0x1f),
            border: rgb(0xa8, 0xae, 0xb4),
        }
    }

    pub const fn solarized_dark() -> Self {
        Self::from_palette(&palettes::SOLARIZED_DARK)
    }

    // --- imported via Palette16 -------------------------------------------

    pub const fn tokyo_night() -> Self {
        Self::from_palette(&palettes::TOKYO_NIGHT)
    }
    pub const fn tokyo_night_light() -> Self {
        Self::from_palette(&palettes::TOKYO_NIGHT_LIGHT)
    }
    pub const fn dracula() -> Self {
        Self::from_palette(&palettes::DRACULA)
    }
    pub const fn gruvbox_dark() -> Self {
        Self::from_palette(&palettes::GRUVBOX_DARK)
    }
    pub const fn nord() -> Self {
        Self::from_palette(&palettes::NORD)
    }
    pub const fn one_dark() -> Self {
        Self::from_palette(&palettes::ONE_DARK)
    }
    pub const fn monokai() -> Self {
        Self::from_palette(&palettes::MONOKAI)
    }
    pub const fn catppuccin_mocha() -> Self {
        Self::from_palette(&palettes::CATPPUCCIN_MOCHA)
    }
    pub const fn catppuccin_latte() -> Self {
        Self::from_palette(&palettes::CATPPUCCIN_LATTE)
    }
    pub const fn ayu_dark() -> Self {
        Self::from_palette(&palettes::AYU_DARK)
    }
    pub const fn ayu_light() -> Self {
        Self::from_palette(&palettes::AYU_LIGHT)
    }
    pub const fn github_dark() -> Self {
        Self::from_palette(&palettes::GITHUB_DARK)
    }

    /// Reduce a 16-color ANSI palette to our semantic theme.
    ///
    /// Mapping (held constant for every imported palette so themes feel coherent):
    /// - `bg/fg` — palette default background / foreground
    /// - `surface` — `bg` (we differentiate via borders, not fills)
    /// - `surface_2` — `br_black` (subtle non-bg backdrop)
    /// - `muted` — `br_black`
    /// - `accent` — `br_cyan` (the chip / sparkline color)
    /// - `accent_2` — `blue`
    /// - `ok` — `green` (or `br_green` if green looks washed)
    /// - `warn` — `yellow`
    /// - `err` — `red`
    /// - `border` — `br_black`
    pub const fn from_palette(p: &Palette16) -> Self {
        Self {
            bg: p.bg,
            surface: p.bg,
            surface_2: p.br_black,
            fg: p.fg,
            muted: p.br_black,
            accent: p.br_cyan,
            accent_2: p.blue,
            ok: p.green,
            warn: p.yellow,
            err: p.red,
            border: p.br_black,
        }
    }

    /// Resolve a theme by name. Unknown names fall back to `default_dark`
    /// with a tracing warning.
    pub fn from_name(name: &str) -> Self {
        for (n, ctor) in REGISTRY {
            if *n == name {
                return ctor();
            }
        }
        tracing::warn!(
            requested = name,
            fallback = "default-dark",
            available = ?theme_names(),
            "unknown theme; using default-dark"
        );
        Self::default_dark()
    }

    /// Cycle through the registry — wraps at the end.
    pub fn next_name(current: &str) -> &'static str {
        let idx = REGISTRY
            .iter()
            .position(|(n, _)| *n == current)
            .unwrap_or(0);
        REGISTRY[(idx + 1) % REGISTRY.len()].0
    }

    pub fn border_style(&self) -> Style {
        Style::default().fg(self.border)
    }

    pub fn title_style(&self) -> Style {
        Style::default().fg(self.muted).add_modifier(Modifier::BOLD)
    }

    pub const fn tone_color(&self, tone: StatusTone) -> Color {
        match tone {
            StatusTone::Neutral => self.fg,
            StatusTone::Muted => self.muted,
            StatusTone::Info => self.accent_2,
            StatusTone::Accent => self.accent,
            StatusTone::Warning => self.warn,
            StatusTone::Success => self.ok,
            StatusTone::Error | StatusTone::Alert => self.err,
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::default_dark()
    }
}

const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::Rgb(r, g, b)
}

/// 16-color ANSI palettes. Hex values from each project's canonical source;
/// the catalogue itself is inspired by <https://ansicolor.com/>.
mod palettes {
    use super::{Palette16, rgb};

    /// Solarized Dark — Ethan Schoonover, <https://ethanschoonover.com/solarized/>.
    pub const SOLARIZED_DARK: Palette16 = Palette16 {
        bg: rgb(0x00, 0x2b, 0x36),
        fg: rgb(0xee, 0xe8, 0xd5),
        black: rgb(0x07, 0x36, 0x42),
        red: rgb(0xdc, 0x32, 0x2f),
        green: rgb(0x85, 0x99, 0x00),
        yellow: rgb(0xb5, 0x89, 0x00),
        blue: rgb(0x26, 0x8b, 0xd2),
        magenta: rgb(0xd3, 0x36, 0x82),
        cyan: rgb(0x2a, 0xa1, 0x98),
        white: rgb(0xee, 0xe8, 0xd5),
        br_black: rgb(0x58, 0x6e, 0x75),
        br_red: rgb(0xcb, 0x4b, 0x16),
        br_green: rgb(0x58, 0x6e, 0x75),
        br_yellow: rgb(0x65, 0x7b, 0x83),
        br_blue: rgb(0x83, 0x94, 0x96),
        br_magenta: rgb(0x6c, 0x71, 0xc4),
        br_cyan: rgb(0x93, 0xa1, 0xa1),
        br_white: rgb(0xfd, 0xf6, 0xe3),
    };

    /// Tokyo Night — enkia, <https://github.com/enkia/tokyo-night-vscode-theme>.
    pub const TOKYO_NIGHT: Palette16 = Palette16 {
        bg: rgb(0x1a, 0x1b, 0x26),
        fg: rgb(0xc0, 0xca, 0xf5),
        black: rgb(0x15, 0x16, 0x1e),
        red: rgb(0xf7, 0x76, 0x8e),
        green: rgb(0x9e, 0xce, 0x6a),
        yellow: rgb(0xe0, 0xaf, 0x68),
        blue: rgb(0x7a, 0xa2, 0xf7),
        magenta: rgb(0xbb, 0x9a, 0xf7),
        cyan: rgb(0x7d, 0xcf, 0xff),
        white: rgb(0xa9, 0xb1, 0xd6),
        br_black: rgb(0x41, 0x48, 0x68),
        br_red: rgb(0xf7, 0x76, 0x8e),
        br_green: rgb(0x9e, 0xce, 0x6a),
        br_yellow: rgb(0xe0, 0xaf, 0x68),
        br_blue: rgb(0x7a, 0xa2, 0xf7),
        br_magenta: rgb(0xbb, 0x9a, 0xf7),
        br_cyan: rgb(0x7d, 0xcf, 0xff),
        br_white: rgb(0xc0, 0xca, 0xf5),
    };

    pub const TOKYO_NIGHT_LIGHT: Palette16 = Palette16 {
        bg: rgb(0xd5, 0xd6, 0xdb),
        fg: rgb(0x34, 0x3b, 0x58),
        black: rgb(0x0f, 0x0f, 0x14),
        red: rgb(0x8c, 0x43, 0x51),
        green: rgb(0x33, 0x63, 0x5c),
        yellow: rgb(0x8f, 0x5e, 0x15),
        blue: rgb(0x34, 0x54, 0x8a),
        magenta: rgb(0x5a, 0x4a, 0x78),
        cyan: rgb(0x0f, 0x4b, 0x6e),
        white: rgb(0x68, 0x6c, 0x83),
        br_black: rgb(0x4c, 0x50, 0x5e),
        br_red: rgb(0x8c, 0x43, 0x51),
        br_green: rgb(0x33, 0x63, 0x5c),
        br_yellow: rgb(0x8f, 0x5e, 0x15),
        br_blue: rgb(0x34, 0x54, 0x8a),
        br_magenta: rgb(0x5a, 0x4a, 0x78),
        br_cyan: rgb(0x0f, 0x4b, 0x6e),
        br_white: rgb(0x34, 0x3b, 0x58),
    };

    /// Dracula — <https://draculatheme.com/>.
    pub const DRACULA: Palette16 = Palette16 {
        bg: rgb(0x28, 0x2a, 0x36),
        fg: rgb(0xf8, 0xf8, 0xf2),
        black: rgb(0x21, 0x22, 0x2c),
        red: rgb(0xff, 0x55, 0x55),
        green: rgb(0x50, 0xfa, 0x7b),
        yellow: rgb(0xf1, 0xfa, 0x8c),
        blue: rgb(0xbd, 0x93, 0xf9),
        magenta: rgb(0xff, 0x79, 0xc6),
        cyan: rgb(0x8b, 0xe9, 0xfd),
        white: rgb(0xf8, 0xf8, 0xf2),
        br_black: rgb(0x62, 0x72, 0xa4),
        br_red: rgb(0xff, 0x6e, 0x6e),
        br_green: rgb(0x69, 0xff, 0x94),
        br_yellow: rgb(0xff, 0xff, 0xa5),
        br_blue: rgb(0xd6, 0xac, 0xff),
        br_magenta: rgb(0xff, 0x92, 0xdf),
        br_cyan: rgb(0xa4, 0xff, 0xff),
        br_white: rgb(0xff, 0xff, 0xff),
    };

    /// Gruvbox Dark — morhetz, <https://github.com/morhetz/gruvbox>.
    pub const GRUVBOX_DARK: Palette16 = Palette16 {
        bg: rgb(0x28, 0x28, 0x28),
        fg: rgb(0xeb, 0xdb, 0xb2),
        black: rgb(0x28, 0x28, 0x28),
        red: rgb(0xcc, 0x24, 0x1d),
        green: rgb(0x98, 0x97, 0x1a),
        yellow: rgb(0xd7, 0x99, 0x21),
        blue: rgb(0x45, 0x85, 0x88),
        magenta: rgb(0xb1, 0x62, 0x86),
        cyan: rgb(0x68, 0x9d, 0x6a),
        white: rgb(0xa8, 0x99, 0x84),
        br_black: rgb(0x92, 0x83, 0x74),
        br_red: rgb(0xfb, 0x49, 0x34),
        br_green: rgb(0xb8, 0xbb, 0x26),
        br_yellow: rgb(0xfa, 0xbd, 0x2f),
        br_blue: rgb(0x83, 0xa5, 0x98),
        br_magenta: rgb(0xd3, 0x86, 0x9b),
        br_cyan: rgb(0x8e, 0xc0, 0x7c),
        br_white: rgb(0xeb, 0xdb, 0xb2),
    };

    /// Nord — Arctic Ice Studio, <https://www.nordtheme.com/>.
    pub const NORD: Palette16 = Palette16 {
        bg: rgb(0x2e, 0x34, 0x40),
        fg: rgb(0xd8, 0xde, 0xe9),
        black: rgb(0x3b, 0x42, 0x52),
        red: rgb(0xbf, 0x61, 0x6a),
        green: rgb(0xa3, 0xbe, 0x8c),
        yellow: rgb(0xeb, 0xcb, 0x8b),
        blue: rgb(0x81, 0xa1, 0xc1),
        magenta: rgb(0xb4, 0x8e, 0xad),
        cyan: rgb(0x88, 0xc0, 0xd0),
        white: rgb(0xe5, 0xe9, 0xf0),
        br_black: rgb(0x4c, 0x56, 0x6a),
        br_red: rgb(0xbf, 0x61, 0x6a),
        br_green: rgb(0xa3, 0xbe, 0x8c),
        br_yellow: rgb(0xeb, 0xcb, 0x8b),
        br_blue: rgb(0x81, 0xa1, 0xc1),
        br_magenta: rgb(0xb4, 0x8e, 0xad),
        br_cyan: rgb(0x8f, 0xbc, 0xbb),
        br_white: rgb(0xec, 0xef, 0xf4),
    };

    /// One Dark — Atom, <https://github.com/atom/one-dark-syntax>.
    pub const ONE_DARK: Palette16 = Palette16 {
        bg: rgb(0x28, 0x2c, 0x34),
        fg: rgb(0xab, 0xb2, 0xbf),
        black: rgb(0x28, 0x2c, 0x34),
        red: rgb(0xe0, 0x6c, 0x75),
        green: rgb(0x98, 0xc3, 0x79),
        yellow: rgb(0xe5, 0xc0, 0x7b),
        blue: rgb(0x61, 0xaf, 0xef),
        magenta: rgb(0xc6, 0x78, 0xdd),
        cyan: rgb(0x56, 0xb6, 0xc2),
        white: rgb(0xab, 0xb2, 0xbf),
        br_black: rgb(0x5c, 0x63, 0x70),
        br_red: rgb(0xe0, 0x6c, 0x75),
        br_green: rgb(0x98, 0xc3, 0x79),
        br_yellow: rgb(0xe5, 0xc0, 0x7b),
        br_blue: rgb(0x61, 0xaf, 0xef),
        br_magenta: rgb(0xc6, 0x78, 0xdd),
        br_cyan: rgb(0x56, 0xb6, 0xc2),
        br_white: rgb(0xff, 0xff, 0xff),
    };

    /// Monokai — Wimer Hazenberg.
    pub const MONOKAI: Palette16 = Palette16 {
        bg: rgb(0x27, 0x28, 0x22),
        fg: rgb(0xf8, 0xf8, 0xf2),
        black: rgb(0x27, 0x28, 0x22),
        red: rgb(0xf9, 0x26, 0x72),
        green: rgb(0xa6, 0xe2, 0x2e),
        yellow: rgb(0xf4, 0xbf, 0x75),
        blue: rgb(0x66, 0xd9, 0xef),
        magenta: rgb(0xae, 0x81, 0xff),
        cyan: rgb(0xa1, 0xef, 0xe4),
        white: rgb(0xf8, 0xf8, 0xf2),
        br_black: rgb(0x75, 0x71, 0x5e),
        br_red: rgb(0xf9, 0x26, 0x72),
        br_green: rgb(0xa6, 0xe2, 0x2e),
        br_yellow: rgb(0xf4, 0xbf, 0x75),
        br_blue: rgb(0x66, 0xd9, 0xef),
        br_magenta: rgb(0xae, 0x81, 0xff),
        br_cyan: rgb(0xa1, 0xef, 0xe4),
        br_white: rgb(0xf9, 0xf8, 0xf5),
    };

    /// Catppuccin Mocha — <https://github.com/catppuccin/catppuccin>.
    pub const CATPPUCCIN_MOCHA: Palette16 = Palette16 {
        bg: rgb(0x1e, 0x1e, 0x2e),
        fg: rgb(0xcd, 0xd6, 0xf4),
        black: rgb(0x45, 0x47, 0x5a),
        red: rgb(0xf3, 0x8b, 0xa8),
        green: rgb(0xa6, 0xe3, 0xa1),
        yellow: rgb(0xf9, 0xe2, 0xaf),
        blue: rgb(0x89, 0xb4, 0xfa),
        magenta: rgb(0xf5, 0xc2, 0xe7),
        cyan: rgb(0x94, 0xe2, 0xd5),
        white: rgb(0xba, 0xc2, 0xde),
        br_black: rgb(0x58, 0x5b, 0x70),
        br_red: rgb(0xf3, 0x8b, 0xa8),
        br_green: rgb(0xa6, 0xe3, 0xa1),
        br_yellow: rgb(0xf9, 0xe2, 0xaf),
        br_blue: rgb(0x89, 0xb4, 0xfa),
        br_magenta: rgb(0xf5, 0xc2, 0xe7),
        br_cyan: rgb(0x94, 0xe2, 0xd5),
        br_white: rgb(0xa6, 0xad, 0xc8),
    };

    /// Catppuccin Latte (light variant).
    pub const CATPPUCCIN_LATTE: Palette16 = Palette16 {
        bg: rgb(0xef, 0xf1, 0xf5),
        fg: rgb(0x4c, 0x4f, 0x69),
        black: rgb(0x5c, 0x5f, 0x77),
        red: rgb(0xd2, 0x0f, 0x39),
        green: rgb(0x40, 0xa0, 0x2b),
        yellow: rgb(0xdf, 0x8e, 0x1d),
        blue: rgb(0x1e, 0x66, 0xf5),
        magenta: rgb(0xea, 0x76, 0xcb),
        cyan: rgb(0x17, 0x92, 0x99),
        white: rgb(0xac, 0xb0, 0xbe),
        br_black: rgb(0x6c, 0x6f, 0x85),
        br_red: rgb(0xd2, 0x0f, 0x39),
        br_green: rgb(0x40, 0xa0, 0x2b),
        br_yellow: rgb(0xdf, 0x8e, 0x1d),
        br_blue: rgb(0x1e, 0x66, 0xf5),
        br_magenta: rgb(0xea, 0x76, 0xcb),
        br_cyan: rgb(0x17, 0x92, 0x99),
        br_white: rgb(0xbc, 0xc0, 0xcc),
    };

    /// Ayu Dark — <https://github.com/ayu-theme/ayu-colors>.
    pub const AYU_DARK: Palette16 = Palette16 {
        bg: rgb(0x0a, 0x0e, 0x14),
        fg: rgb(0xb3, 0xb1, 0xad),
        black: rgb(0x01, 0x06, 0x0e),
        red: rgb(0xea, 0x6c, 0x73),
        green: rgb(0x91, 0xb3, 0x62),
        yellow: rgb(0xf9, 0xaf, 0x4f),
        blue: rgb(0x53, 0xbd, 0xfa),
        magenta: rgb(0xfa, 0xe9, 0x94),
        cyan: rgb(0x90, 0xe1, 0xc6),
        white: rgb(0xc7, 0xc7, 0xc7),
        br_black: rgb(0x68, 0x68, 0x68),
        br_red: rgb(0xf0, 0x71, 0x78),
        br_green: rgb(0xc2, 0xd9, 0x4c),
        br_yellow: rgb(0xff, 0xb4, 0x54),
        br_blue: rgb(0x59, 0xc2, 0xff),
        br_magenta: rgb(0xff, 0xee, 0x99),
        br_cyan: rgb(0x95, 0xe6, 0xcb),
        br_white: rgb(0xff, 0xff, 0xff),
    };

    pub const AYU_LIGHT: Palette16 = Palette16 {
        bg: rgb(0xfa, 0xfa, 0xfa),
        fg: rgb(0x5c, 0x61, 0x66),
        black: rgb(0x00, 0x00, 0x00),
        red: rgb(0xf5, 0x18, 0x18),
        green: rgb(0x86, 0xb3, 0x00),
        yellow: rgb(0xf2, 0x97, 0x18),
        blue: rgb(0x41, 0xa6, 0xd9),
        magenta: rgb(0xf0, 0x71, 0x78),
        cyan: rgb(0x4c, 0xbf, 0x99),
        white: rgb(0xfc, 0xfc, 0xfc),
        br_black: rgb(0x82, 0x8c, 0x99),
        br_red: rgb(0xf0, 0x71, 0x71),
        br_green: rgb(0x86, 0xb3, 0x00),
        br_yellow: rgb(0xf2, 0xae, 0x49),
        br_blue: rgb(0x55, 0xb4, 0xd4),
        br_magenta: rgb(0xa3, 0x7a, 0xcc),
        br_cyan: rgb(0x4c, 0xbf, 0x99),
        br_white: rgb(0xff, 0xff, 0xff),
    };

    /// GitHub Dark — primer/primitives.
    pub const GITHUB_DARK: Palette16 = Palette16 {
        bg: rgb(0x0d, 0x11, 0x17),
        fg: rgb(0xc9, 0xd1, 0xd9),
        black: rgb(0x48, 0x4f, 0x58),
        red: rgb(0xff, 0x7b, 0x72),
        green: rgb(0x3f, 0xb9, 0x50),
        yellow: rgb(0xd2, 0x99, 0x22),
        blue: rgb(0x58, 0xa6, 0xff),
        magenta: rgb(0xbc, 0x8c, 0xff),
        cyan: rgb(0x39, 0xc5, 0xcf),
        white: rgb(0xb1, 0xba, 0xc4),
        br_black: rgb(0x6e, 0x76, 0x81),
        br_red: rgb(0xff, 0xa1, 0x98),
        br_green: rgb(0x56, 0xd3, 0x64),
        br_yellow: rgb(0xe3, 0xb3, 0x41),
        br_blue: rgb(0x79, 0xc0, 0xff),
        br_magenta: rgb(0xd2, 0xa8, 0xff),
        br_cyan: rgb(0x56, 0xd4, 0xdd),
        br_white: rgb(0xf0, 0xf6, 0xfc),
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_at_least_a_dozen_themes() {
        assert!(theme_names().len() >= 12);
        assert_eq!(theme_names()[0], "default-dark");
    }

    #[test]
    fn known_names_resolve_to_distinct_themes() {
        let mut bgs: Vec<_> = theme_names()
            .into_iter()
            .map(|n| format!("{:?}", Theme::from_name(n).bg))
            .collect();
        bgs.sort();
        let n = bgs.len();
        bgs.dedup();
        assert_eq!(bgs.len(), n, "two themes share the same bg color");
    }

    #[test]
    fn unknown_name_falls_back_to_default_dark() {
        let t = Theme::from_name("nope");
        let d = Theme::default_dark();
        assert_eq!(t.bg, d.bg);
        assert_eq!(t.accent, d.accent);
    }

    #[test]
    fn next_name_cycles_through_registry() {
        let names = theme_names();
        for i in 0..names.len() {
            let expected = names[(i + 1) % names.len()];
            assert_eq!(Theme::next_name(names[i]), expected);
        }
    }

    #[test]
    fn next_name_treats_unknown_as_first() {
        assert_eq!(Theme::next_name("does-not-exist"), theme_names()[1]);
    }

    #[test]
    fn from_palette_maps_semantic_slots() {
        let p = palettes::DRACULA;
        let t = Theme::from_palette(&p);
        assert_eq!(t.bg, p.bg);
        assert_eq!(t.fg, p.fg);
        assert_eq!(t.muted, p.br_black);
        assert_eq!(t.accent, p.br_cyan);
        assert_eq!(t.err, p.red);
    }
}
