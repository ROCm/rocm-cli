// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The pre-split `ui::modal` paths.
//!
//! The Modal renderers that used to live in this file are now in
//! `ui/help_modal.rs`, `ui/warnings_modal.rs`, `ui/theme_picker.rs` and
//! `ui/menu.rs`, and the popup geometry and the Clear-then-block pattern
//! (`centered_rect`, `centered_width`, `draw_popup_frame`, `grey_overlay`) live
//! in `ui/panel.rs`. This module re-exports all of them so every item that was
//! public at `ui::modal` before the split still resolves there.

// Re-exported, not just imported: these were public at this path before the
// split, so the pre-split public paths stay valid.
pub use crate::ui::help_modal::{draw_global_help, draw_help};
pub use crate::ui::menu::{draw_logo, draw_menu, draw_options, draw_palette, opt_row};
pub use crate::ui::panel::{centered_rect, centered_width, draw_popup_frame, grey_overlay};
pub use crate::ui::theme_picker::{draw_theme_picker, draw_theme_preview};
pub use crate::ui::warnings_modal::draw_warnings;
