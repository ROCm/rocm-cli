// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The popup geometry helpers' pre-split path.
//!
//! The Modal renderers that used to live in this file are now in
//! `ui/help_modal.rs`, `ui/warnings_modal.rs`, `ui/theme_picker.rs` and
//! `ui/menu.rs`. The popup geometry and the Clear-then-block pattern
//! (`centered_rect`, `centered_width`, `draw_popup_frame`, `grey_overlay`) live
//! in `ui/panel.rs`; this module re-exports those four helpers so their
//! pre-split `ui::modal` paths still resolve.

// Re-exported, not just imported: these four were public at this path before
// the geometry moved into `panel`, so the pre-split public paths stay valid.
pub use crate::ui::panel::{centered_rect, centered_width, draw_popup_frame, grey_overlay};
