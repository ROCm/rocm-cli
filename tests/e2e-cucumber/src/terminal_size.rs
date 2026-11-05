// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Validation of a requested terminal size before it reaches the emulator.

use std::num::NonZeroU16;

/// `rows` and `cols` as the `NonZeroU16`s the `vt100` emulator takes, or an
/// error naming the requested size if either is zero.
///
/// A pure function on purpose: the PTY driver lives in the cucumber binary,
/// where plain `#[test]`s never run, so the zero check is kept here where
/// `cargo test -p e2e-cucumber --lib` covers it.
pub fn non_zero_dimensions(rows: u16, cols: u16) -> Result<(NonZeroU16, NonZeroU16), String> {
    match (NonZeroU16::new(rows), NonZeroU16::new(cols)) {
        (Some(rows), Some(cols)) => Ok((rows, cols)),
        _ => Err(format!(
            "cannot resize to rows={rows} cols={cols}: zero dimension"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::non_zero_dimensions;

    #[test]
    fn accepts_non_zero_dimensions_and_keeps_their_order() {
        let (rows, cols) = non_zero_dimensions(24, 80).expect("24x80 is a valid size");
        assert_eq!((rows.get(), cols.get()), (24, 80));
    }

    #[test]
    fn rejects_zero_rows_and_reports_the_requested_size() {
        assert_eq!(
            non_zero_dimensions(0, 80),
            Err("cannot resize to rows=0 cols=80: zero dimension".to_owned())
        );
    }

    #[test]
    fn rejects_zero_cols_and_reports_the_requested_size() {
        assert_eq!(
            non_zero_dimensions(24, 0),
            Err("cannot resize to rows=24 cols=0: zero dimension".to_owned())
        );
    }

    #[test]
    fn rejects_zero_in_both_dimensions() {
        assert_eq!(
            non_zero_dimensions(0, 0),
            Err("cannot resize to rows=0 cols=0: zero dimension".to_owned())
        );
    }
}
