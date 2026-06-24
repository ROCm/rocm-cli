// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Exponential backoff helper for reconnect loops. Ported from ctux pattern.

use std::time::Duration;

pub struct Backoff {
    current: Duration,
    max: Duration,
    factor: u32,
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new(Duration::from_millis(250), Duration::from_secs(10), 2)
    }
}

impl Backoff {
    pub const fn new(initial: Duration, max: Duration, factor: u32) -> Self {
        Self {
            current: initial,
            max,
            factor,
        }
    }

    pub fn next_delay(&mut self) -> Duration {
        let d = self.current;
        self.current = (self.current * self.factor).min(self.max);
        d
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles_until_cap() {
        let mut b = Backoff::new(Duration::from_millis(100), Duration::from_millis(800), 2);
        assert_eq!(b.next_delay(), Duration::from_millis(100));
        assert_eq!(b.next_delay(), Duration::from_millis(200));
        assert_eq!(b.next_delay(), Duration::from_millis(400));
        assert_eq!(b.next_delay(), Duration::from_millis(800));
        assert_eq!(b.next_delay(), Duration::from_millis(800));
    }
}
