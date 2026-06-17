//! Humanized number / unit formatters.
//!
//! Pure functions, no rendering deps. Used by every tab to keep numeric
//! columns scannable at a glance.
//!
//! Conventions:
//! - Binary units for memory (MiB / GiB), since amd-smi and sysinfo report
//!   tibibytes-of-bytes. We translate field names that say `..._mb` into
//!   "MiB / GiB" once they cross 1024.
//! - SI units (k / M / B) for token throughput and request counts.
//! - Percentages always with 1 decimal unless < 0.1, then 2 decimals.
//! - Optional values render `-`.

/// Format a byte count that's already in mebibytes (e.g. amd-smi `vram_used_mb`).
/// Promotes to GiB at 1024, TiB at 1024², with one decimal.
pub fn mib(value: u64) -> String {
    if value >= 1024 * 1024 {
        format!("{:.1} TiB", value as f64 / (1024.0 * 1024.0))
    } else if value >= 1024 {
        format!("{:.1} GiB", value as f64 / 1024.0)
    } else {
        format!("{value} MiB")
    }
}

/// Pair of (used_mib, total_mib) → "used / total" with promotion. Both promoted
/// to the same unit (driven by total) so they compare visually.
pub fn mib_pair(used: u64, total: u64) -> String {
    if total >= 1024 * 1024 {
        let scale = 1024.0 * 1024.0;
        format!(
            "{:.1} / {:.1} TiB",
            used as f64 / scale,
            total as f64 / scale
        )
    } else if total >= 1024 {
        let scale = 1024.0;
        format!(
            "{:.1} / {:.1} GiB",
            used as f64 / scale,
            total as f64 / scale
        )
    } else {
        format!("{used} / {total} MiB")
    }
}

/// Percentage rendered with one decimal, two when very small.
pub fn pct(value: f32) -> String {
    if value > 0.0 && value < 0.1 {
        format!("{value:.2}%")
    } else {
        format!("{value:.1}%")
    }
}

/// `Option<f32>` percentage → `-` when None.
pub fn pct_opt(value: Option<f32>) -> String {
    match value {
        Some(v) => pct(v),
        None => "-".to_string(),
    }
}

/// SI-suffixed number: 1234 → "1.23 k", 1_234_567 → "1.23 M".
/// Below 1000 returns the raw integer with no suffix.
pub fn si(value: f64) -> String {
    let av = value.abs();
    if av < 1_000.0 {
        if value.fract() == 0.0 {
            format!("{}", value as i64)
        } else {
            format!("{value:.1}")
        }
    } else if av < 1_000_000.0 {
        format!("{:.2} k", value / 1_000.0)
    } else if av < 1_000_000_000.0 {
        format!("{:.2} M", value / 1_000_000.0)
    } else {
        format!("{:.2} B", value / 1_000_000_000.0)
    }
}

/// Byte-rate (bytes per second), SI-suffixed: `512/s`, `1.20 k/s`, `1.20 M/s`.
///
/// Used for disk and network throughput on the Hardware tab. Reuses [`si`], so
/// the magnitude suffix (k/M/B) carries the scale and `/s` marks it as a rate;
/// the unit is bytes-per-second by context (the panel labels say disk / net).
/// No panic at 0 or non-finite input.
pub fn bps(value: f64) -> String {
    if !value.is_finite() {
        return "-".to_string();
    }
    format!("{}/s", si(value.max(0.0)))
}

/// Token throughput. `123.4 tok/s`, `1.23 k tok/s`. `-` when None.
pub fn tps_opt(value: Option<f64>) -> String {
    match value {
        Some(v) if v >= 1_000.0 => format!("{} tok/s", si(v)),
        Some(v) => format!("{v:.1} tok/s"),
        None => "-".to_string(),
    }
}

/// Energy efficiency: generation throughput per watt. `0.42 tok/W`. `-` when
/// None or non-finite (no throughput sample or no GPU power telemetry).
pub fn tokens_per_watt(value: Option<f64>) -> String {
    match value {
        Some(v) if v.is_finite() => format!("{v:.2} tok/W"),
        _ => "-".to_string(),
    }
}

/// Human duration from seconds. Sub-second → `ms`; otherwise `Hh Mm Ss`,
/// dropping any leading zero components.
pub fn duration(seconds: f64) -> String {
    if seconds < 1.0 {
        let ms = (seconds * 1000.0).round() as i64;
        return format!("{ms} ms");
    }
    let total = seconds.round() as i64;
    let h = total / 3_600;
    let m = (total % 3_600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}h {m}m {s}s")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

/// Request counter rendered with SI suffix for big numbers and `-` for None.
pub fn reqs_opt(value: Option<u32>) -> String {
    match value {
        Some(v) if v >= 1_000 => si(f64::from(v)),
        Some(v) => v.to_string(),
        None => "-".to_string(),
    }
}

/// Power in watts. One decimal, always trailing `W`.
pub fn watts(value: f32) -> String {
    format!("{value:.1} W")
}

/// Temperature in °C. One decimal, always trailing `°C`.
pub fn celsius(value: f32) -> String {
    format!("{value:.1}°C")
}

/// Clock in MHz, promoted to GHz once it crosses 1000.
pub fn mhz(value: u64) -> String {
    if value >= 1000 {
        format!("{:.2} GHz", value as f64 / 1000.0)
    } else {
        format!("{value} MHz")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mib_promotes_to_gib_then_tib() {
        assert_eq!(mib(0), "0 MiB");
        assert_eq!(mib(512), "512 MiB");
        assert_eq!(mib(1024), "1.0 GiB");
        assert_eq!(mib(2048 + 512), "2.5 GiB");
        assert_eq!(mib(1024 * 1024), "1.0 TiB");
        assert_eq!(mib(1024 * 1024 * 3), "3.0 TiB");
    }

    #[test]
    fn mib_pair_uses_total_to_pick_unit() {
        assert_eq!(mib_pair(256, 512), "256 / 512 MiB");
        assert_eq!(mib_pair(2048, 4096), "2.0 / 4.0 GiB");
        assert_eq!(mib_pair(1024, 1024 * 1024), "0.0 / 1.0 TiB");
    }

    #[test]
    fn pct_uses_two_decimals_for_tiny_values() {
        assert_eq!(pct(0.0), "0.0%");
        assert_eq!(pct(0.05), "0.05%");
        assert_eq!(pct(42.3), "42.3%");
        assert_eq!(pct(100.0), "100.0%");
    }

    #[test]
    fn pct_opt_handles_none() {
        assert_eq!(pct_opt(None), "-");
        assert_eq!(pct_opt(Some(75.0)), "75.0%");
    }

    #[test]
    fn si_scales_into_k_m_b() {
        assert_eq!(si(0.0), "0");
        assert_eq!(si(123.0), "123");
        assert_eq!(si(999.0), "999");
        assert_eq!(si(1234.0), "1.23 k");
        assert_eq!(si(1_234_567.0), "1.23 M");
        assert_eq!(si(2_500_000_000.0), "2.50 B");
    }

    #[test]
    fn bps_appends_rate_suffix_and_scales() {
        assert_eq!(bps(0.0), "0/s");
        assert_eq!(bps(512.0), "512/s");
        assert!(bps(512.0).contains("/s"));
        assert_eq!(bps(1_200_000.0), "1.20 M/s");
        assert!(bps(1_200_000.0).contains("M/s"));
        assert_eq!(bps(2_500.0), "2.50 k/s");
        // non-finite and negative are handled without panic
        assert_eq!(bps(f64::NAN), "-");
        assert_eq!(bps(-5.0), "0/s");
    }

    #[test]
    fn tps_opt_promotes_at_thousand() {
        assert_eq!(tps_opt(None), "-");
        assert_eq!(tps_opt(Some(45.6)), "45.6 tok/s");
        assert_eq!(tps_opt(Some(1500.0)), "1.50 k tok/s");
    }

    #[test]
    fn tokens_per_watt_renders_or_dashes() {
        assert_eq!(tokens_per_watt(None), "-");
        assert_eq!(tokens_per_watt(Some(0.42)), "0.42 tok/W");
        assert_eq!(tokens_per_watt(Some(f64::INFINITY)), "-");
    }

    #[test]
    fn duration_picks_smallest_unit_combo() {
        assert_eq!(duration(0.42), "420 ms");
        assert_eq!(duration(1.0), "1s");
        assert_eq!(duration(75.0), "1m 15s");
        assert_eq!(duration(3700.0), "1h 1m 40s");
    }

    #[test]
    fn reqs_opt_collapses_big_counts() {
        assert_eq!(reqs_opt(None), "-");
        assert_eq!(reqs_opt(Some(5)), "5");
        assert_eq!(reqs_opt(Some(12_000)), "12.00 k");
    }

    #[test]
    fn watts_and_celsius_and_mhz() {
        assert_eq!(watts(123.4), "123.4 W");
        assert_eq!(celsius(67.0), "67.0°C");
        assert_eq!(mhz(2400), "2.40 GHz");
        assert_eq!(mhz(800), "800 MHz");
    }
}
