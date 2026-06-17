//! Small UI helpers shared across tabs.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use rocm_dash_core::metrics::{GpuMetrics, Instance, Snapshot};

use crate::ui::theme::Theme;

/// Temperature warning threshold (°C). At or above → warn color.
pub const TEMP_WARN_C: f32 = 60.0;
/// Temperature critical threshold (°C). At or above → err color.
pub const TEMP_CRIT_C: f32 = 80.0;

/// Board-power warning threshold (W). At or above → warn color.
///
/// Fixed semantic thresholds make heatmap/gauge colors mean "near the limit"
/// rather than "near the largest value seen this session". Tuned for
/// MI355X-class parts (TDP ~750 W); the demo generator peaks ~740 W at 100%
/// util. No per-GPU TDP exists in the data model, so these are constants.
pub const POWER_WARN_W: f32 = 525.0;
/// Board-power critical threshold (W). At or above → err color. ~TDP-adjacent.
pub const POWER_CRIT_W: f32 = 700.0;

/// Truncate `s` to at most `n` characters (not bytes).
pub fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect()
    }
}

/// Color a temperature value per the instinct-dash conventions.
pub fn temperature_style(c: f32, theme: &Theme) -> Style {
    let color = if c >= TEMP_CRIT_C {
        theme.err
    } else if c >= TEMP_WARN_C {
        theme.warn
    } else {
        theme.ok
    };
    Style::default().fg(color)
}

/// Color a board-power value against the fixed [`POWER_WARN_W`] /
/// [`POWER_CRIT_W`] thresholds, mirroring [`temperature_style`].
pub fn power_style(w: f32, theme: &Theme) -> Style {
    let color = if w >= POWER_CRIT_W {
        theme.err
    } else if w >= POWER_WARN_W {
        theme.warn
    } else {
        theme.ok
    };
    Style::default().fg(color)
}

/// Trailing run of ASCII digits in `s` (e.g. `"gpu-3"` → `"3"`, `"3"` → `"3"`).
/// Returns `None` when `s` has no trailing digits.
fn trailing_digits(s: &str) -> Option<&str> {
    let start = s.len() - s.chars().rev().take_while(char::is_ascii_digit).count();
    if start == s.len() {
        None
    } else {
        Some(&s[start..])
    }
}

/// Instances scheduled on the GPU identified by `device_id`.
///
/// GPU `device_id` (`"gpu-3"`) and `Instance.gpu_ids` (`"3"`) use different
/// shapes, so matching normalizes both to their trailing digit run before
/// comparing. A bare `"3"` device_id also matches `"3"`.
pub fn instances_on_gpu<'a>(device_id: &str, instances: &'a [Instance]) -> Vec<&'a Instance> {
    let want = trailing_digits(device_id);
    instances
        .iter()
        .filter(|inst| {
            inst.gpu_ids.iter().any(|gid| {
                // match on normalized trailing digits, falling back to raw eq
                match (want, trailing_digits(gid)) {
                    (Some(a), Some(b)) => a == b,
                    _ => gid == device_id,
                }
            })
        })
        .collect()
}

/// Node-level energy efficiency: total generation throughput divided by total board power, in tokens per watt.
///
/// `None` when there is no traffic
/// (`sum gen_tps == 0`) or no power telemetry (`sum power_w == 0`), or when the
/// result is non-finite.
pub fn node_efficiency(snap: &Snapshot) -> Option<f64> {
    let tps: f64 = snap.instances.iter().filter_map(|i| i.gen_tps).sum();
    let power: f64 = snap.gpus.iter().map(|g| f64::from(g.power_w)).sum();
    if tps > 0.0 && power > 0.0 {
        let eff = tps / power;
        eff.is_finite().then_some(eff)
    } else {
        None
    }
}

/// One-line GPU stats: id, util, vram, temp, power.
pub fn gpu_stats_line<'a>(g: &'a GpuMetrics, theme: &Theme) -> Line<'a> {
    let vram_pct = if g.vram_total_mb > 0 {
        100.0 * g.vram_used_mb as f64 / g.vram_total_mb as f64
    } else {
        0.0
    };
    Line::from(vec![
        Span::styled(
            format!("{:<8}", g.device_id),
            Style::default().fg(theme.accent),
        ),
        Span::styled(
            format!(" util {:5.1}%", g.gpu_utilization_pct),
            Style::default().fg(theme.fg),
        ),
        Span::styled(
            format!(
                "  vram {:>5}/{:<5} MB ({:4.1}%)",
                g.vram_used_mb, g.vram_total_mb, vram_pct
            ),
            Style::default().fg(theme.muted),
        ),
        Span::styled(
            format!("  {:>5.1}°C  {:>5.1} W", g.temperature_c, g.power_w),
            temperature_style(g.temperature_c, theme),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(name: &str, gpu_ids: &[&str], gen_tps: Option<f64>) -> Instance {
        Instance {
            container_name: name.into(),
            model_name: name.into(),
            gpu_ids: gpu_ids
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            gen_tps,
            ..Default::default()
        }
    }

    fn gpu(device_id: &str, power_w: f32) -> GpuMetrics {
        GpuMetrics {
            device_id: device_id.into(),
            power_w,
            ..Default::default()
        }
    }

    #[test]
    fn trunc_keeps_short_strings() {
        assert_eq!(trunc("abc", 5), "abc");
        assert_eq!(trunc("abcde", 5), "abcde");
    }

    #[test]
    fn trunc_cuts_long_strings_by_chars() {
        assert_eq!(trunc("abcdefgh", 5), "abcde");
        assert_eq!(trunc("αβγδεζη", 3), "αβγ");
    }

    #[test]
    fn power_style_uses_fixed_thresholds() {
        let theme = Theme::default_dark();
        assert_eq!(power_style(720.0, &theme).fg, Some(theme.err));
        assert_eq!(power_style(600.0, &theme).fg, Some(theme.warn));
        assert_eq!(power_style(300.0, &theme).fg, Some(theme.ok));
        // boundaries are inclusive at the threshold
        assert_eq!(power_style(POWER_CRIT_W, &theme).fg, Some(theme.err));
        assert_eq!(power_style(POWER_WARN_W, &theme).fg, Some(theme.warn));
    }

    #[test]
    fn temperature_style_uses_named_thresholds() {
        let theme = Theme::default_dark();
        assert_eq!(temperature_style(85.0, &theme).fg, Some(theme.err));
        assert_eq!(temperature_style(65.0, &theme).fg, Some(theme.warn));
        assert_eq!(temperature_style(40.0, &theme).fg, Some(theme.ok));
    }

    #[test]
    fn trailing_digits_extracts_index() {
        assert_eq!(trailing_digits("gpu-3"), Some("3"));
        assert_eq!(trailing_digits("3"), Some("3"));
        assert_eq!(trailing_digits("gpu-12"), Some("12"));
        assert_eq!(trailing_digits("gpu"), None);
        assert_eq!(trailing_digits(""), None);
    }

    #[test]
    fn instances_on_gpu_matches_normalized_index() {
        let xs = vec![
            inst("vllm-a", &["0", "1"], Some(10.0)),
            inst("vllm-b", &["3"], Some(20.0)),
        ];
        // "gpu-3" device_id normalizes to "3" → matches vllm-b
        let on3 = instances_on_gpu("gpu-3", &xs);
        assert_eq!(on3.len(), 1);
        assert_eq!(on3[0].model_name, "vllm-b");
        // bare "0" matches the "0" gpu_id
        let on0 = instances_on_gpu("0", &xs);
        assert_eq!(on0.len(), 1);
        assert_eq!(on0[0].model_name, "vllm-a");
    }

    #[test]
    fn instances_on_gpu_empty_when_no_match() {
        let xs = vec![inst("vllm-a", &["0"], Some(10.0))];
        assert!(instances_on_gpu("gpu-7", &xs).is_empty());
        assert!(instances_on_gpu("gpu-7", &[]).is_empty());
    }

    #[test]
    fn node_efficiency_divides_tps_by_power() {
        let snap = Snapshot {
            gpus: vec![gpu("gpu-0", 400.0), gpu("gpu-1", 600.0)],
            instances: vec![
                inst("a", &["0"], Some(300.0)),
                inst("b", &["1"], Some(200.0)),
            ],
            ..Default::default()
        };
        // (300 + 200) / (400 + 600) = 0.5 tok/W
        let eff = node_efficiency(&snap).expect("some");
        assert!((eff - 0.5).abs() < 1e-9, "got {eff}");
    }

    #[test]
    fn node_efficiency_none_without_power_or_traffic() {
        // no power
        let no_power = Snapshot {
            gpus: vec![gpu("gpu-0", 0.0)],
            instances: vec![inst("a", &["0"], Some(100.0))],
            ..Default::default()
        };
        assert_eq!(node_efficiency(&no_power), None);
        // no traffic
        let no_traffic = Snapshot {
            gpus: vec![gpu("gpu-0", 500.0)],
            instances: vec![inst("a", &["0"], None)],
            ..Default::default()
        };
        assert_eq!(node_efficiency(&no_traffic), None);
    }
}
