// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! vLLM log slicer — verbatim port of `inspect_bench/log_slicer.py`.
//! See `../wiki/entities/log-slicer.md` and `../wiki/concepts/log-derived-metrics.md`.

use regex::Regex;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VllmPeaks {
    pub prompt_tps: Option<f64>,
    pub gen_tps: Option<f64>,
    pub max_running_reqs: Option<u32>,
    pub max_waiting_reqs: Option<u32>,
    pub n_requests: u32,
}

#[allow(clippy::struct_field_names)] // `_re` suffix is meaningful (these are Regex fields)
pub struct VllmLogSlicer {
    prompt_tps_re: Regex,
    gen_tps_re: Regex,
    running_re: Regex,
    waiting_re: Regex,
    req_re: Regex,
}

impl Default for VllmLogSlicer {
    fn default() -> Self {
        Self::new()
    }
}

impl VllmLogSlicer {
    pub fn new() -> Self {
        Self {
            prompt_tps_re: Regex::new(r"Avg prompt throughput: ([0-9.]+)").unwrap(),
            gen_tps_re: Regex::new(r"Avg generation throughput: ([0-9.]+)").unwrap(),
            running_re: Regex::new(r"Running: ([0-9]+)").unwrap(),
            waiting_re: Regex::new(r"Waiting: ([0-9]+)").unwrap(),
            req_re: Regex::new(r"POST /v1/(chat/completions|completions|messages)").unwrap(),
        }
    }

    /// Aggregate peaks over a byte-bounded slice of the vLLM log.
    pub fn parse_slice(&self, text: &str) -> VllmPeaks {
        let mut out = VllmPeaks::default();
        for cap in self.prompt_tps_re.captures_iter(text) {
            if let Ok(v) = cap[1].parse::<f64>() {
                out.prompt_tps = Some(out.prompt_tps.map_or(v, |p| p.max(v)));
            }
        }
        for cap in self.gen_tps_re.captures_iter(text) {
            if let Ok(v) = cap[1].parse::<f64>() {
                out.gen_tps = Some(out.gen_tps.map_or(v, |p| p.max(v)));
            }
        }
        for cap in self.running_re.captures_iter(text) {
            if let Ok(v) = cap[1].parse::<u32>() {
                out.max_running_reqs = Some(out.max_running_reqs.map_or(v, |p| p.max(v)));
            }
        }
        for cap in self.waiting_re.captures_iter(text) {
            if let Ok(v) = cap[1].parse::<u32>() {
                out.max_waiting_reqs = Some(out.max_waiting_reqs.map_or(v, |p| p.max(v)));
            }
        }
        out.n_requests = self.req_re.find_iter(text).count() as u32;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_throughput_peaks_and_request_count() {
        let log = r"
[2026-05-26 10:00:00] Avg prompt throughput: 1234.5 tokens/s
[2026-05-26 10:00:01] Avg generation throughput: 67.8 tokens/s
[2026-05-26 10:00:02] Running: 8, Waiting: 2
[2026-05-26 10:00:03] Avg prompt throughput: 2000.1 tokens/s
[2026-05-26 10:00:03] Avg generation throughput: 50.0 tokens/s
[2026-05-26 10:00:04] Running: 16, Waiting: 0
INFO POST /v1/chat/completions 200
INFO POST /v1/completions 200
INFO POST /v1/messages 200
INFO POST /v1/chat/completions 200
";
        let slicer = VllmLogSlicer::new();
        let peaks = slicer.parse_slice(log);
        assert_eq!(peaks.prompt_tps, Some(2000.1));
        assert_eq!(peaks.gen_tps, Some(67.8));
        assert_eq!(peaks.max_running_reqs, Some(16));
        assert_eq!(peaks.max_waiting_reqs, Some(2));
        assert_eq!(peaks.n_requests, 4);
    }

    #[test]
    fn empty_log_is_all_none() {
        let slicer = VllmLogSlicer::new();
        let peaks = slicer.parse_slice("");
        assert_eq!(peaks, VllmPeaks::default());
    }
}
