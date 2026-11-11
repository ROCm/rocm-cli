// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Host system metrics via sysinfo. Backs the daemon's per-tick `SystemMetrics`.

use rocm_dash_core::metrics::SystemMetrics;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

pub struct HostCollector {
    // `None` when sysinfo could not initialise on this platform; `tick` then
    // reports empty metrics instead of failing the whole dashboard.
    sys: Option<System>,
}

impl Default for HostCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl HostCollector {
    pub fn new() -> Self {
        let refresh = RefreshKind::nothing()
            .with_cpu(sysinfo::CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());
        Self::from_system(System::new_with_specifics(refresh))
    }

    fn from_system<E: std::fmt::Display>(sys: Result<System, E>) -> Self {
        match sys {
            Ok(mut sys) => {
                // First refresh primes CPU deltas; the very first read is meaningless.
                sys.refresh_cpu_all();
                sys.refresh_memory();
                Self { sys: Some(sys) }
            }
            Err(error) => {
                tracing::warn!(%error, "host metrics unavailable: sysinfo failed to initialise");
                Self { sys: None }
            }
        }
    }

    pub fn tick(&mut self) -> SystemMetrics {
        let Some(sys) = self.sys.as_mut() else {
            return SystemMetrics::default();
        };
        sys.refresh_cpu_all();
        sys.refresh_memory();

        let cpu_overall_pct = sys.global_cpu_usage();
        let cpu_per_core_pct: Vec<f32> = sys.cpus().iter().map(sysinfo::Cpu::usage).collect();

        // sysinfo reports memory in bytes.
        let memory_used_mb = sys.used_memory() / 1024 / 1024;
        let memory_total_mb = sys.total_memory() / 1024 / 1024;
        let swap_used_mb = sys.used_swap() / 1024 / 1024;
        let swap_total_mb = sys.total_swap() / 1024 / 1024;

        SystemMetrics {
            cpu_overall_pct,
            cpu_per_core_pct,
            memory_used_mb,
            memory_total_mb,
            swap_used_mb,
            swap_total_mb,
            // TODO: wire `sysinfo::Disks` / `sysinfo::Networks` for I/O deltas.
            disk_read_bps: 0,
            disk_write_bps: 0,
            net_rx_bps: 0,
            net_tx_bps: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_produces_metrics_with_some_cpus() {
        let mut c = HostCollector::new();
        let m = c.tick();
        assert!(!m.cpu_per_core_pct.is_empty(), "expected at least one CPU");
        assert!(m.memory_total_mb > 0, "expected nonzero total memory");
    }

    #[test]
    fn a_collector_whose_sysinfo_failed_to_initialise_reports_empty_metrics() {
        // sysinfo 0.40 made `System` construction fallible. The dashboard must
        // keep ticking, with nothing to show for the host, rather than panic.
        let mut c = HostCollector::from_system::<&str>(Err("unsupported platform"));
        let m = c.tick();
        assert!(m.cpu_per_core_pct.is_empty(), "{m:?}");
        assert_eq!(m.memory_total_mb, 0, "{m:?}");
        assert_eq!(m.swap_total_mb, 0, "{m:?}");
    }
}
