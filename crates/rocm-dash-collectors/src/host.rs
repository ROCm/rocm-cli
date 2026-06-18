//! Host system metrics via sysinfo. Backs the daemon's per-tick `SystemMetrics`.

use rocm_dash_core::metrics::SystemMetrics;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

pub struct HostCollector {
    sys: System,
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
        let mut sys = System::new_with_specifics(refresh);
        // First refresh primes CPU deltas; the very first read is meaningless.
        sys.refresh_cpu_all();
        sys.refresh_memory();
        Self { sys }
    }

    pub fn tick(&mut self) -> SystemMetrics {
        self.sys.refresh_cpu_all();
        self.sys.refresh_memory();

        let cpu_overall_pct = self.sys.global_cpu_usage();
        let cpu_per_core_pct: Vec<f32> = self
            .sys
            .cpus()
            .iter()
            .map(sysinfo::Cpu::cpu_usage)
            .collect();

        // sysinfo reports memory in bytes.
        let memory_used_mb = self.sys.used_memory() / 1024 / 1024;
        let memory_total_mb = self.sys.total_memory() / 1024 / 1024;
        let swap_used_mb = self.sys.used_swap() / 1024 / 1024;
        let swap_total_mb = self.sys.total_swap() / 1024 / 1024;

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
}
