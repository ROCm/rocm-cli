//! sysfs/hwmon collector for Strix Halo (gfx1151). Stub.

use rocm_dash_core::metrics::{GpuMetrics, GpuSystemInfo};
use rocm_dash_core::traits::{CollectorError, GpuCollector, GpuDevice, GpuProcess, Result};

#[derive(Debug, Default)]
pub struct SysfsGpuCollector;

impl SysfsGpuCollector {
    pub fn new() -> Self {
        Self
    }
}

impl GpuCollector for SysfsGpuCollector {
    fn name(&self) -> &'static str {
        "sysfs"
    }

    fn devices(&self) -> Result<Vec<GpuDevice>> {
        Err(CollectorError::Unsupported("sysfs collector stub".into()))
    }

    fn metrics(&self) -> Result<Vec<GpuMetrics>> {
        Err(CollectorError::Unsupported("sysfs collector stub".into()))
    }

    fn system_info(&self) -> Result<GpuSystemInfo> {
        Err(CollectorError::Unsupported("sysfs collector stub".into()))
    }

    fn processes(&self) -> Result<Vec<GpuProcess>> {
        Err(CollectorError::Unsupported("sysfs collector stub".into()))
    }
}
