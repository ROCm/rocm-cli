// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Result, bail};
use rocm_engine_protocol::{DevicePolicy, EngineDeviceAvailability, GpuSelection};
use serde_json::Value;

pub(crate) fn normalize_device_policy(policy: Option<DevicePolicy>) -> Result<DevicePolicy> {
    match policy.unwrap_or(DevicePolicy::GpuRequired) {
        DevicePolicy::GpuRequired | DevicePolicy::GpuPreferred => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::CpuOnly => {
            bail!("Lemonade adapter requires ROCm GPU execution; no CPU fallback is used")
        }
    }
}

pub(crate) fn require_gpu_required(policy: &DevicePolicy) -> Result<()> {
    match policy {
        DevicePolicy::GpuRequired | DevicePolicy::GpuPreferred => Ok(()),
        DevicePolicy::CpuOnly => {
            bail!("Lemonade adapter requires ROCm GPU execution; no CPU fallback is used")
        }
    }
}

/// Probe real GPU availability and resolve the device ordinal(s) to pin before a
/// GPU-required launch. Returns the resolved indices, or an error that stops
/// serving before any process is spawned:
/// - no usable device → fail with actionable guidance (no CPU/GPU-0 fallback);
/// - `auto` (empty `requested`) → the first present, visible device;
/// - explicit index not among the usable devices → fail.
///
/// When availability cannot be probed on this platform, the requested selection
/// is passed through unchanged so serving is not blocked on that basis.
pub(crate) fn resolve_serve_gpu_indices(requested: &[u32]) -> Result<Vec<u32>> {
    resolve_gpu_indices_against(requested, rocm_core::usable_amd_gpu_indices())
}

/// Pure resolution used by [`resolve_serve_gpu_indices`], split out so the
/// auto/explicit/no-device policy can be unit-tested without real hardware.
/// `usable` is the probe result: `None` (unprobeable) passes the request
/// through; `Some(_)` is authoritative.
fn resolve_gpu_indices_against(requested: &[u32], usable: Option<Vec<u32>>) -> Result<Vec<u32>> {
    let Some(usable) = usable else {
        return Ok(requested.to_vec());
    };
    if usable.is_empty() {
        bail!(
            "no usable AMD GPU detected; `rocm serve` requires a GPU under the default \
             GPU-required policy and does not fall back to CPU. Check the driver with \
             `rocm examine`, confirm /dev/kfd is present, and ensure HIP_VISIBLE_DEVICES / \
             ROCR_VISIBLE_DEVICES are not masking every device."
        );
    }
    if requested.is_empty() {
        return Ok(vec![usable[0]]);
    }
    for index in requested {
        if !usable.contains(index) {
            let visible = usable
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "requested GPU {index} is not available on this host; usable GPU indices: [{visible}]"
            );
        }
    }
    Ok(requested.to_vec())
}

/// GPU device availability for `detect`, sourced from the real device probe. When
/// the probe is authoritative it reports true/false with a reason; when it cannot
/// determine availability on this platform, it falls back to whether the runtime
/// is installed so `detect` stays informative.
pub(crate) fn gpu_availability_device(runtime_installed: bool) -> EngineDeviceAvailability {
    match rocm_core::usable_amd_gpu_indices() {
        Some(indices) if !indices.is_empty() => EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: true,
            reason: None,
        },
        Some(_) => EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: false,
            reason: Some(
                "no AMD GPU detected, or every device is masked by \
                 HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES"
                    .to_owned(),
            ),
        },
        None => EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: runtime_installed,
            reason: if runtime_installed {
                None
            } else {
                Some("Lemonade embeddable runtime is not installed".to_owned())
            },
        },
    }
}

pub(crate) fn parse_device_policy_arg(value: Option<&str>) -> Result<DevicePolicy> {
    match value.unwrap_or("gpu_required") {
        "gpu" | "gpu_required" | "gpu_preferred" => Ok(DevicePolicy::GpuRequired),
        "cpu" | "cpu_only" => Ok(DevicePolicy::CpuOnly),
        other => bail!("unknown device policy `{other}`"),
    }
}

/// Parse a `--gpu` CLI value into an optional `GpuSelection` for `LaunchRequest`.
pub(crate) fn parse_gpu_selection_arg(value: Option<&str>) -> Result<Option<GpuSelection>> {
    value
        .map(|raw| GpuSelection::parse_cli_value(raw).map_err(anyhow::Error::msg))
        .transpose()
}

/// Parse a `--gpu` CLI value into explicit device ordinals (empty for `auto`).
pub(crate) fn parse_gpu_indices_arg(value: Option<&str>) -> Result<Vec<u32>> {
    Ok(rocm_engine_protocol::launch_gpu_indices(
        parse_gpu_selection_arg(value)?.as_ref(),
    ))
}

pub(crate) const fn device_policy_name(policy: &DevicePolicy) -> &'static str {
    match policy {
        DevicePolicy::GpuRequired => "gpu_required",
        DevicePolicy::GpuPreferred => "gpu_preferred",
        DevicePolicy::CpuOnly => "cpu_only",
    }
}

/// The device string reported once a model is loaded. Reflects the backend that
/// actually ran (`rocm`, `vulkan`, …) and the pinned GPU ordinal when known,
/// rather than a hardcoded value, so the report matches observed execution.
pub(crate) fn reported_device(state: Option<&Value>, backend: &str) -> String {
    match state.and_then(first_gpu_index_from_state) {
        Some(index) => format!("{backend} gpu {index}"),
        None => format!("{backend} gpu"),
    }
}

/// The first pinned GPU ordinal recorded in service state, if any.
fn first_gpu_index_from_state(state: &Value) -> Option<u32> {
    state
        .get("gpu_indices")
        .and_then(Value::as_array)
        .and_then(|indices| indices.first())
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serve_gpu_resolution_fails_fast_when_no_device_usable() {
        // No-device / masked-device path: the probe authoritatively reports zero
        // usable GPUs, so serving is refused before any process is spawned.
        let error = resolve_gpu_indices_against(&[], Some(vec![]))
            .expect_err("zero usable devices must be rejected");
        assert!(error.to_string().contains("no usable AMD GPU"));
    }

    #[test]
    fn serve_gpu_resolution_auto_pins_first_present_device() {
        // `auto` (empty request) resolves to the first present device — never an
        // assumed device 0 when device 0 is not present.
        assert_eq!(
            resolve_gpu_indices_against(&[], Some(vec![1, 2])).unwrap(),
            vec![1]
        );
    }

    #[test]
    fn serve_gpu_resolution_validates_explicit_index() {
        // An explicitly requested present device is honored.
        assert_eq!(
            resolve_gpu_indices_against(&[2], Some(vec![0, 1, 2])).unwrap(),
            vec![2]
        );
        // A requested device that is not present is rejected rather than silently
        // remapped to another GPU.
        let error = resolve_gpu_indices_against(&[3], Some(vec![0, 1]))
            .expect_err("absent device must be rejected");
        assert!(
            error
                .to_string()
                .contains("requested GPU 3 is not available")
        );
    }

    #[test]
    fn serve_gpu_resolution_passes_through_when_unprobeable() {
        // When availability cannot be determined, the request is passed through
        // unchanged so serving is not blocked on that basis.
        assert_eq!(
            resolve_gpu_indices_against(&[], None).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(resolve_gpu_indices_against(&[1], None).unwrap(), vec![1]);
    }

    #[test]
    fn reported_device_reflects_backend_and_pinned_ordinal() {
        let state = json!({ "gpu_indices": [1] });
        assert_eq!(reported_device(Some(&state), "vulkan"), "vulkan gpu 1");
        // Without a recorded ordinal the backend is still reported truthfully.
        assert_eq!(reported_device(None, "rocm"), "rocm gpu");
    }

    #[test]
    fn device_policy_rejects_cpu_without_fallback() {
        let error = normalize_device_policy(Some(DevicePolicy::CpuOnly))
            .expect_err("cpu should be rejected")
            .to_string();
        assert!(error.contains("no CPU fallback"));
    }
}
