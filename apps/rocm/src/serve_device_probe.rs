// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! When `rocm serve`'s summary is allowed to spend a device probe.
//!
//! One decision, lifted out of `serve_cmd.rs` so that file stays inside the
//! module-size limit. It is a pure predicate over what the launch report says
//! happened, which is also what makes the cost — not just the rendered row —
//! assertable without a running engine.

/// Whether the deployment summary may spend a device probe, given what the
/// launch report says happened.
///
/// `!already_running` is the load-bearing half, not a cosmetic echo of the
/// smoke test's gate. A service that was already running is `"ready"` by
/// definition, so keying on the status alone opens this on *every* repeat
/// `rocm serve` of a live model — and the probe is not free: on a miss it costs
/// `engine_ipc::SERVE_DEVICE_PROBE_BUDGET` plus one in-flight probe, about 14s
/// with Lemonade's 8s inference probe, paid sequentially before the summary's
/// remaining waits. That is a long stall bought for a row this command cannot
/// learn anything new for — the engine answered the same thing when the service
/// was launched, and a *running* service's device is what `rocm services list`
/// reads, with its own budget for exactly that.
///
/// `probe` is a closure rather than a value for that reason: on the reuse path
/// the cost must not be paid at all, which is what the unit test pins.
pub(crate) fn serve_device_probe_opens<T>(
    already_running: bool,
    status: &str,
    probe: impl FnOnce() -> Option<T>,
) -> Option<T> {
    (!already_running && status == "ready")
        .then(probe)
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression coverage for the device probe's gate (ROCMAI-550 review).
    ///
    /// The gate shipped as `report.status == "ready"` alone, under a comment
    /// claiming it was "the same gate as the smoke test" — which it was not: the
    /// smoke test and the Anthropic-endpoint wait beside it both also require
    /// `!already_running`. A reused service is `"ready"` by definition, so the
    /// omission made every repeat `rocm serve` of a live model pay the probe's
    /// full retry budget (plus one in-flight probe, ~14s with Lemonade) before
    /// printing a summary of a launch that did not happen.
    ///
    /// The probe is asserted *not consulted*, not merely that the row is absent:
    /// the cost is the bug, and a gate that probed and then discarded the answer
    /// would satisfy a value-only assertion.
    #[test]
    fn the_device_probe_is_not_spent_on_a_service_that_was_already_running() {
        let mut probed = false;
        assert!(
            serve_device_probe_opens(true, "ready", || {
                probed = true;
                Some("gpu")
            })
            .is_none(),
            "a reused service must not open the device probe: it is `ready` by \
             definition, so the budget would be spent on every repeat `serve`"
        );
        assert!(
            !probed,
            "the probe must not even be called on the reuse path — the wall time \
             is the regression, not the `device:` row"
        );

        // A launch that is not ready has nothing to ask the engine about yet.
        let mut probed_starting = false;
        assert!(
            serve_device_probe_opens(false, "starting", || {
                probed_starting = true;
                Some("gpu")
            })
            .is_none()
        );
        assert!(!probed_starting);

        // The row itself still works for the case it was added for: a fresh
        // launch that reached ready, where the engine knows what it selected.
        assert_eq!(
            serve_device_probe_opens(false, "ready", || Some("gpu")),
            Some("gpu"),
            "a fresh ready launch must still read the engine's device, or the \
             Vulkan-only `device:` row this PR adds is gone"
        );
        // `None` from the engine stays a supported answer: no row, no error.
        assert_eq!(
            serve_device_probe_opens(false, "ready", || Option::<&str>::None),
            None
        );
    }
}
