// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Which device memory pool the serve drain-wait judges, and the floor it
//! demands of it.
//!
//! The wait between serve scenarios (`wait_for_free_vram` in the serving
//! steps) exists to see a just-killed engine hand its device memory back. It
//! has to look at the pool the engine allocates from. On a discrete card that
//! is VRAM. On an APU such as Strix Halo the VRAM total is only the fixed BIOS
//! carveout (512 MiB on the CI hosts) while the engine draws on GTT-backed
//! system memory (about 100 GiB there), so a floor taken from the carveout can
//! never be met and every GPU serve spent its whole drain deadline waiting.
//! `.github/scripts/gpu-preflight.sh` makes the same switch for the same reason.

/// Upper bound on the free-memory floor (MiB).
///
/// Sized so the largest single
/// scenario model can allocate its engine's memory share without tripping its
/// startup check: Qwen3.6-27B plus vLLM's default fraction-of-total KV
/// reservation needs the MI300X mostly clear. Only a data-center GPU has this
/// much, so on smaller pools the floor is capped to 90% of the pool total (see
/// [`required_free_mib`]) — otherwise the check could never pass. Note the cap
/// no longer sits above what vLLM asks for: rocm-cli used to pin utilization
/// below it, and now defers to vLLM's own (higher) default, so the wait has
/// little headroom left and can time out on a device that is merely holding
/// display memory. The wait is best-effort and the serve proceeds regardless,
/// so this costs time rather than correctness; sizing a deliberate margin is
/// follow-up work.
pub const MAX_FREE_FLOOR_MIB: u64 = 150_000;

/// A VRAM total below this is an APU's BIOS carveout, not private device memory.
///
/// The engine's allocations come out of GTT instead. It is the
/// 8 GiB floor the Strix Halo preflight lanes pass as
/// `GPU_PREFLIGHT_MIN_FREE_GIB`, which `gpu-preflight.sh` compares the VRAM
/// total against for the same purpose. Every discrete card the lanes run on is
/// far above it (48 GiB and up).
pub const CARVEOUT_MAX_MIB: u64 = 8 * 1024;

/// One memory pool's size and free space, in MiB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pool {
    pub total_mib: u64,
    pub free_mib: u64,
}

/// What the device tooling reported for GPU 0. `gtt` is `None` when the tool
/// that answered has no GTT figure (or none was readable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemory {
    pub vram: Pool,
    pub gtt: Option<Pool>,
}

/// Which pool a [`DrainTarget`] was read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    Vram,
    Gtt,
}

/// The pool to wait on, its current state, and the floor it must reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrainTarget {
    pub kind: PoolKind,
    pub total_mib: u64,
    pub free_mib: u64,
    pub floor_mib: u64,
}

impl DrainTarget {
    /// Whether the pool has handed back enough memory.
    #[must_use]
    pub const fn drained(&self) -> bool {
        self.free_mib >= self.floor_mib
    }
}

/// The free-memory floor to wait for on a pool of `total_mib`: the
/// model-sized ceiling, but never more than 90% of the pool.
#[must_use]
pub fn required_free_mib(total_mib: u64) -> u64 {
    MAX_FREE_FLOOR_MIB.min(total_mib / 100 * 90)
}

/// Choose the pool the engine allocates from and the floor to hold it to.
///
/// A VRAM total under [`CARVEOUT_MAX_MIB`] is a carveout, so the GTT pool is
/// used when one was read and is non-empty. A zero GTT total is a failed
/// reading, not a pool, and falls back to VRAM, as it does in
/// `gpu-preflight.sh`. Without GTT data the answer is VRAM, which is what
/// every host got before this choice existed.
#[must_use]
pub fn select_drain_target(memory: DeviceMemory) -> DrainTarget {
    let (kind, pool) = match memory.gtt {
        Some(gtt) if memory.vram.total_mib < CARVEOUT_MAX_MIB && gtt.total_mib > 0 => {
            (PoolKind::Gtt, gtt)
        }
        _ => (PoolKind::Vram, memory.vram),
    };
    DrainTarget {
        kind,
        total_mib: pool.total_mib,
        free_mib: pool.free_mib,
        floor_mib: required_free_mib(pool.total_mib),
    }
}

/// Read `(total, free)` in MiB from `rocm-smi --showmeminfo <pool> --csv` output.
///
/// `total_column` and `used_column` are substrings of the two header
/// names (`Total Memory` and `Total Used Memory` for VRAM; the `GTT ` forms for
/// GTT); the first data row is the first GPU. Values are bytes in the output.
#[must_use]
pub fn parse_rocm_smi_csv_pool(
    text: &str,
    total_column: &str,
    used_column: &str,
) -> Option<(u64, u64)> {
    let mut lines = text.lines();
    let header = lines.next()?;
    let cols: Vec<&str> = header.split(',').collect();
    let total_idx = cols.iter().position(|c| c.contains(total_column))?;
    let used_idx = cols.iter().position(|c| c.contains(used_column))?;
    let row = lines.next()?;
    let vals: Vec<&str> = row.split(',').collect();
    let total: u64 = vals.get(total_idx)?.trim().parse().ok()?;
    let used: u64 = vals.get(used_idx)?.trim().parse().ok()?;
    let mib = 1024 * 1024;
    Some((total / mib, total.saturating_sub(used) / mib))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pools `gpu-preflight.sh` printed on a Strix Halo lane host
    /// (`VRAM 203321344/536870912, GTT 41787392/107374182400`, used/total in
    /// bytes), and the same reading in MiB.
    fn strix_halo() -> DeviceMemory {
        DeviceMemory {
            vram: Pool {
                total_mib: 512,
                free_mib: 318,
            },
            gtt: Some(Pool {
                total_mib: 102_400,
                free_mib: 102_360,
            }),
        }
    }

    fn mi300x() -> DeviceMemory {
        DeviceMemory {
            vram: Pool {
                total_mib: 196_592,
                free_mib: 196_309,
            },
            gtt: Some(Pool {
                total_mib: 1_000_000,
                free_mib: 1_000_000,
            }),
        }
    }

    #[test]
    fn a_carveout_host_is_judged_by_gtt_and_a_healthy_strix_halo_counts_as_drained() {
        let target = select_drain_target(strix_halo());
        assert_eq!(target.kind, PoolKind::Gtt);
        assert_eq!(target.total_mib, 102_400);
        assert_eq!(target.floor_mib, 92_160);
        assert!(
            target.drained(),
            "an idle Strix Halo host must not wait out the drain deadline: {target:?}"
        );
    }

    #[test]
    fn a_carveout_host_whose_gtt_is_mostly_held_is_not_drained() {
        let mut memory = strix_halo();
        memory.gtt = Some(Pool {
            total_mib: 102_400,
            free_mib: 60_000,
        });
        let target = select_drain_target(memory);
        assert_eq!(target.kind, PoolKind::Gtt);
        assert!(!target.drained(), "{target:?}");
    }

    #[test]
    fn a_discrete_card_is_still_judged_by_vram_with_the_model_sized_cap() {
        let target = select_drain_target(mi300x());
        assert_eq!(target.kind, PoolKind::Vram);
        assert_eq!(target.total_mib, 196_592);
        assert_eq!(target.floor_mib, MAX_FREE_FLOOR_MIB);
        assert!(target.drained());
    }

    #[test]
    fn a_discrete_card_holding_most_of_its_vram_is_not_drained() {
        let mut memory = mi300x();
        memory.vram.free_mib = 40_000;
        assert!(!select_drain_target(memory).drained());
    }

    #[test]
    fn a_host_without_gtt_data_keeps_the_vram_behaviour() {
        let mut memory = strix_halo();
        memory.gtt = None;
        let target = select_drain_target(memory);
        assert_eq!(target.kind, PoolKind::Vram);
        assert_eq!(
            target.floor_mib, 450,
            "the floor the CI log showed for the carveout"
        );
        assert!(!target.drained());
    }

    #[test]
    fn a_zero_gtt_total_is_a_failed_reading_not_a_pool() {
        let mut memory = strix_halo();
        memory.gtt = Some(Pool {
            total_mib: 0,
            free_mib: 0,
        });
        assert_eq!(select_drain_target(memory).kind, PoolKind::Vram);
    }

    #[test]
    fn a_vram_total_at_the_carveout_limit_is_private_memory() {
        let mut memory = strix_halo();
        memory.vram = Pool {
            total_mib: CARVEOUT_MAX_MIB,
            free_mib: CARVEOUT_MAX_MIB,
        };
        assert_eq!(select_drain_target(memory).kind, PoolKind::Vram);
    }

    #[test]
    fn rocm_smi_csv_gtt_is_read_in_mib() {
        let text = "device,GTT Total Memory (B),GTT Total Used Memory (B)\n\
                    card0,107374182400,41787392\n";
        assert_eq!(
            parse_rocm_smi_csv_pool(text, "GTT Total Memory", "GTT Total Used Memory"),
            Some((102_400, 102_360))
        );
    }

    #[test]
    fn rocm_smi_csv_vram_keeps_its_original_column_match() {
        let text = "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n\
                    card0,536870912,203321344\n";
        assert_eq!(
            parse_rocm_smi_csv_pool(text, "Total Memory", "Total Used Memory"),
            Some((512, 318))
        );
    }

    #[test]
    fn rocm_smi_csv_without_the_pool_is_none() {
        let text = "device,VRAM Total Memory (B),VRAM Total Used Memory (B)\n\
                    card0,536870912,203321344\n";
        assert_eq!(
            parse_rocm_smi_csv_pool(text, "GTT Total Memory", "GTT Total Used Memory"),
            None
        );
        assert_eq!(
            parse_rocm_smi_csv_pool("", "Total Memory", "Total Used Memory"),
            None
        );
    }
}
