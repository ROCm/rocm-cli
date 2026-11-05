// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `Examination` builders shared by the tests of more than one `diagnose` module.

use crate::examine::{Examination, Gpu, WslFacts};

pub(super) fn linux_base() -> Examination {
    Examination {
        os_family: "linux".to_owned(),
        ..Examination::default()
    }
}

/// An APU + discrete pairing, on the OS family given, that fires fix-9.
pub(super) fn igpu_dgpu_host(os_family: &str) -> Examination {
    Examination {
        os_family: os_family.to_owned(),
        has_apu: true,
        has_discrete_amd: true,
        gpus: vec![
            Gpu {
                gfx_target: "gfx1103".to_owned(),
                is_amd: true,
                is_apu: Some(true),
                ..Gpu::default()
            },
            Gpu {
                gfx_target: "gfx1100".to_owned(),
                is_amd: true,
                is_apu: Some(false),
                ..Gpu::default()
            },
        ],
        ..Examination::default()
    }
}

/// A WSL2 host with the GPU stack fully in place.
pub(super) fn wsl_base() -> Examination {
    let mut e = linux_base();
    e.is_wsl = true;
    e.distro_id = "ubuntu".to_owned();
    e.distro_version = "24.04".to_owned();
    e.kernel_release = "6.6.87.2-microsoft-standard-WSL2".to_owned();
    e.wsl = Some(WslFacts {
        version: 2,
        dxg_device: true,
        dxcore: true,
        wsl_lib_dir: true,
        librocdxg: true,
        rocdxg_dids: true,
        ldconfig_librocdxg: Some(true),
        rocminfo: true,
        rocm_sees_gpu: Some(true),
        distro_supported: Some(true),
        host_driver_version: Some("32.0.12033.1030".to_owned()),
        host_reachable: true,
        locally_probed: true,
    });
    // The shared cross-platform checks read these, and on a real WSL host
    // `probe_wsl` fills them from the WSL facts. A fixture that left them at
    // their defaults would be a healthier machine than any real one.
    e.rocminfo_present = true;
    e.rocminfo_status = "ok".to_owned();
    e
}

pub(super) fn windows_driver_exam(version: &str, date: &str) -> Examination {
    Examination {
        os_family: "windows".to_owned(),
        adrenalin_version: version.to_owned(),
        adrenalin_date: date.to_owned(),
        ..Examination::default()
    }
}
