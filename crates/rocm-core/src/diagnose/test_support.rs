// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `Examination` builders, shell stubs and a symptom, shared by the tests of
//! more than one `diagnose` module.

use crate::diagnose::SymptomEngine;
use crate::examine::{Examination, Gpu, WslFacts};

/// Every `SymptomEngine` variant, for tests that must cover each one. The
/// exhaustive `match` in [`every_engine`] stops compiling when a variant is
/// added and not listed here.
pub(super) const ALL_ENGINES: [SymptomEngine; 1] = [SymptomEngine::Vllm];

/// [`ALL_ENGINES`], checked against the enum by an exhaustive `match`.
pub(super) fn every_engine() -> [SymptomEngine; 1] {
    for engine in ALL_ENGINES {
        match engine {
            SymptomEngine::Vllm => {}
        }
    }
    ALL_ENGINES
}

/// An `Examination` of the given catalog platform family (`linux`, `wsl` or
/// `windows`), with nothing else set.
pub(super) fn host_of_family(family: &str) -> Examination {
    match family {
        "linux" => linux_base(),
        "wsl" => wsl_base(),
        "windows" => Examination {
            os_family: "windows".to_owned(),
            ..Examination::default()
        },
        other => panic!("no test host for platform family {other:?}"),
    }
}

/// A paste carrying text no catalog message contains: an accented path
/// (two-byte UTF-8), a box-drawing character (three bytes, what a `rich`
/// traceback frames itself with) and an emoji beyond the Basic Multilingual
/// Plane (four bytes, a surrogate pair once escaped). Used where a test needs
/// to know the `--json` output is ASCII because it was escaped, not because the
/// input happened to be.
pub(super) const NON_ASCII_PASTE: &str = "torch.OutOfMemoryError: HIP out of memory. \
     Tried to allocate 2.00 GiB\n\u{2502} File \"/home/jos\u{e9}/caf\u{e9}/serve.py\" \u{1f525}";

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
        librocdxg_on_ld_library_path: false,
        librocdxg_beside_runtime: false,
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

/// Writes each `(name, script)` as an executable in a new directory, to be
/// the whole `PATH` of a shell that runs a Linux verify line.
#[cfg(unix)]
pub(super) fn stub_bin_dir(stubs: &[(&str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("the stub directory is created");
    for (name, script) in stubs {
        let path = dir.path().join(name);
        crate::fake_executable::write_fake_executable(&path, format!("#!/bin/sh\n{script}\n"))
            .expect("a stub is planted");
    }
    dir
}

/// Runs one printed line in `/bin/sh` whose `PATH` is only `bin` and whose
/// environment is only `PATH` and `env`. Returns whether it exited 0, and
/// its stdout and stderr.
#[cfg(unix)]
pub(super) fn run_in_sh(
    line: &str,
    bin: &std::path::Path,
    env: &[(&str, &str)],
) -> (bool, String, String) {
    let out = std::process::Command::new("/bin/sh")
        .arg("-c")
        .arg(line)
        .env_clear()
        .env("PATH", bin)
        .envs(env.iter().copied())
        .output()
        .expect("/bin/sh runs");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}
