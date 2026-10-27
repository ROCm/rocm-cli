// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Given steps that describe the machine a scenario runs on by planting a
//! simulated one (see `e2e_cucumber::simulated_host`). The CLI then reads its
//! hardware from that machine on every lane, so the scenario behaves the same
//! on a GitHub-hosted runner, a GPU box, and a WSL2 laptop.

use std::path::Path;

use cucumber::{given, then, when};
use e2e_cucumber::simulated_host::{
    SimulatedHost, bare_metal_layout_drift, lspci_drift, wsl_layout_drift,
};

use crate::E2eWorld;

#[given("a machine with an AMD Instinct GPU")]
async fn instinct_machine(world: &mut E2eWorld) {
    world.plant_simulated_host(SimulatedHost::instinct_mi300x(1));
}

#[given("a machine with two AMD Instinct GPUs")]
async fn two_gpu_instinct_machine(world: &mut E2eWorld) {
    world.plant_simulated_host(SimulatedHost::instinct_mi300x(2));
}

#[given("a machine with eight AMD Instinct GPUs")]
async fn eight_gpu_instinct_machine(world: &mut E2eWorld) {
    world.plant_simulated_host(SimulatedHost::instinct_mi300x(8));
}

#[given("a Linux machine with no AMD GPU")]
async fn machine_without_gpu(world: &mut E2eWorld) {
    world.plant_simulated_host(SimulatedHost::bare_metal_without_gpu());
}

/// Bare-metal Linux with the in-tree amdgpu driver and one GPU — the premise of
/// the diagnose catalog's bare-metal half, which a WSL2 host never runs.
#[given("a bare-metal Linux machine with an AMD GPU")]
async fn bare_metal_machine(world: &mut E2eWorld) {
    world.plant_simulated_host(SimulatedHost::instinct_mi300x(1));
}

/// A WSL2 distribution whose ROCm passthrough to the Windows host's GPU is set
/// up. `/dev/dxg` and the kernel string are what the CLI recognises WSL by, so
/// this is the only way to put a WSL premise in front of it off a WSL lane.
#[given("a WSL machine with an AMD GPU passed through")]
async fn wsl_machine(world: &mut E2eWorld) {
    world.plant_simulated_host(SimulatedHost::wsl2_strix_halo());
}

/// Plant `host` somewhere this scenario's `rocm` invocations never look: the
/// comparison is between the fixture and the real machine, so the real machine
/// must stay what the probes see.
fn planted_for_comparison(world: &E2eWorld, host: &SimulatedHost) -> std::path::PathBuf {
    let root = world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .join("simulated-for-comparison");
    host.plant(&root, &root.join("unused-tool"), std::ffi::OsStr::new(""))
        .unwrap_or_else(|e| panic!("failed to plant the simulated machine: {e}"));
    root
}

#[when("the simulated GPU machine is compared with this machine's kernel")]
async fn compare_bare_metal(world: &mut E2eWorld) {
    let simulated = planted_for_comparison(world, &SimulatedHost::instinct_mi300x(1));
    let mut drift = bare_metal_layout_drift(Path::new("/"), &simulated);
    match std::process::Command::new("lspci")
        .args(["-nn", "-D"])
        .output()
    {
        Ok(output) if output.status.success() => {
            drift.extend(lspci_drift(&String::from_utf8_lossy(&output.stdout)));
        }
        // The product enumerates PCI GPUs with this very command, so a GPU host
        // without it is a premise to report, not a comparison to skip.
        _ => drift.push("`lspci -nn -D` could not be run on this machine".to_owned()),
    }
    world.layout_drift = Some(drift);
}

#[when("the simulated WSL machine is compared with this machine")]
async fn compare_wsl(world: &mut E2eWorld) {
    world.layout_drift = Some(wsl_layout_drift(Path::new("/")));
}

#[then("the simulated machine is laid out the same way")]
async fn assert_no_layout_drift(world: &mut E2eWorld) {
    let drift = world.layout_drift.as_ref().expect("no comparison was made");
    assert!(
        drift.is_empty(),
        "the simulated machine no longer matches this real one, so scenarios on it can pass \
         while the CLI fails on hardware — update tests/e2e-cucumber/src/simulated_host.rs:\n  {}",
        drift.join("\n  ")
    );
}
