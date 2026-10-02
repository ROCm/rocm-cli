// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Given steps that describe the machine a scenario runs on by planting a
//! simulated one (see `e2e_cucumber::simulated_host`). The CLI then reads its
//! hardware from that machine on every lane, so the scenario behaves the same
//! on a GitHub-hosted runner, a GPU box, and a WSL2 laptop.

use cucumber::given;
use e2e_cucumber::simulated_host::SimulatedHost;

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
