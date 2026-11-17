// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for the side-by-side scenarios in `runtime_lifecycle.feature`: three
//! managed wheel runtimes for one GPU family — a ROCm 7 runtime installed before
//! the ROCm 10 layout existed, a ROCm 10 release runtime from that layout, and a
//! ROCm 10 nightly prerelease — listed, activated, uninstalled and selected by
//! the name two of them share.
//!
//! The runtimes are planted, not installed: a real install downloads several
//! GiB of wheels. Each record holds what the wheel install in `therock.rs`
//! writes for its role (see [`Role`]): a registry entry and the same manifest in
//! the folder's own `.rocm-cli-runtime.json`, with every field the writing
//! version serialized. The readiness check (`runtime_usability_status` in
//! `apps/rocm/src/runtime_registry.rs`) reads none of the files it checks and
//! runs nothing: it asks for the install folder and the probe's SDK root and bin
//! to be folders, and for the folder's local manifest, the Python executable and
//! a listed `amdhip64` and `hipblas` library to be files, so stub files stand in
//! for all of them.
//!
//! The local manifest is a full copy rather than the `{}` some other fixtures
//! write, because uninstall parses it before deciding the folder's fate
//! (`local_runtime_manifest_matches` in `apps/rocm/src/runtime_uninstall.rs`):
//! one that does not parse as a manifest fails the uninstall with a parse
//! error, and only one naming the same key, id and folder as the registry entry
//! lets the folder be removed.
//!
//! Every command runs with the startup update check disabled. With a runtime
//! installed, a command otherwise first asks the package index whether that
//! runtime has an update, which is network traffic and cache writes these
//! scenarios are not about.

use std::path::{Path, PathBuf};

use cucumber::{given, then, when};
use e2e_cucumber::cli_failure_report;
use e2e_cucumber::fixture_binary::fixture_target_name;
use sha2::{Digest as _, Sha256};

use super::comfyui_steps::disable_startup_update_check;
use super::runtime_lifecycle_steps::{
    active_marker_path, config_path, read_json_if_present, registry_dir,
};
use crate::E2eWorld;

/// Shared by the ROCm 7 and the ROCm 10 release runtimes: `rocm install sdk`
/// derives the id from the channel and GPU family alone, so both carry it, and
/// only their keys tell them apart.
const SHARED_RELEASE_RUNTIME_ID: &str = "therock-release:gfx94X-dcgpu";
const NIGHTLY_RUNTIME_ID: &str = "therock-nightly:gfx94X-dcgpu";
/// The family `rocm install sdk` resolves for a host reporting `gfx942`.
const FAMILY: &str = "gfx94X-dcgpu";
const DEVICE_TARGET: &str = "gfx942";

/// One of the three planted runtimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// ROCm 7.14.0 from the canonical release stream, installed by the CLI as
    /// it stood just before the ROCm 10 layout was added (commit `a6fa76db`'s
    /// parent). That install already recorded a wheel composition and keyed the
    /// runtime by its fingerprint, but wrote neither `source_layout_generation`
    /// nor `devel`, which did not exist yet, and always installed the compiler
    /// toolchain.
    Rocm7,
    /// ROCm 10.0.0, a stable release and so on the release channel only, from
    /// the ROCm 10 (`next`) layout, installed by the current CLI without the
    /// toolchain (now opt-in).
    Rocm10Release,
    /// The ROCm 10.1 nightly prerelease `10.1.0a20260822`, the last nightly the
    /// canonical layout published: nightlies from 2026-08-23 on come from the
    /// `next` layout instead. So this one records the canonical layout
    /// (`multi-arch-v2`), installed by the current CLI without the toolchain.
    Rocm10Nightly,
}

impl Role {
    const ALL: [Self; 3] = [Self::Rocm7, Self::Rocm10Release, Self::Rocm10Nightly];

    const fn channel(self) -> &'static str {
        match self {
            Self::Rocm7 | Self::Rocm10Release => "release",
            Self::Rocm10Nightly => "nightly",
        }
    }

    const fn runtime_id(self) -> &'static str {
        match self {
            Self::Rocm7 | Self::Rocm10Release => SHARED_RELEASE_RUNTIME_ID,
            Self::Rocm10Nightly => NIGHTLY_RUNTIME_ID,
        }
    }

    const fn version(self) -> &'static str {
        match self {
            Self::Rocm7 => "7.14.0",
            Self::Rocm10Release => "10.0.0",
            Self::Rocm10Nightly => "10.1.0a20260822",
        }
    }

    /// The layout generation the install recorded in its composition.
    const fn layout_generation(self) -> &'static str {
        match self {
            Self::Rocm7 | Self::Rocm10Nightly => "multi-arch-v2",
            Self::Rocm10Release => "next-v1",
        }
    }

    /// The pip index the install read from, which it records both as
    /// `index_url` and as `selected_artifact_url`. The hosts stand in for the
    /// published ones the way `ROCM_CLI_THEROCK_*_PIP_BASE` overrides do; the
    /// install records whichever base it resolved.
    const fn index_url(self) -> &'static str {
        match self {
            Self::Rocm7 => "https://example.invalid/rocm/whl-multi-arch",
            Self::Rocm10Release => "https://example.invalid/rocm/whl-next",
            Self::Rocm10Nightly => "https://example.invalid/nightlies/whl-multi-arch",
        }
    }

    /// Whether the install included the compiler toolchain: always before it
    /// became opt-in, and not by default since.
    const fn includes_devel(self) -> bool {
        matches!(self, Self::Rocm7)
    }

    /// `torch`, `torchvision` and `torchaudio` as the install pinned them.
    const fn torch_stack(self) -> [&'static str; 3] {
        match self {
            Self::Rocm7 => [
                "2.11.0+rocm7.14.0",
                "0.26.0+rocm7.14.0",
                "2.11.0+rocm7.14.0",
            ],
            Self::Rocm10Release => [
                "2.13.0+rocm10.0.0",
                "0.28.0+rocm10.0.0",
                "2.11.0.2+rocm10.0.0",
            ],
            Self::Rocm10Nightly => [
                "2.12.0+rocm10.1.0a20260822",
                "0.27.0+rocm10.1.0a20260822",
                "2.12.0+rocm10.1.0a20260822",
            ],
        }
    }

    /// The exact specs handed to `uv`, as `therock_pip_package_specs` builds
    /// them.
    fn package_specs(self) -> Vec<String> {
        let device = format!("device-{DEVICE_TARGET}");
        let extras = if self.includes_devel() {
            "libraries,devel"
        } else {
            "libraries"
        };
        let [torch, torchvision, torchaudio] = self.torch_stack();
        vec![
            format!("rocm[{extras},{device}]=={}", self.version()),
            format!("torch[{device}]=={torch}"),
            format!("torchvision[{device}]=={torchvision}"),
            format!("torchaudio=={torchaudio}"),
        ]
    }

    /// The key `wheel_runtime_key` in `apps/rocm/src/therock.rs` derives: the
    /// channel and version, then the first 16 hex digits of a SHA-256 over the
    /// composition. The derivation is the same before and after the ROCm 10
    /// layout, so the layout generation inside the composition is what keeps a
    /// ROCm 10 key from ever colliding with a canonical one. Planting checks the
    /// result against [`Self::expected_key`], the literal keys the unit test
    /// `wheel_runtime_key_matches_the_side_by_side_e2e_fixture` in `therock.rs`
    /// derives with the production code from the same inputs.
    fn key(self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.layout_generation().as_bytes());
        for spec in self.package_specs() {
            hasher.update([0]);
            hasher.update(spec.as_bytes());
        }
        hasher.update([0]);
        hasher.update(b"rocm-sdk-target");
        hasher.update([0]);
        hasher.update(DEVICE_TARGET.as_bytes());
        let fingerprint = format!("{:x}", hasher.finalize());
        format!(
            "{}-wheel-multi-arch-{}-{}",
            self.channel(),
            self.version(),
            &fingerprint[..16]
        )
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
    }

    /// The key [`Self::key`] must produce: the literals
    /// `wheel_runtime_key_matches_the_side_by_side_e2e_fixture` in `therock.rs`
    /// gets from the production `wheel_runtime_key` for the same inputs. The
    /// unit test ties these literals to the production derivation, and
    /// planting ties them to this fixture, so neither side can drift alone.
    const fn expected_key(self) -> &'static str {
        match self {
            Self::Rocm7 => "release-wheel-multi-arch-7-14-0-992a08553d8be8f0",
            Self::Rocm10Release => "release-wheel-multi-arch-10-0-0-3d0e256f664a960d",
            Self::Rocm10Nightly => "nightly-wheel-multi-arch-10-1-0a20260822-95e3a823499c530b",
        }
    }

    /// What `rocm runtimes list` must report for the compiler toolchain.
    const fn toolchain(self) -> &'static str {
        if self.includes_devel() {
            "included"
        } else {
            "excluded"
        }
    }
}

/// A planted runtime and the bytes of its two install records as planted.
#[derive(Debug)]
struct PlantedRuntime {
    role: Role,
    key: String,
    install_root: PathBuf,
    registry_path: PathBuf,
    registry_bytes: Vec<u8>,
    local_manifest_bytes: Vec<u8>,
}

impl PlantedRuntime {
    fn local_manifest_path(&self) -> PathBuf {
        self.install_root.join(".rocm-cli-runtime.json")
    }
}

/// What one `rocm runtimes activate` left behind.
#[derive(Debug)]
struct Activation {
    role: Role,
    in_use: InUse,
}

/// What the CLI's two records of the runtime in use name: the config
/// (`config.json`) and the active-runtime marker (`<data>/runtimes/active.json`).
/// `None` throughout means the record names nothing, or does not exist.
#[derive(Debug, PartialEq, Eq)]
struct InUse {
    config_active: Option<String>,
    config_previous: Option<String>,
    config_default_id: Option<String>,
    marker_key: Option<String>,
    marker_id: Option<String>,
    marker_previous_key: Option<String>,
    marker_previous_id: Option<String>,
}

/// Per-scenario state, held on the World.
#[derive(Debug)]
pub struct SideBySide {
    runtimes: Vec<PlantedRuntime>,
    activations: Vec<Activation>,
}

impl SideBySide {
    fn runtime(&self, role: Role) -> &PlantedRuntime {
        self.runtimes
            .iter()
            .find(|runtime| runtime.role == role)
            .unwrap_or_else(|| panic!("{role:?} runtime was not planted"))
    }

    fn key(&self, role: Role) -> &str {
        &self.runtime(role).key
    }
}

const fn state(world: &E2eWorld) -> &SideBySide {
    world
        .side_by_side
        .as_ref()
        .expect("no side-by-side runtimes were planted in this scenario")
}

fn key(world: &E2eWorld, role: Role) -> String {
    state(world).key(role).to_owned()
}

fn write_file(path: &Path, contents: &[u8]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", parent.display()));
    }
    std::fs::write(path, contents)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}

/// The `site-packages` of a venv at `install_root`, as Python reports it.
fn site_packages(install_root: &Path) -> PathBuf {
    if cfg!(windows) {
        install_root.join("Lib").join("site-packages")
    } else {
        install_root
            .join("lib")
            .join("python3.12")
            .join("site-packages")
    }
}

/// Where `rocm_sdk.find_libraries` finds `shortname` inside a package root.
fn library_path(package_root: &Path, shortname: &str) -> PathBuf {
    if cfg!(windows) {
        package_root.join("bin").join(format!("{shortname}.dll"))
    } else {
        package_root.join("lib").join(format!("lib{shortname}.so"))
    }
}

/// The `rocm_sdk` probe `rocm install sdk` records, in the shape the probe
/// script in `apps/rocm/src/therock/probe.rs` prints: every package root's `bin`
/// and `lib` folders, the venv's scripts folder, and an SDK root that is the
/// `devel` package when the toolchain is installed and the first package root
/// otherwise, with a CMake path only for `devel`. The script also lists the
/// installed `rocm*` distributions under `packages`; that list is left empty
/// here because nothing on these paths (listing, activating, uninstalling)
/// reads it. Two libraries stand for the full list, being the two the
/// readiness check asks for. Creates the files and folders the record names.
fn plant_rocm_sdk_probe(role: Role, install_root: &Path, scripts: &Path) -> serde_json::Value {
    let site_packages = site_packages(install_root);
    let core = site_packages.join("_rocm_sdk_core");
    let libraries = site_packages.join(format!("_rocm_sdk_libraries_{FAMILY}"));
    let devel = site_packages.join("_rocm_sdk_devel");
    let mut runtime_roots = vec![core.clone(), libraries.clone()];
    if role.includes_devel() {
        runtime_roots.insert(0, devel);
    }
    let root = runtime_roots[0].clone();
    let mut bin_paths: Vec<PathBuf> = runtime_roots.iter().map(|root| root.join("bin")).collect();
    bin_paths.push(scripts.to_path_buf());
    let library_paths: Vec<PathBuf> = runtime_roots
        .iter()
        .flat_map(|root| [root.join("bin"), root.join("lib")])
        .collect();
    for dir in &library_paths {
        std::fs::create_dir_all(dir)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", dir.display()));
    }
    let amdhip = library_path(&core, "amdhip64");
    let hipblas = library_path(&libraries, "hipblas");
    for file in [&amdhip, &hipblas] {
        write_file(file, b"stub");
    }
    serde_json::json!({
        "import_ok": true,
        "rocm_sdk_version": role.version(),
        "site_packages": site_packages,
        "root_path": root,
        "bin_path": root.join("bin"),
        "cmake_path": role.includes_devel().then(|| root.join("lib").join("cmake")),
        "runtime_roots": runtime_roots,
        "bin_paths": bin_paths,
        "library_paths": library_paths,
        "default_target_family": DEVICE_TARGET,
        "available_target_families": [DEVICE_TARGET],
        "resolved_target_family": DEVICE_TARGET,
        "packages": [],
        "library_shortnames": ["amdhip64", "hipblas"],
        "resolved_libraries": [
            {"shortname": "amdhip64", "paths": [amdhip]},
            {"shortname": "hipblas", "paths": [hipblas]},
        ],
        "error": null,
    })
}

/// The manifest the wheel install writes for `role`, with every field the
/// writing version serialized, and the files its readiness check asks for.
fn plant_manifest(world: &E2eWorld, role: Role, key: &str) -> (PathBuf, serde_json::Value) {
    let data_dir = world.app_paths().data_dir;
    let install_root = data_dir.join("runtimes").join("wheel").join(key);
    let scripts = install_root.join(if cfg!(windows) { "Scripts" } else { "bin" });
    let python = scripts.join(fixture_target_name("python"));
    write_file(&python, b"stub");
    // The interpreter the venv was created from: ROCm CLI's managed Python.
    let python_launcher = data_dir
        .join("tools")
        .join("python")
        .join(fixture_target_name("python"));
    let [torch, ..] = role.torch_stack();

    let mut manifest = serde_json::json!({
        "runtime_key": key,
        "runtime_id": role.runtime_id(),
        "channel": role.channel(),
        "format": "wheel",
        "family": FAMILY,
        "family_source": "host",
        "version": role.version(),
        "install_root": install_root,
        "selected_artifact_url": role.index_url(),
        "index_url": role.index_url(),
        "tarball_file_name": null,
        "python_launcher": python_launcher,
        "python_executable": python,
        "pip_cache_dir": null,
        "rocm_sdk": plant_rocm_sdk_probe(role, &install_root, &scripts),
        "sdk_torch": torch,
        "wheel_composition": {
            "source_layout_generation": role.layout_generation(),
            "package_specs": role.package_specs(),
            "rocm_sdk_target": DEVICE_TARGET,
        },
        "read_only": false,
        "imported_from": null,
        "installed_at_unix_ms": match role {
            Role::Rocm7 => 1_757_000_000_000u64,
            Role::Rocm10Release => 1_762_000_000_000,
            Role::Rocm10Nightly => 1_762_500_000_000,
        },
    });
    if role != Role::Rocm7 {
        let fields = manifest
            .as_object_mut()
            .expect("the manifest is a JSON object");
        fields.insert(
            "source_layout_generation".into(),
            role.layout_generation().into(),
        );
        fields.insert("devel".into(), role.includes_devel().into());
    }
    (install_root, manifest)
}

/// Runs `rocm` with the startup update check disabled. `command_env` is
/// consumed by each run, so it is set again for every one.
fn run(world: &mut E2eWorld, args: &[&str]) -> (String, String, i32) {
    disable_startup_update_check(world);
    crate::run_rocm_with_scenario_env(world, args)
}

/// [`run`], recording the outcome on the World for the Then steps.
fn run_recorded(world: &mut E2eWorld, args: &[&str]) {
    let (stdout, stderr, rc) = run(world, args);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// [`run`], failing the step with the full diagnostic bundle unless it exits 0.
/// For actions that only set the stage or have their outcome checked on disk.
fn run_ok(world: &mut E2eWorld, args: &[&str]) {
    let (stdout, stderr, rc) = run(world, args);
    assert!(
        rc == 0,
        "{}",
        cli_failure_report(args, rc, &stdout, &stderr)
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

fn string_field(value: Option<&serde_json::Value>, field: &str) -> Option<String> {
    value
        .and_then(|value| value.get(field))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// What the config and the active-runtime marker name now.
fn in_use(world: &E2eWorld) -> InUse {
    let config = read_json_if_present(&config_path(world));
    let marker = read_json_if_present(&active_marker_path(world));
    InUse {
        config_active: string_field(config.as_ref(), "active_runtime_key"),
        config_previous: string_field(config.as_ref(), "previous_runtime_key"),
        config_default_id: string_field(config.as_ref(), "default_runtime_id"),
        marker_key: string_field(marker.as_ref(), "runtime_key"),
        marker_id: string_field(marker.as_ref(), "runtime_id"),
        marker_previous_key: string_field(marker.as_ref(), "previous_runtime_key"),
        marker_previous_id: string_field(marker.as_ref(), "previous_runtime_id"),
    }
}

/// What both records must name while `role` is in use with `previous` as its
/// rollback target.
fn expected_in_use(world: &E2eWorld, role: Role, previous: Option<Role>) -> InUse {
    let role_key = Some(key(world, role));
    let previous_key = previous.map(|previous| key(world, previous));
    let previous_id = previous.map(|previous| previous.runtime_id().to_owned());
    InUse {
        config_active: role_key.clone(),
        config_previous: previous_key.clone(),
        config_default_id: Some(role.runtime_id().to_owned()),
        marker_key: role_key,
        marker_id: Some(role.runtime_id().to_owned()),
        marker_previous_key: previous_key,
        marker_previous_id: previous_id,
    }
}

fn activate(world: &mut E2eWorld, role: Role) {
    let key = key(world, role);
    run_ok(world, &["runtimes", "activate", &key]);
    let in_use = in_use(world);
    world
        .side_by_side
        .as_mut()
        .expect("no side-by-side runtimes were planted in this scenario")
        .activations
        .push(Activation { role, in_use });
}

fn assert_in_use(world: &E2eWorld, role: Role, previous: Option<Role>) {
    assert_eq!(
        in_use(world),
        expected_in_use(world, role, previous),
        "the config and the active-runtime marker must name the {role:?} runtime as in use, \
         with {previous:?} as its rollback target"
    );
}

/// Fails unless `runtime`'s folder is still there and both its install records
/// hold exactly the bytes that were planted.
fn assert_records_untouched(runtime: &PlantedRuntime) {
    let role = runtime.role;
    assert!(
        runtime.install_root.is_dir(),
        "the {role:?} runtime's folder was removed: {}",
        runtime.install_root.display()
    );
    for (path, planted) in [
        (runtime.registry_path.clone(), &runtime.registry_bytes),
        (runtime.local_manifest_path(), &runtime.local_manifest_bytes),
    ] {
        let now = std::fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "the {role:?} runtime's record {} is gone: {error}",
                path.display()
            )
        });
        assert!(
            &now == planted,
            "the {role:?} runtime's record {} was rewritten.\nplanted:\n{}\nnow:\n{}",
            path.display(),
            String::from_utf8_lossy(planted),
            String::from_utf8_lossy(&now)
        );
    }
}

fn output_text(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or_default(),
        world.cli_stderr.as_deref().unwrap_or_default()
    )
}

// ── Given ──────────────────────────────────────────────────────────

#[given(
    "a ROCm 7 runtime installed before the ROCm 10 layout existed, a ROCm 10 release runtime and a ROCm 10 nightly runtime, all for one GPU family"
)]
async fn plant_side_by_side_runtimes(world: &mut E2eWorld) {
    let registry = registry_dir(world);
    let runtimes = Role::ALL
        .into_iter()
        .map(|role| {
            let key = role.key();
            assert_eq!(
                key,
                role.expected_key(),
                "the {role:?} fixture no longer derives the key the production code \
                 gives these inputs; update `Role::expected_key` together with \
                 `wheel_runtime_key_matches_the_side_by_side_e2e_fixture` in \
                 apps/rocm/src/therock.rs"
            );
            let (install_root, manifest) = plant_manifest(world, role, &key);
            let manifest = serde_json::to_vec_pretty(&manifest)
                .expect("failed to serialize the runtime manifest");
            let registry_path = registry.join(format!("{key}.json"));
            write_file(&install_root.join(".rocm-cli-runtime.json"), &manifest);
            write_file(&registry_path, &manifest);
            PlantedRuntime {
                role,
                key,
                install_root,
                registry_path,
                registry_bytes: manifest.clone(),
                local_manifest_bytes: manifest,
            }
        })
        .collect();
    world.side_by_side = Some(SideBySide {
        runtimes,
        activations: Vec::new(),
    });
}

#[given("the nightly runtime is in use after the ROCm 10 release runtime")]
async fn nightly_in_use_after_rocm10_release(world: &mut E2eWorld) {
    activate(world, Role::Rocm10Release);
    activate(world, Role::Rocm10Nightly);
    assert_in_use(world, Role::Rocm10Nightly, Some(Role::Rocm10Release));
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user lists the installed runtimes")]
async fn list_installed_runtimes(world: &mut E2eWorld) {
    run_recorded(world, &["runtimes", "list"]);
}

#[when("the user activates each of the three runtimes in turn")]
async fn activate_each_in_turn(world: &mut E2eWorld) {
    for role in Role::ALL {
        activate(world, role);
    }
}

#[when("the user uninstalls the ROCm 10 release runtime")]
async fn uninstall_rocm10_release(world: &mut E2eWorld) {
    let key = key(world, Role::Rocm10Release);
    run_ok(world, &["runtimes", "uninstall", &key, "--yes"]);
}

#[when("the user uninstalls the runtime in use")]
async fn uninstall_runtime_in_use(world: &mut E2eWorld) {
    let in_use = in_use(world)
        .config_active
        .expect("the config names no runtime in use to uninstall");
    run_ok(world, &["runtimes", "uninstall", &in_use, "--yes"]);
}

#[when("the user activates the ROCm 7 runtime")]
async fn activate_rocm7(world: &mut E2eWorld) {
    activate(world, Role::Rocm7);
}

#[when("the user activates a runtime by the name the two release runtimes share")]
async fn activate_by_shared_name(world: &mut E2eWorld) {
    run_recorded(world, &["runtimes", "activate", SHARED_RELEASE_RUNTIME_ID]);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("all three runtimes are listed as ready, each under its own key")]
async fn all_three_listed_ready(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    assert_eq!(
        world.cli_rc,
        Some(0),
        "`rocm runtimes list` failed:\n{}",
        output_text(world)
    );
    let rows: Vec<&str> = stdout
        .lines()
        .filter(|line| line.contains(" runtime_id="))
        .collect();
    assert_eq!(
        rows.len(),
        3,
        "expected one row per planted runtime:\n{stdout}"
    );
    for role in Role::ALL {
        let prefix = format!("{} runtime_id={} ", key(world, role), role.runtime_id());
        let matching: Vec<&&str> = rows.iter().filter(|row| row.contains(&prefix)).collect();
        let [row] = matching.as_slice() else {
            panic!("expected exactly one row for the {role:?} runtime ({prefix}):\n{stdout}");
        };
        assert!(
            row.contains(" status=ready "),
            "the {role:?} runtime must be listed as ready: {row}"
        );
        assert!(
            row.ends_with(&format!(" toolchain={}", role.toolchain())),
            "the {role:?} runtime must be listed with toolchain={}: {row}",
            role.toolchain()
        );
    }
}

#[then("each one becomes the runtime in use in turn")]
async fn each_becomes_in_use(world: &mut E2eWorld) {
    let activations = &state(world).activations;
    let order: Vec<Role> = activations
        .iter()
        .map(|activation| activation.role)
        .collect();
    assert_eq!(order, Role::ALL, "the activations did not all run");
    let mut previous = None;
    for activation in activations {
        assert_eq!(
            activation.in_use,
            expected_in_use(world, activation.role, previous),
            "activating the {:?} runtime did not make it the one in use, with {previous:?} \
             as its rollback target",
            activation.role
        );
        previous = Some(activation.role);
    }
}

#[then("none of the three runtimes' install records was rewritten")]
async fn no_record_rewritten(world: &mut E2eWorld) {
    for runtime in &state(world).runtimes {
        assert_records_untouched(runtime);
    }
}

#[then("that runtime's folder and record are gone")]
async fn rocm10_release_gone(world: &mut E2eWorld) {
    let runtime = state(world).runtime(Role::Rocm10Release);
    assert!(
        !runtime.install_root.exists(),
        "the uninstalled runtime's folder is still there: {}\n{}",
        runtime.install_root.display(),
        output_text(world)
    );
    assert!(
        !runtime.registry_path.exists(),
        "the uninstalled runtime's registry entry is still there: {}",
        runtime.registry_path.display()
    );
}

#[then("the other two runtimes keep their folders and unchanged records")]
async fn others_untouched(world: &mut E2eWorld) {
    let state = state(world);
    for role in [Role::Rocm7, Role::Rocm10Nightly] {
        assert_records_untouched(state.runtime(role));
    }
}

#[then("the nightly runtime is still in use, with no rollback target left")]
async fn nightly_still_in_use(world: &mut E2eWorld) {
    assert_in_use(world, Role::Rocm10Nightly, None);
}

#[then("no runtime is in use")]
async fn no_runtime_in_use(world: &mut E2eWorld) {
    let nothing = InUse {
        config_active: None,
        config_previous: None,
        config_default_id: None,
        marker_key: None,
        marker_id: None,
        marker_previous_key: None,
        marker_previous_id: None,
    };
    assert_eq!(
        in_use(world),
        nothing,
        "neither the config nor the active-runtime marker may name a runtime in use or a \
         rollback target:\n{}",
        output_text(world)
    );
}

#[then("the ROCm 7 runtime is the runtime in use")]
async fn rocm7_in_use(world: &mut E2eWorld) {
    assert_in_use(world, Role::Rocm7, None);
}

#[then("the CLI refuses and names both release runtimes")]
async fn refused_naming_both_release_runtimes(world: &mut E2eWorld) {
    let text = output_text(world);
    assert_ne!(
        world.cli_rc,
        Some(0),
        "a name shared by two runtimes must be refused:\n{text}"
    );
    for role in [Role::Rocm7, Role::Rocm10Release] {
        let key = key(world, role);
        assert!(
            text.contains(&key),
            "the refusal must name the {role:?} runtime ({key}):\n{text}"
        );
    }
    let nightly = key(world, Role::Rocm10Nightly);
    assert!(
        !text.contains(&nightly),
        "the refusal must not name the nightly runtime, which does not share the name:\n{text}"
    );
    // The way out it names works: runtime-lifecycle-19's "each one becomes the
    // runtime in use in turn" runs `rocm runtimes activate <runtime_key>` for
    // each of these runtimes (`activate`, above) and checks it takes effect.
    assert!(
        text.contains("activate one by runtime_key"),
        "the refusal must say to activate one of them by its key:\n{text}"
    );
}
