// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::path::{Path, PathBuf};
use std::process::Command;

use cucumber::{given, then, when};
use e2e_cucumber::loopback_http::LoopbackServer;

use crate::E2eWorld;

const ARTIFACT_REF: &str = "E2E/Atomic#direct-bin";
const ARTIFACT_BYTES: &[u8] = b"atomic-download-fixture";
const ARTIFACT_SHA256: &str = "12bc9c098e689d7eac74bf6ff8b0fdacddae2e9eba48c2eede0c46eeeb008500";

fn root(world: &E2eWorld) -> &Path {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("e2e-cucumber must live under <workspace>/tests")
        .to_path_buf()
}

fn xtask_command() -> Command {
    if let Some(binary) = std::env::var_os("ROCM_XTASK_BINARY") {
        Command::new(binary)
    } else {
        let mut command = Command::new(
            std::env::var_os("CARGO").unwrap_or_else(|| std::ffi::OsString::from("cargo")),
        );
        command.arg("xtask");
        command
    }
}

fn rocmd_binary() -> PathBuf {
    let configured = std::env::var_os("ROCM_CLI_ROCMD_BINARY").unwrap_or_else(|| {
        panic!(
            "this rocmd-backed scenario requires ROCM_CLI_ROCMD_BINARY; when using a prebuilt \
             ROCM_CLI_BINARY, provide the matching prebuilt rocmd path explicitly"
        )
    });
    let configured = PathBuf::from(configured);
    configured.canonicalize().unwrap_or_else(|error| {
        panic!(
            "failed to resolve ROCM_CLI_ROCMD_BINARY {}: {error}",
            configured.display()
        )
    })
}

fn signed_index_paths(world: &E2eWorld) -> (PathBuf, PathBuf, PathBuf) {
    let root = root(world);
    (
        root.join("recipes.json"),
        root.join("recipes.json.sig"),
        root.join("recipe-public.pem"),
    )
}

fn run_rocmd(world: &E2eWorld, args: &[&str]) -> (String, String, i32) {
    let (index, signature, public_key) = signed_index_paths(world);
    let empty_path = root(world).join("empty-path");
    std::fs::create_dir_all(&empty_path).expect("failed to create isolated PATH directory");
    let mut command = Command::new(rocmd_binary());
    command.args(args);
    world.isolate_cmd(&mut command);
    // Force the documented restricted-native fallback so this scenario has the
    // same writable data/cache view on hosts that happen to provide bubblewrap.
    command.env("PATH", empty_path);
    command.env("ROCM_CLI_MODEL_RECIPE_INDEX_PATH", index);
    command.env("ROCM_CLI_MODEL_RECIPE_INDEX_SIGNATURE_PATH", signature);
    command.env("ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH", public_key);
    let output = command.output().expect("failed to run rocmd");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        output.status.code().unwrap_or(-1),
    )
}

#[given("a signed direct-download artifact fixture")]
async fn signed_direct_download_fixture(world: &mut E2eWorld) {
    // Fail before fixture setup when a prebuilt-mode caller selected this
    // rocmd-backed scenario without supplying its explicit harness contract.
    let _ = rocmd_binary();
    let served = root(world).join("served-artifacts");
    std::fs::create_dir_all(&served).expect("failed to create served artifact directory");
    std::fs::write(served.join("artifact.bin"), ARTIFACT_BYTES)
        .expect("failed to write artifact fixture");
    let server = LoopbackServer::start(&served);
    let artifact_url = format!("{}/artifact.bin", server.base_url());

    let (index, signature, public_key) = signed_index_paths(world);
    let private_key = root(world).join("recipe-private.pem");
    let document = serde_json::json!({
        "schema_version": 1,
        "source": "e2e-atomic-download",
        "recipes": [{
            "canonical_model_id": "E2E/Atomic",
            "aliases": [],
            "task": "chat",
            "source": "signed_recipe_index",
            "revision": "main",
            "loader": "transformers",
            "trust_remote_code": false,
            "dtype": "float16",
            "device_policy": "gpu_required",
            "artifacts": [{
                "artifact_id": "direct-bin",
                "kind": "url",
                "uri": artifact_url,
                "revision": null,
                "sha256": ARTIFACT_SHA256,
                "size_bytes": ARTIFACT_BYTES.len(),
                "license": "test-only",
                "gated": false,
                "quantization": null,
                "engines": ["vllm"]
            }],
            "engine_recipes": [],
            "manual_alternatives": [],
            "featured": false,
            "chat_template_mode": "auto",
            "preferred_engines": ["vllm"],
            "warnings": []
        }]
    });
    std::fs::write(
        &index,
        serde_json::to_vec_pretty(&document).expect("failed to serialize recipe fixture"),
    )
    .expect("failed to write recipe fixture");

    let keygen = xtask_command()
        .args(["keygen", "--private-out"])
        .arg(&private_key)
        .arg("--public-out")
        .arg(&public_key)
        .current_dir(workspace_root())
        .status()
        .expect("failed to run xtask keygen");
    assert!(keygen.success(), "xtask keygen failed");
    let sign = xtask_command()
        .args(["sign", "--private-key"])
        .arg(&private_key)
        .arg("--in")
        .arg(&index)
        .arg("--out")
        .arg(&signature)
        .current_dir(workspace_root())
        .status()
        .expect("failed to run xtask sign");
    assert!(sign.success(), "xtask sign failed");
    world.artifact_server = Some(server);
}

#[given("its cache marker destination is occupied by a directory")]
async fn occupy_cache_marker(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = run_rocmd(
        world,
        &[
            "sandbox-run",
            "prefetch_artifact",
            "--artifact-ref",
            ARTIFACT_REF,
            "--allow-native-fallback",
        ],
    );
    assert_eq!(rc, 0, "marker discovery failed: {stderr}");
    let report: serde_json::Value =
        serde_json::from_str(&stdout).expect("rocmd did not print a JSON prefetch report");
    let marker = report
        .get("output")
        .and_then(|output| output.get("cache"))
        .and_then(|cache| cache.get("marker_path"))
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from)
        .expect("prefetch report did not include cache.marker_path");
    std::fs::create_dir_all(marker.join("occupied"))
        .expect("failed to occupy cache marker destination");
    std::fs::write(marker.join("occupied").join("keep"), b"keep")
        .expect("failed to seed occupied cache marker destination");
    world.artifact_marker_path = Some(marker);
}

#[when("the user approves the artifact prefetch")]
async fn approve_artifact_prefetch(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = run_rocmd(
        world,
        &[
            "sandbox-run",
            "prefetch_artifact",
            "--artifact-ref",
            ARTIFACT_REF,
            "--allow-artifact-download",
            "--artifact-max-bytes",
            "1024",
            "--allow-native-fallback",
        ],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the artifact download completes before marker publication fails")]
async fn artifact_download_completes_before_marker_publication_fails(world: &mut E2eWorld) {
    assert_ne!(
        world.cli_rc,
        Some(0),
        "prefetch unexpectedly succeeded: {}",
        world.cli_output.as_deref().unwrap_or("")
    );
    let marker = world
        .artifact_marker_path
        .as_ref()
        .expect("no marker path recorded");
    let artifact = marker.with_extension("bin");
    assert_eq!(
        std::fs::read(&artifact).expect("artifact download did not complete"),
        ARTIFACT_BYTES,
        "downloaded artifact did not match the fixture"
    );
    let stderr = world.cli_stderr.as_deref().unwrap_or("");
    assert!(
        stderr.contains("os error"),
        "prefetch did not report the expected marker-publication filesystem error: {stderr}"
    );
}

#[then("no temporary cache marker is left behind")]
async fn no_temporary_marker_remains(world: &mut E2eWorld) {
    let marker = world
        .artifact_marker_path
        .as_ref()
        .expect("no marker path recorded");
    let marker_name = marker
        .file_name()
        .expect("marker has no file name")
        .to_string_lossy();
    let legacy_stem = marker
        .file_stem()
        .expect("marker has no file stem")
        .to_string_lossy();
    let temp_prefixes = [format!("{marker_name}.tmp-"), format!("{legacy_stem}.tmp-")];
    let leftovers = std::fs::read_dir(marker.parent().expect("marker has no parent"))
        .expect("failed to inspect marker directory")
        .map(|entry| entry.expect("failed to read marker entry").file_name())
        .filter(|name| {
            let name = name.to_string_lossy();
            temp_prefixes.iter().any(|prefix| name.starts_with(prefix))
        })
        .collect::<Vec<_>>();
    assert!(
        leftovers.is_empty(),
        "failed prefetch left temporary cache markers: {leftovers:?}"
    );
}

#[then("the occupied cache marker destination remains unchanged")]
async fn occupied_marker_remains_unchanged(world: &mut E2eWorld) {
    let marker = world
        .artifact_marker_path
        .as_ref()
        .expect("no marker path recorded");
    assert!(marker.is_dir(), "occupied marker destination was removed");
    assert_eq!(
        std::fs::read(marker.join("occupied").join("keep"))
            .expect("occupied marker sentinel was removed"),
        b"keep",
        "occupied marker sentinel bytes changed"
    );
}
