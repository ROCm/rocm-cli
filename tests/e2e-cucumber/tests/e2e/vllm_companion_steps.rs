// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Offline CLI coverage of strict companion recovery and SDK torch retention.
//! The stand-in uv records installed versions separately from its argv log.

use std::path::PathBuf;

use cucumber::{given, then, when};

use crate::E2eWorld;

const KEY: &str = "release-wheel-gfx1151-e2e-companion";
const INDEX: &str = "https://example.invalid/sdk-wheels";

fn runtime(world: &E2eWorld) -> PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("isolated root")
        .path()
        .join("companion-runtime")
}

#[given(
    regex = r"^a ROCm (10\.[12]\.0) runtime whose companion index (also serves plain wheels|has no plain wheels)$"
)]
async fn plant(world: &mut E2eWorld, sdk: String, availability: String) {
    let root = world.isolated_root.as_ref().expect("isolated root").path();
    let rt = runtime(world);
    let bin = rt.join("bin");
    let shim = rt.join("shim");
    std::fs::create_dir_all(&bin).expect("runtime bin");
    std::fs::create_dir_all(&shim).expect("shim directory");
    std::fs::create_dir_all(rt.join("lib/python3.14/site-packages"))
        .expect("venv package directory");
    let (torch, vision, audio, vllm, other) = if sdk == "10.1.0" {
        ("2.13.0", "0.28.0", "2.11.0.2", "0.29.1", "10.2.0")
    } else {
        ("2.14.0", "0.29.2", "2.11.1", "0.32.1", "10.1.0")
    };
    let stack = format!(
        "torch=={torch}+rocm{sdk}\ntorchvision=={vision}+rocm{sdk}\ntorchaudio=={audio}+rocm{sdk}\n"
    );
    std::fs::write(rt.join("sdk-stack"), &stack).expect("SDK stack");
    rocm_core::fake_executable::write_fake_executable(
        &bin.join("python"),
        format!(
            r#"#!/bin/sh
case "$*" in
  *sys.version_info*) echo cp314 ;;
  *find_spec*) echo '{{"present": true, "version": "{vllm}"}}' ;;
esac
exit 0
"#
        ),
    )
    .expect("Python shim");
    rocm_core::fake_executable::write_fake_executable(&bin.join("vllm"), "#!/bin/sh\nexit 0\n")
        .expect("vLLM shim");
    let plain = availability == "also serves plain wheels";
    rocm_core::fake_executable::write_fake_executable(
        &shim.join("uv"),
        format!(
            r#"#!/bin/sh
echo "$*" >> "{state}/uv.log"
for last; do :; done
case "$*" in
  *"pip freeze"*) printf '%s' '{stack}' ;;
  *--dry-run*--config-file*)
    echo ' + flash-attn==2.8.3+rocm{other}' >&2
    echo ' + amd-aiter==0.1.24.post1+rocm{other}' >&2 ;;
  *--dry-run*)
    case "$last" in
      flash-attn===2.8.3|amd-aiter===0.1.24.post1)
        [ '{plain}' = true ] || exit 1
        pkg=${{last%%===*}}
        release=${{last#*===}}
        echo " + $pkg==$release" >&2 ;;
      vllm==*) echo ' + vllm=={vllm}+rocm{sdk}' >&2 ;;
      *) echo " + $last" >&2 ;;
    esac ;;
  *"pip install"*)
    echo "$*" >> "{state}/installs"
    for arg in "$@"; do
      case "$arg" in
        flash-attn===2.8.3) echo 2.8.3 > "{state}/flash-attn" ;;
        flash-attn==*) echo '2.8.3+rocm{other}' > "{state}/flash-attn" ;;
        amd-aiter===0.1.24.post1) echo 0.1.24.post1 > "{state}/amd-aiter" ;;
        amd-aiter==*) echo '0.1.24.post1+rocm{other}' > "{state}/amd-aiter" ;;
      esac
    done ;;
esac
exit 0
"#,
            state = rt.display()
        ),
    )
    .expect("uv shim");
    for (name, body) in [
        (
            "ldconfig",
            "#!/bin/sh\necho 'libmpi.so.40 (libc6,x86-64) => /usr/lib/libmpi.so.40'\n",
        ),
        ("mpirun", "#!/bin/sh\nexit 0\n"),
    ] {
        rocm_core::fake_executable::write_fake_executable(&shim.join(name), body)
            .expect("MPI shim");
    }
    let registry = root.join("data/runtimes/registry");
    std::fs::create_dir_all(&registry).expect("registry");
    let manifest = serde_json::json!({
        "runtime_key": KEY, "runtime_id": KEY, "channel": "release", "format": "wheel",
        "family": "gfx1151", "family_source": "e2e", "version": sdk,
        "install_root": rt, "python_executable": bin.join("python"),
        "index_url": INDEX, "installed_at_unix_ms": 1,
        "selected_artifact_url": "https://example.invalid/sdk.whl",
    });
    std::fs::write(
        registry.join(format!("{KEY}.json")),
        serde_json::to_vec(&manifest).expect("manifest JSON"),
    )
    .expect("manifest");
}

#[when("the user reinstalls vLLM against the mixed companion index")]
async fn reinstall(world: &mut E2eWorld) {
    let shim = runtime(world).join("shim");
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &[
            "engines",
            "install",
            "vllm",
            "--runtime-id",
            KEY,
            "--yes",
            "--reinstall",
        ],
        &[
            ("PATH", shim.to_str().expect("shim path")),
            (
                "ROCM_CLI_UV_BINARY",
                shim.join("uv").to_str().expect("uv path"),
            ),
        ],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

fn output(world: &E2eWorld) -> String {
    format!(
        "{}{}",
        world.cli_output.as_deref().unwrap_or_default(),
        world.cli_stderr.as_deref().unwrap_or_default()
    )
}

#[then("the plain companion wheels are installed and the recovery is reported")]
async fn assert_recovered(world: &mut E2eWorld) {
    let output = output(world);
    assert_eq!(world.cli_rc, Some(0), "{output}");
    for (pkg, release) in [("flash-attn", "2.8.3"), ("amd-aiter", "0.1.24.post1")] {
        assert_eq!(
            std::fs::read_to_string(runtime(world).join(pkg))
                .expect("installed version")
                .trim(),
            release,
            "{output}"
        );
        assert!(
            output.contains(&format!("with `{pkg}=={release}` from `{pkg}==={release}`")),
            "{output}"
        );
    }
}

#[then("the SDK torch stack is realigned from its recorded index")]
async fn assert_stack(world: &mut E2eWorld) {
    let rt = runtime(world);
    let installs = std::fs::read_to_string(rt.join("installs")).expect("install log");
    let realign = installs
        .lines()
        .find(|line| line.contains("torchvision=="))
        .expect("trio realign");
    for pin in std::fs::read_to_string(rt.join("sdk-stack"))
        .expect("SDK stack")
        .lines()
    {
        assert!(
            realign.split_whitespace().any(|arg| arg == pin),
            "{realign}"
        );
    }
    assert!(
        realign.contains(&format!("--index-url {INDEX}")),
        "{realign}"
    );
}

#[then("the cross-line companion is refused and no packages are installed")]
async fn assert_refused(world: &mut E2eWorld) {
    let output = output(world);
    assert_ne!(world.cli_rc, Some(0), "{output}");
    assert!(
        output.contains("flash-attn==2.8.3+rocm10.1.0") && output.contains("10.2"),
        "{output}"
    );
    assert!(!runtime(world).join("installs").exists(), "{output}");
}
