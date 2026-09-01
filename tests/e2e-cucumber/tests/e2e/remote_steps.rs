// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm remote`.
//!
//! Every scenario here plants a stand-in `tailscale` on the CLI's `PATH` and
//! points it at a status document the step wrote. Without that these would pass
//! or fail depending on whether the machine running them happens to have
//! Tailscale installed and peered — and "no targets" would look the same as
//! "the feature is broken".

use std::path::PathBuf;

use cucumber::{given, then, when};

use crate::E2eWorld;

/// A tailnet with one GPU machine and one machine that is not a GPU machine, so
/// tag filtering has something to exclude.
fn tailnet(gpu_online: bool) -> String {
    format!(
        r#"{{
          "BackendState": "Running",
          "TUN": true,
          "MagicDNSSuffix": "example-tailnet.ts.net",
          "Self": {{ "HostName": "laptop", "DNSName": "laptop.example-tailnet.ts.net.", "Online": true }},
          "Peer": {{
            "nodekey:aaa": {{
              "HostName": "gpu-box",
              "DNSName": "gpu-box.example-tailnet.ts.net.",
              "OS": "linux",
              "TailscaleIPs": ["100.88.14.21"],
              "Tags": ["tag:gpu"],
              "Online": {gpu_online}
            }},
            "nodekey:bbb": {{
              "HostName": "phone",
              "DNSName": "phone.example-tailnet.ts.net.",
              "OS": "iOS",
              "TailscaleIPs": ["100.88.51.6"],
              "Online": true
            }}
          }}
        }}"#
    )
}

/// Put the stand-in on `PATH` and, when given, the status document it serves.
fn plant_tailscale(world: &E2eWorld, status: Option<&str>) -> Vec<(String, String)> {
    let root = world
        .isolated_root
        .as_ref()
        .expect("scenario root")
        .path()
        .to_path_buf();
    let bin_dir = root.join("fake-bin");
    std::fs::create_dir_all(&bin_dir).expect("create fake bin dir");

    let built = fake_tailscale_binary();
    let installed = bin_dir.join(if cfg!(windows) {
        "tailscale.exe"
    } else {
        "tailscale"
    });
    std::fs::copy(&built, &installed).unwrap_or_else(|error| {
        panic!(
            "failed to install the tailscale stand-in from {}: {error}",
            built.display()
        )
    });

    let mut env = vec![(
        "PATH".to_owned(),
        format!(
            "{}{}{}",
            bin_dir.display(),
            if cfg!(windows) { ";" } else { ":" },
            std::env::var("PATH").unwrap_or_default()
        ),
    )];
    if let Some(status) = status {
        let path = root.join("tailscale-status.json");
        std::fs::write(&path, status).expect("write status document");
        env.push((
            "FAKE_TAILSCALE_STATUS".to_owned(),
            path.display().to_string(),
        ));
    }
    env
}

/// The stand-in is a bin target of this crate, so cargo has already built it
/// next to the test binary.
fn fake_tailscale_binary() -> PathBuf {
    let mut dir = std::env::current_exe().expect("test binary path");
    dir.pop();
    if dir.ends_with("deps") {
        dir.pop();
    }
    let candidate = dir.join(if cfg!(windows) {
        "fake-tailscale.exe"
    } else {
        "fake-tailscale"
    });
    assert!(
        candidate.exists(),
        "the tailscale stand-in was not built at {}; it is a [[bin]] of this crate",
        candidate.display()
    );
    candidate
}

fn run_remote(world: &mut E2eWorld, args: &[&str], env: &[(String, String)]) {
    let borrowed = env
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect::<Vec<_>>();
    let (stdout, stderr, rc) = crate::run_rocm_with_stdin(world, args, "", &borrowed);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// Everything the command said, however it said it. A refusal is an error, and
/// a scenario asserting on what the user was told should not care which stream
/// carried it.
fn said(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.clone().unwrap_or_default(),
        world.cli_stderr.clone().unwrap_or_default()
    )
}

#[given("a private network with a GPU machine and a phone")]
async fn given_tailnet(world: &mut E2eWorld) {
    world.remote_env = plant_tailscale(world, Some(&tailnet(true)));
}

#[given("a private network whose GPU machine is offline")]
async fn given_offline_tailnet(world: &mut E2eWorld) {
    world.remote_env = plant_tailscale(world, Some(&tailnet(false)));
}

#[given("the private network client is installed but not connected")]
async fn given_not_connected(world: &mut E2eWorld) {
    // No status document: the stand-in then reports a daemon that is installed
    // but not logged in, which is the state this scenario is about.
    world.remote_env = plant_tailscale(world, None);
}

#[when("the user asks which remote targets exist")]
async fn when_targets(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    run_remote(world, &["remote", "targets"], &env);
}

#[when("the user asks for remote targets tagged as GPU machines")]
async fn when_targets_tagged(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    run_remote(world, &["remote", "targets", "--tag", "gpu"], &env);
}

#[when("the user asks to serve a model on a machine that is not there")]
async fn when_serve_unknown(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    run_remote(
        world,
        &["remote", "serve", "not-a-machine", "some-model"],
        &env,
    );
}

#[when("the user asks to serve a model on the GPU machine")]
async fn when_serve_gpu_box(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    run_remote(world, &["remote", "serve", "gpu-box", "some-model"], &env);
}

#[when("the user asks about their remote sessions")]
async fn when_status(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    run_remote(world, &["remote", "status"], &env);
}

#[when("the user asks to stop a remote session that does not exist")]
async fn when_stop_unknown(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    run_remote(world, &["remote", "stop", "no-such-session"], &env);
}

#[when("the user lists local servers as JSON")]
async fn when_services_json(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    run_remote(world, &["services", "list", "--json"], &env);
}

#[then("both machines are listed")]
async fn then_both_listed(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("- gpu-box"), "{said}");
    assert!(said.contains("- phone"), "{said}");
}

#[then("the listing says it is not a readiness check")]
async fn then_not_readiness(world: &mut E2eWorld) {
    let said = said(world);
    assert!(
        said.contains("does not mean they have a GPU"),
        "a list of reachable machines must not read as a list of usable ones:\n{said}"
    );
}

#[then("only the GPU machine is listed")]
async fn then_only_gpu(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("- gpu-box"), "{said}");
    assert!(!said.contains("- phone"), "{said}");
}

#[then("the GPU machine is listed as offline")]
async fn then_offline_listed(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("- gpu-box"), "{said}");
    assert!(said.contains("online: no"), "{said}");
}

#[then("the user is told it is not connected and how to connect")]
async fn then_not_connected(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("not connected"), "{said}");
    assert!(said.contains("tailscale up"), "{said}");
}

#[then("the command still succeeds")]
async fn then_succeeds(world: &mut E2eWorld) {
    assert_eq!(
        world.cli_rc,
        Some(0),
        "asking what exists should answer, not fail:\n{}",
        said(world)
    );
}

#[then("the user is told it is not on the network")]
async fn then_not_on_network(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("not a machine on this tailnet"), "{said}");
    assert_ne!(world.cli_rc, Some(0), "{said}");
}

#[then("they are pointed at the list of machines that are")]
async fn then_pointed_at_targets(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("rocm remote targets"), "{said}");
}

#[then("the user is told the machine is offline")]
async fn then_told_offline(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("offline"), "{said}");
    assert_ne!(world.cli_rc, Some(0), "{said}");
}

#[then("the user is told there are none and how to start one")]
async fn then_no_sessions(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("No remote sessions"), "{said}");
    assert!(said.contains("rocm remote serve"), "{said}");
}

#[then("the user is told no such session is recorded")]
async fn then_no_such_session(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("no remote sessions are recorded"), "{said}");
    assert_ne!(world.cli_rc, Some(0), "{said}");
}

#[then("the output is valid JSON")]
async fn then_valid_json(world: &mut E2eWorld) {
    let stdout = world.cli_output.clone().unwrap_or_default();
    serde_json::from_str::<serde_json::Value>(stdout.trim())
        .unwrap_or_else(|error| panic!("not JSON ({error}):\n{stdout}"));
}

// ── Scenarios needing a second machine ─────────────────────────────
//
// `rocm remote` opens a real SSH connection, so the successful paths cannot be
// covered by stubbing alone — there has to be a host on the other end. These
// stand one up as a container built from tests/remote-ssh, whose `rocm` and
// `tailscale` are stand-ins. What is real: the connection, the arguments the
// CLI builds, and the session records it writes.

/// A container standing in for a GPU machine, torn down with the World.
#[derive(Debug)]
pub struct RemoteMachine {
    container: String,
    /// Env the CLI needs to reach it: an ssh config, the tailnet stand-in, PATH.
    pub env: Vec<(String, String)>,
}

impl Drop for RemoteMachine {
    fn drop(&mut self) {
        let _ = std::process::Command::new("docker")
            .args(["rm", "-f", &self.container])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}

impl RemoteMachine {
    /// Run a command on the machine itself, to set up or inspect state the CLI
    /// is not responsible for.
    pub fn exec(&self, args: &[&str]) -> String {
        let output = std::process::Command::new("docker")
            .arg("exec")
            .arg(&self.container)
            .args(args)
            .output()
            .expect("docker exec");
        String::from_utf8_lossy(&output.stdout).into_owned()
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("repo root")
        .to_path_buf()
}

fn docker(args: &[&str]) -> std::process::Output {
    std::process::Command::new("docker")
        .args(args)
        .output()
        .expect("docker")
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind")
        .local_addr()
        .expect("addr")
        .port()
}

/// Build the image, start it, and set up everything the CLI needs to reach it.
fn start_remote_machine(world: &E2eWorld) -> RemoteMachine {
    let fixtures = repo_root().join("tests").join("remote-ssh");
    let build = docker(&[
        "build",
        "-q",
        "--build-arg",
        &format!(
            "APK_REPO_FLAGS={}",
            std::env::var("ROCM_TEST_APK_REPOS").unwrap_or_default()
        ),
        "-t",
        "rocm-remote-ssh-test",
        fixtures.to_str().expect("fixtures path"),
    ]);
    assert!(
        build.status.success(),
        "failed to build the stand-in machine: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let root = world
        .isolated_root
        .as_ref()
        .expect("scenario root")
        .path()
        .to_path_buf();
    let port = free_port();
    let container = format!("rocm-e2e-remote-{}-{port}", std::process::id());
    let run = docker(&[
        "run",
        "-d",
        "--name",
        &container,
        "-p",
        &format!("127.0.0.1:{port}:22"),
        "rocm-remote-ssh-test",
    ]);
    assert!(
        run.status.success(),
        "failed to start the stand-in machine: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let mut machine = RemoteMachine {
        container: container.clone(),
        env: Vec::new(),
    };

    // A key for this scenario only.
    let key = root.join("id");
    let keygen = std::process::Command::new("ssh-keygen")
        .args(["-q", "-t", "ed25519", "-N", "", "-f"])
        .arg(&key)
        .status()
        .expect("ssh-keygen");
    assert!(keygen.success(), "ssh-keygen failed");
    let public = std::fs::read_to_string(key.with_extension("pub")).expect("read public key");
    let install = std::process::Command::new("docker")
        .args([
            "exec",
            &container,
            "sh",
            "-c",
            &format!(
                "printf '%s\\n' '{}' >> /root/.ssh/authorized_keys",
                public.trim()
            ),
        ])
        .status()
        .expect("install key");
    assert!(install.success(), "failed to authorize the scenario key");

    // ssh resolves ~/.ssh/config from the account database rather than from
    // HOME, so the CLI has to be told where this scenario's config is.
    let ssh_config = root.join("ssh-config");
    std::fs::write(
        &ssh_config,
        format!(
            "Host gpu-box\n  HostName 127.0.0.1\n  Port {port}\n  User root\n  \
             IdentityFile {}\n  IdentitiesOnly yes\n  StrictHostKeyChecking no\n  \
             UserKnownHostsFile /dev/null\n  LogLevel ERROR\n",
            key.display()
        ),
    )
    .expect("write ssh config");

    let status = root.join("tailnet.json");
    std::fs::write(&status, tailnet(true)).expect("write tailnet status");
    let mut env = plant_tailscale(world, Some(&tailnet(true)));
    env.push((
        "FAKE_TAILSCALE_STATUS".to_owned(),
        status.display().to_string(),
    ));
    env.push((
        "ROCM_REMOTE_SSH_CONFIG".to_owned(),
        ssh_config.display().to_string(),
    ));

    // Wait for sshd rather than sleeping a fixed amount.
    let mut ready = false;
    for _ in 0..80 {
        let probe = std::process::Command::new("ssh")
            .args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-F"])
            .arg(&ssh_config)
            .args(["gpu-box", "true"])
            .status();
        if probe.is_ok_and(|status| status.success()) {
            ready = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    assert!(ready, "the stand-in machine never accepted a connection");

    machine.env = env;
    machine
}

const fn machine(world: &E2eWorld) -> &RemoteMachine {
    world.remote_machine.as_ref().expect("no machine started")
}

#[given("a reachable GPU machine on the private network")]
async fn given_reachable_machine(world: &mut E2eWorld) {
    let started = start_remote_machine(world);
    world.remote_env = started.env.clone();
    world.remote_machine = Some(started);
}

#[given("a model serving on a reachable GPU machine")]
async fn given_serving_machine(world: &mut E2eWorld) {
    given_reachable_machine(world).await;
    let env = world.remote_env.clone();
    run_remote(world, &["remote", "serve", "gpu-box", "test-model"], &env);
    assert_eq!(world.cli_rc, Some(0), "serve failed:\n{}", said(world));
}

#[when("the user serves a model on that machine")]
async fn when_serve_on_machine(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    run_remote(world, &["remote", "serve", "gpu-box", "test-model"], &env);
}

#[when("the user checks that machine's health")]
async fn when_check_health(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    run_remote(world, &["remote", "doctor", "gpu-box"], &env);
}

#[when("the endpoint is withdrawn on the machine itself")]
async fn when_withdrawn_remotely(world: &mut E2eWorld) {
    machine(world).exec(&["tailscale", "serve", "--tcp=8000", "off"]);
}

#[when("the user re-publishes the endpoint")]
async fn when_reattach(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    let session = current_session(world);
    run_remote(world, &["remote", "attach", &session], &env);
}

#[when("the user stops the session")]
async fn when_stop_session(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    let session = current_session(world);
    run_remote(world, &["remote", "stop", &session], &env);
}

/// The one session this scenario started, read back from the CLI's own listing
/// rather than reconstructed — if the id were derived here the test could pass
/// against a listing that shows something else.
fn current_session(world: &mut E2eWorld) -> String {
    let env = world.remote_env.clone();
    run_remote(world, &["remote", "status"], &env);
    said(world)
        .lines()
        .find_map(|line| line.trim().strip_prefix("- ").map(str::to_owned))
        .expect("no session in the listing")
}

#[then("the user is given an endpoint and a credential")]
async fn then_endpoint_and_credential(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("endpoint: http://gpu-box."), "{said}");
    assert!(said.contains("api key:"), "{said}");
}

#[then("the user is told the endpoint is reachable by the whole network")]
async fn then_reach_stated(world: &mut E2eWorld) {
    let said = said(world);
    assert!(
        said.contains("every machine on your tailnet"),
        "the loopback intuition from local serving does not carry over and must be \
         corrected explicitly:\n{said}"
    );
}

#[then("the machine is publishing that endpoint")]
async fn then_machine_publishing(world: &mut E2eWorld) {
    let config = machine(world).exec(&["tailscale", "serve", "status", "--json"]);
    assert!(
        config.contains("\"8000\""),
        "machine is not publishing:\n{config}"
    );
}

#[then("the model and the endpoint are both reported healthy")]
async fn then_both_healthy(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("model server: healthy"), "{said}");
    assert!(said.contains("endpoint published: yes"), "{said}");
}

#[then("the model is still healthy but the endpoint is reported gone")]
async fn then_model_healthy_endpoint_gone(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("model server: healthy"), "{said}");
    assert!(said.contains("endpoint published: no"), "{said}");
}

#[then("the endpoint is restored without restarting the model")]
async fn then_endpoint_restored(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("Endpoint re-published"), "{said}");
    assert!(said.contains("not restarted"), "{said}");
}

#[then("the endpoint and the model are both reported stopped")]
async fn then_both_stopped(world: &mut E2eWorld) {
    let said = said(world);
    assert!(said.contains("endpoint withdrawn: yes"), "{said}");
    assert!(said.contains("model server stopped: yes"), "{said}");
}

#[then("the machine is publishing nothing")]
async fn then_publishing_nothing(world: &mut E2eWorld) {
    let config = machine(world).exec(&["tailscale", "serve", "status", "--json"]);
    assert!(
        !config.contains("\"8000\""),
        "an endpoint was left published:\n{config}"
    );
}

#[then("the session is no longer listed")]
async fn then_session_gone(world: &mut E2eWorld) {
    let env = world.remote_env.clone();
    run_remote(world, &["remote", "status"], &env);
    let said = said(world);
    assert!(said.contains("No remote sessions"), "{said}");
}

#[then("the report names that machine")]
async fn then_report_names_machine(world: &mut E2eWorld) {
    let said = said(world);
    assert!(
        said.contains("Health of gpu-box"),
        "a remote report indistinguishable from a local one gets acted on against the \
         wrong computer:\n{said}"
    );
}
