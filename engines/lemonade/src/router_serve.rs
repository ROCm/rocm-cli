// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Serving through the Lemonade router (`lemond`): spawn the server, select a
//! GPU llama.cpp backend, load the model by its router name and verify it.
//!
//! The orchestration lives behind [`RouterSteps`] so the state it persists and
//! the way it fails can be tested without a real `lemond`. Production uses
//! [`LemondRouterSteps`]. There is deliberately no direct-server step here: any
//! failure is returned to the caller as an error and the server is stopped. The
//! engine state is then left at `failed`, except for a model load or
//! verification failure: that is tagged [`RouterLoadFailure`] and does not write
//! `failed`, so the caller can decide, on Linux, whether the selected GPU
//! backend can serve the model directly instead and own the final state.

use anyhow::{Context, Result, anyhow};
use serde_json::json;
use std::path::Path;
use std::time::Duration;

use crate::backend_alignment::LemonadeRuntime;
use crate::process::{LemonadeProcessEnvironment, LemondChild, LemondExitStatus};
use crate::runtime_dir::ParentRuntimeEnvironment;
use crate::{LLAMACPP_RECIPE, ServeHttpRequest};

/// Marks a router model-load or load-verification failure (as opposed to a
/// server that never became ready, a failed backend selection, or a server that
/// exited later). Only these are eligible for the direct-serving fallback.
#[derive(Debug)]
pub(crate) struct RouterLoadFailure(pub(crate) String);

impl std::fmt::Display for RouterLoadFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Whether `error` came from the router failing to load or verify the model.
pub(crate) fn is_router_load_failure(error: &anyhow::Error) -> bool {
    error.downcast_ref::<RouterLoadFailure>().is_some()
}

/// The side-effecting steps of a router serve.
pub(crate) trait RouterSteps {
    fn runtime(&self) -> &LemonadeRuntime;
    /// Start `lemond` and return its PID.
    fn spawn_server(&mut self) -> Result<u32>;
    fn wait_ready(&mut self) -> Result<()>;
    /// Select (and install if needed) the GPU llama.cpp backend; returns its name.
    fn select_backend(&mut self) -> Result<String>;
    fn load_model(&mut self, router_model_ref: &str, backend: &str) -> Result<()>;
    fn model_loaded(&mut self, router_model_ref: &str, backend: &str) -> bool;
    fn chat_smoke(&mut self, router_model_ref: &str) -> bool;
    /// Kill and reap the spawned server.
    fn stop_server(&mut self);
    fn server_start_ticks(&self, pid: u32) -> Option<u64>;
    /// Block until the server exits.
    fn wait_server(&mut self) -> Result<LemondExitStatus>;
}

/// Serve `router_model_ref` through the router. `request.model_ref` keeps the
/// text the user asked for; every router call uses the normalized name.
pub(crate) fn serve_router<S: RouterSteps>(
    request: &ServeHttpRequest,
    steps: &mut S,
    router_model_ref: &str,
) -> Result<()> {
    let server_pid = steps.spawn_server()?;
    // Every step after the spawn is inside this one fallible block, so a failure
    // anywhere in it (state writes included) stops the server and marks the state
    // `failed`; the server has no `Drop` and would otherwise keep running.
    let outcome = (|| -> Result<()> {
        crate::state::write_running_state_for_model(
            request,
            steps.runtime(),
            std::process::id(),
            Some(server_pid),
            "running",
            router_model_ref,
        )?;
        steps
            .wait_ready()
            .context("Lemonade server did not become ready")?;
        let backend = steps
            .select_backend()
            .context("failed to select a supported Lemonade llama.cpp backend")?;
        // A load failure is tagged `RouterLoadFailure`, which is what lets
        // `with_direct_fallback` serve a registered model directly on Linux. The
        // router's own error is kept and returned when no direct route is open.
        steps
            .load_model(router_model_ref, &backend)
            .map_err(|error| {
                error.context(RouterLoadFailure(format!(
                    "failed to load {} with Lemonade {LLAMACPP_RECIPE}:{backend}",
                    request.model_ref
                )))
            })?;
        if !steps.model_loaded(router_model_ref, &backend) || !steps.chat_smoke(router_model_ref) {
            return Err(anyhow!(
                "Lemonade load completed but the endpoint did not report a {LLAMACPP_RECIPE}:{backend}-loaded model"
            )
            .context(RouterLoadFailure(format!(
                "failed to verify {} with Lemonade {LLAMACPP_RECIPE}:{backend}",
                request.model_ref
            ))));
        }
        crate::state::merge_json_state(
            &request.state_path,
            &json!({
                "status": "ready",
                "server_pid": server_pid,
                // Identity token for the server PID, captured while the child is alive.
                "server_start_ticks": steps.server_start_ticks(server_pid),
                "backend_state": "ready",
                "backend_requested": backend,
                "load_response": {
                    "status": "loaded",
                    "method": "lemonade-cli",
                    "model_name": router_model_ref,
                    "llamacpp_backend": backend
                },
            }),
        )?;
        // Only now is it true that Lemonade is serving on this backend: the load
        // and its verification passed and the `ready` state is written. Announcing
        // earlier would claim it for a load that fails, for a ready write that
        // fails (the server is then stopped), and again on the fallback.
        crate::state::announce_non_rocm_backend(&backend, request.log_path.as_deref());
        Ok(())
    })();
    if let Err(error) = outcome {
        steps.stop_server();
        // A load or verification failure may still be served directly, so the
        // fallback decides the final state: it sets `starting` when it runs and
        // `failed` when it does not (see `with_direct_fallback`). Marking `failed`
        // here would let a reader in the gap lock the record at `failed`.
        if !is_router_load_failure(&error) {
            let _ = crate::state::mark_json_status(&request.state_path, "failed");
        }
        return Err(error);
    }
    let status = match steps.wait_server() {
        Ok(status) => status,
        Err(error) => {
            steps.stop_server();
            let _ = crate::state::mark_json_status(&request.state_path, "failed");
            return Err(error);
        }
    };
    crate::state::mark_json_status(
        &request.state_path,
        if status.success() {
            "stopped"
        } else {
            "failed"
        },
    )?;
    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("Lemonade server exited with status {status}")
    }
}

/// The real `lemond` and `lemonade` CLI behind [`RouterSteps`].
pub(crate) struct LemondRouterSteps<'a> {
    pub(crate) runtime: LemonadeRuntime,
    pub(crate) request: &'a ServeHttpRequest,
    pub(crate) process_env: &'a LemonadeProcessEnvironment,
    pub(crate) log_path: Option<&'a Path>,
    /// The environment the child processes inherit from; tests confine it.
    pub(crate) parent: ParentRuntimeEnvironment,
    pub(crate) child: Option<LemondChild>,
}

impl RouterSteps for LemondRouterSteps<'_> {
    fn runtime(&self) -> &LemonadeRuntime {
        &self.runtime
    }

    fn spawn_server(&mut self) -> Result<u32> {
        let child = crate::process::spawn_lemond_with_parent(
            &self.runtime.manifest,
            &self.request.host,
            self.request.port,
            self.log_path,
            self.process_env,
            &self.parent,
        )?;
        let pid = child.id();
        self.child = Some(child);
        Ok(pid)
    }

    fn wait_ready(&mut self) -> Result<()> {
        crate::process::wait_for_lemonade_cli_status_with_parent(
            &self.runtime.manifest,
            &self.request.host,
            self.request.port,
            Duration::from_secs(30),
            self.log_path,
            self.process_env,
            &self.parent,
        )
    }

    fn select_backend(&mut self) -> Result<String> {
        crate::backend_alignment::ensure_best_llamacpp_backend_with_parent(
            &mut self.runtime.manifest,
            &self.request.host,
            self.request.port,
            self.process_env,
            false,
            &self.parent,
        )
    }

    fn load_model(&mut self, router_model_ref: &str, backend: &str) -> Result<()> {
        crate::process::run_lemonade_model_load_with_parent(
            &self.runtime.manifest,
            &self.request.host,
            self.request.port,
            router_model_ref,
            backend,
            self.request.engine_recipe.as_ref(),
            self.log_path,
            self.process_env,
            &self.parent,
        )
    }

    fn model_loaded(&mut self, router_model_ref: &str, backend: &str) -> bool {
        crate::process::query_loaded_model_endpoint(
            &crate::state::endpoint_url(&self.request.host, self.request.port),
            router_model_ref,
            backend,
        )
        .unwrap_or(false)
    }

    fn chat_smoke(&mut self, router_model_ref: &str) -> bool {
        crate::process::query_chat_smoke_endpoint(
            &self.request.host,
            self.request.port,
            router_model_ref,
        )
        .unwrap_or(false)
    }

    fn stop_server(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = crate::state::terminate_pid(child.id(), true);
            let _ = child.wait();
        }
    }

    fn server_start_ticks(&self, pid: u32) -> Option<u64> {
        rocm_core::process_start_ticks(pid)
    }

    fn wait_server(&mut self) -> Result<LemondExitStatus> {
        self.child
            .as_mut()
            .context("Lemonade server was not started")?
            .wait()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::install::LemonadeInstallManifest;
    use std::path::PathBuf;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Fail {
        None,
        Ready,
        Backend,
        Load,
        Loaded,
        Smoke,
        /// The state file cannot be written once the server is spawned.
        StateWrite,
        /// The state file becomes unwritable between verification and the ready merge.
        ReadyMerge,
        WaitServer,
        FailedExit,
    }

    struct FakeSteps {
        runtime: LemonadeRuntime,
        fail: Fail,
        calls: Vec<String>,
        state_path: PathBuf,
        /// The state file as it stood when the load step ran.
        state_at_load: Option<serde_json::Value>,
    }

    impl FakeSteps {
        fn new(dir: &Path, fail: Fail) -> Self {
            Self {
                runtime: LemonadeRuntime {
                    manifest: LemonadeInstallManifest {
                        env_id: "test".to_owned(),
                        version: "test".to_owned(),
                        runtime_dir: dir.join("runtime"),
                        lemond: dir.join("lemond"),
                        lemonade: dir.join("lemonade"),
                        backend_recipe: LLAMACPP_RECIPE.to_owned(),
                        backend_name: "vulkan".to_owned(),
                        installed_at_unix_ms: 0,
                    },
                },
                fail,
                calls: Vec::new(),
                state_path: dir.join("state.json"),
                state_at_load: None,
            }
        }
    }

    impl RouterSteps for FakeSteps {
        fn runtime(&self) -> &LemonadeRuntime {
            &self.runtime
        }
        fn spawn_server(&mut self) -> Result<u32> {
            self.calls.push("spawn".to_owned());
            Ok(4_000_000)
        }
        fn wait_ready(&mut self) -> Result<()> {
            self.calls.push("ready".to_owned());
            if self.fail == Fail::Ready {
                anyhow::bail!("not ready");
            }
            Ok(())
        }
        fn select_backend(&mut self) -> Result<String> {
            self.calls.push("backend".to_owned());
            if self.fail == Fail::Backend {
                anyhow::bail!("no backend");
            }
            Ok("vulkan".to_owned())
        }
        fn load_model(&mut self, router_model_ref: &str, backend: &str) -> Result<()> {
            self.calls
                .push(format!("load:{router_model_ref}:{backend}"));
            self.state_at_load = crate::state::read_service_state(&self.state_path).ok();
            if self.fail == Fail::Load {
                anyhow::bail!("load exploded");
            }
            Ok(())
        }
        fn model_loaded(&mut self, router_model_ref: &str, backend: &str) -> bool {
            self.calls
                .push(format!("loaded:{router_model_ref}:{backend}"));
            self.fail != Fail::Loaded
        }
        fn chat_smoke(&mut self, router_model_ref: &str) -> bool {
            self.calls.push(format!("smoke:{router_model_ref}"));
            self.fail != Fail::Smoke
        }
        fn stop_server(&mut self) {
            self.calls.push("stop".to_owned());
        }
        fn server_start_ticks(&self, _pid: u32) -> Option<u64> {
            // Asked for while the ready state is assembled, i.e. after every
            // verification step and before the merge is written.
            if self.fail == Fail::ReadyMerge {
                std::fs::remove_file(&self.state_path).unwrap();
                std::fs::create_dir(&self.state_path).unwrap();
            }
            None
        }
        fn wait_server(&mut self) -> Result<LemondExitStatus> {
            self.calls.push("wait".to_owned());
            if self.fail == Fail::WaitServer {
                anyhow::bail!("lost the server");
            }
            Ok(if self.fail == Fail::FailedExit {
                LemondExitStatus::for_test(false, "exit status: 23")
            } else {
                LemondExitStatus::for_test(true, "exit status: 0")
            })
        }
    }

    fn request(dir: &Path) -> ServeHttpRequest {
        ServeHttpRequest {
            service_id: "test".to_owned(),
            model_ref: "qwen-smoke".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: rocm_engine_protocol::DevicePolicy::GpuRequired,
            gpu_indices: vec![0],
            runtime_id: None,
            env_id: None,
            state_path: dir.join("state.json"),
            log_path: None,
            engine_recipe: None,
        }
    }

    const ROUTER_NAME: &str = "Qwen3-0.6B-GGUF";

    #[test]
    fn router_state_keeps_requested_text_and_records_router_name() {
        let temporary = tempfile::tempdir().unwrap();
        let request = request(temporary.path());
        let mut steps = FakeSteps::new(temporary.path(), Fail::None);
        serve_router(&request, &mut steps, ROUTER_NAME).unwrap();

        // The state written before the load: requested text kept, router name served.
        let at_load = steps.state_at_load.expect("state exists before load");
        assert_eq!(at_load["status"], "running");
        assert_eq!(at_load["model_ref"], "qwen-smoke");
        assert_eq!(at_load["canonical_model_id"], ROUTER_NAME);

        let state = crate::state::read_service_state(&request.state_path).unwrap();
        assert_eq!(state["model_ref"], "qwen-smoke");
        assert_eq!(state["canonical_model_id"], ROUTER_NAME);
        assert_eq!(state["load_response"]["model_name"], ROUTER_NAME);
        assert_eq!(state["backend_requested"], "vulkan");
        assert_eq!(state["status"], "stopped");
    }

    #[test]
    fn router_name_not_alias_reaches_load_loaded_and_smoke_checks() {
        let temporary = tempfile::tempdir().unwrap();
        let request = request(temporary.path());
        let mut steps = FakeSteps::new(temporary.path(), Fail::None);
        serve_router(&request, &mut steps, ROUTER_NAME).unwrap();
        assert_eq!(
            steps.calls,
            [
                "spawn".to_owned(),
                "ready".to_owned(),
                "backend".to_owned(),
                format!("load:{ROUTER_NAME}:vulkan"),
                format!("loaded:{ROUTER_NAME}:vulkan"),
                format!("smoke:{ROUTER_NAME}"),
                "wait".to_owned(),
            ]
        );
    }

    #[test]
    fn router_failures_return_error_stop_server_and_mark_state_failed() {
        for (fail, expected, last_call) in [
            (Fail::Ready, "did not become ready", "ready"),
            (Fail::Backend, "failed to select a supported", "backend"),
            (
                Fail::Load,
                "failed to load qwen-smoke with Lemonade",
                "load",
            ),
            (
                Fail::Loaded,
                "failed to verify qwen-smoke with Lemonade",
                "loaded",
            ),
            (
                Fail::Smoke,
                "failed to verify qwen-smoke with Lemonade",
                "smoke",
            ),
        ] {
            let temporary = tempfile::tempdir().unwrap();
            let request = request(temporary.path());
            let mut steps = FakeSteps::new(temporary.path(), fail);
            let error = serve_router(&request, &mut steps, ROUTER_NAME)
                .expect_err(expected)
                .to_string();
            assert!(error.contains(expected), "{error}");
            // The server is stopped, and the router branch has no other step to
            // fall back to: the calls are the router sequence up to the failure,
            // then the stop, and nothing after it.
            assert_eq!(
                steps.calls.last().map(String::as_str),
                Some("stop"),
                "{:?}",
                steps.calls
            );
            assert!(
                steps.calls[steps.calls.len() - 2].starts_with(last_call),
                "{:?}",
                steps.calls
            );
            assert!(!steps.calls.iter().any(|call| call == "wait"));
            let state = crate::state::read_service_state(&request.state_path).unwrap();
            // A load or verification failure is left for the direct fallback to
            // settle (it sets `starting` or `failed`); the other failures are final.
            let expected_status = if matches!(fail, Fail::Load | Fail::Loaded | Fail::Smoke) {
                "running"
            } else {
                "failed"
            };
            assert_eq!(state["status"], expected_status, "{expected}");
            assert_eq!(state["model_ref"], "qwen-smoke");
        }
    }

    /// A failure at a step after the spawn (not a router step) must still stop the
    /// server it spawned.
    fn assert_post_spawn_failure_stops_server(fail: Fail, expected: &str) {
        let temporary = tempfile::tempdir().unwrap();
        let request = request(temporary.path());
        if fail == Fail::StateWrite {
            // A directory where the state file belongs: every write fails.
            std::fs::create_dir(&request.state_path).unwrap();
        }
        let mut steps = FakeSteps::new(temporary.path(), fail);
        let error = format!(
            "{:#}",
            serve_router(&request, &mut steps, ROUTER_NAME).expect_err(expected)
        );
        assert!(error.contains(expected), "{error}");
        // The spawned server is stopped exactly once, and nothing runs after it.
        assert_eq!(
            steps.calls.last().map(String::as_str),
            Some("stop"),
            "{:?}",
            steps.calls
        );
        assert_eq!(
            steps.calls.iter().filter(|call| *call == "stop").count(),
            1,
            "{:?}",
            steps.calls
        );
        if fail == Fail::WaitServer {
            // The state file is writable here, so the failure is recorded.
            let state = crate::state::read_service_state(&request.state_path).unwrap();
            assert_eq!(state["status"], "failed");
        }
    }

    #[test]
    fn state_write_failure_right_after_spawn_stops_the_server() {
        assert_post_spawn_failure_stops_server(Fail::StateWrite, "failed to write");
    }

    #[test]
    fn ready_state_merge_failure_stops_the_server() {
        assert_post_spawn_failure_stops_server(Fail::ReadyMerge, "failed to write");
    }

    /// The ready write fails after the model loaded and verified, and the server
    /// is stopped: the log must not say Lemonade is serving on the backend.
    #[test]
    fn a_failed_ready_state_write_does_not_announce_a_backend() {
        let temporary = tempfile::tempdir().unwrap();
        let mut request = request(temporary.path());
        request.log_path = Some(temporary.path().join("serve.log"));
        let mut steps = FakeSteps::new(temporary.path(), Fail::ReadyMerge);
        serve_router(&request, &mut steps, ROUTER_NAME).unwrap_err();
        assert_eq!(steps.calls.last().map(String::as_str), Some("stop"));
        let log = std::fs::read_to_string(request.log_path.as_ref().unwrap()).unwrap_or_default();
        assert!(!log.contains("not ROCm"), "{log}");
    }

    #[test]
    fn wait_server_failure_stops_the_server_and_marks_state_failed() {
        assert_post_spawn_failure_stops_server(Fail::WaitServer, "lost the server");
    }

    #[test]
    fn only_a_load_or_verification_failure_is_eligible_for_direct_fallback() {
        for (fail, eligible) in [
            (Fail::Load, true),
            (Fail::Loaded, true),
            (Fail::Smoke, true),
            (Fail::Ready, false),
            (Fail::Backend, false),
            (Fail::StateWrite, false),
            (Fail::ReadyMerge, false),
            (Fail::WaitServer, false),
            (Fail::FailedExit, false),
        ] {
            let temporary = tempfile::tempdir().unwrap();
            let request = request(temporary.path());
            if fail == Fail::StateWrite {
                std::fs::create_dir(&request.state_path).unwrap();
            }
            let mut steps = FakeSteps::new(temporary.path(), fail);
            let error = serve_router(&request, &mut steps, ROUTER_NAME).unwrap_err();
            assert_eq!(is_router_load_failure(&error), eligible, "{error:#}");
            // The server was stopped, except when it had already exited by itself.
            assert_eq!(
                steps.calls.iter().any(|call| call == "stop"),
                fail != Fail::FailedExit,
                "{:?}",
                steps.calls
            );
        }
    }

    #[test]
    fn a_non_rocm_backend_is_named_in_the_notice_and_matches_the_state() {
        let temporary = tempfile::tempdir().unwrap();
        let mut request = request(temporary.path());
        request.log_path = Some(temporary.path().join("serve.log"));
        let mut steps = FakeSteps::new(temporary.path(), Fail::None);
        serve_router(&request, &mut steps, ROUTER_NAME).unwrap();
        let state = crate::state::read_service_state(&request.state_path).unwrap();
        let backend = state["backend_requested"].as_str().unwrap();
        assert_eq!(backend, "vulkan");
        // The notice is computed from the recorded backend, so it cannot disagree.
        let notice = crate::state::non_rocm_backend_notice(backend).unwrap();
        assert!(
            notice.contains("vulkan") && notice.contains("not ROCm"),
            "{notice}"
        );
        // A managed serve discards stdout and stderr, so the router path has to
        // put the notice in the serve log, where the user can read it.
        let log = std::fs::read_to_string(request.log_path.as_ref().unwrap()).unwrap();
        assert!(log.contains(&notice), "{log}");
        assert_eq!(crate::state::non_rocm_backend_notice("rocm"), None);
        assert_eq!(crate::state::non_rocm_backend_notice("rocm-stable"), None);
    }

    /// The notice says Lemonade is serving on the backend, so it is written only
    /// once the model is loaded and verified: a failed load or verification claims
    /// nothing (and the fallback that may follow announces for itself).
    #[test]
    fn a_failed_load_or_verification_does_not_announce_a_backend() {
        for fail in [Fail::Load, Fail::Loaded, Fail::Smoke] {
            let temporary = tempfile::tempdir().unwrap();
            let mut request = request(temporary.path());
            request.log_path = Some(temporary.path().join("serve.log"));
            let mut steps = FakeSteps::new(temporary.path(), fail);
            serve_router(&request, &mut steps, ROUTER_NAME).unwrap_err();
            let log =
                std::fs::read_to_string(request.log_path.as_ref().unwrap()).unwrap_or_default();
            assert!(!log.contains("not ROCm"), "{log}");
        }
    }

    #[test]
    fn nonzero_server_exit_returns_error_and_marks_state_failed() {
        let temporary = tempfile::tempdir().unwrap();
        let request = request(temporary.path());
        let mut steps = FakeSteps::new(temporary.path(), Fail::FailedExit);
        let error = serve_router(&request, &mut steps, ROUTER_NAME).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Lemonade server exited with status exit status: 23")
        );
        assert_eq!(
            crate::state::read_service_state(&request.state_path).unwrap()["status"],
            "failed"
        );
        assert_eq!(steps.calls.last().map(String::as_str), Some("wait"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn real_router_stop_kills_and_reaps_its_server() {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let request = request(dir);
        let env = LemonadeProcessEnvironment::default();
        let mut steps = LemondRouterSteps {
            runtime: LemonadeRuntime {
                manifest: crate::test_http::fake_runtime::manifest(dir, 0, None),
            },
            request: &request,
            process_env: &env,
            log_path: None,
            parent: ParentRuntimeEnvironment::rooted_at(dir),
            child: None,
        };
        let pid = steps.spawn_server().unwrap();
        let process = PathBuf::from(format!("/proc/{pid}"));
        assert!(process.exists(), "the real spawn must create a process");
        steps.stop_server();
        // Clean up even if a no-op stop mutation left the child alive.
        let gone = !process.exists();
        if !gone {
            let _ = crate::state::terminate_pid(pid, true);
            if let Some(mut child) = steps.child.take() {
                let _ = child.wait();
            }
        }
        assert!(
            gone,
            "stop_server must terminate and reap the spawned lemond"
        );
        assert!(steps.child.is_none());
    }

    /// Backend selection runs `lemonade backends` through the serve request's
    /// confined parent environment, never the host's: the child must see the
    /// confined runtime directory, and the answer must come from its listing.
    #[cfg(target_os = "linux")]
    #[test]
    fn real_backend_selection_runs_under_the_confined_parent() {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let request = request(dir);
        let env = LemonadeProcessEnvironment::default();
        let mut manifest = crate::test_http::fake_runtime::manifest(dir, 0, None);
        manifest.lemonade = dir.join("lemonade-backends");
        rocm_core::fake_executable::write_fake_executable(
            &manifest.lemonade,
            r#"#!/bin/sh
printf '%s' "$XDG_RUNTIME_DIR" > "$(dirname "$0")/xdg.txt"
printf 'Recipe  Backend  Status  Message  Action
llamacpp  vulkan  installed  ok  none
'
"#,
        )
        .unwrap();
        let mut steps = LemondRouterSteps {
            runtime: LemonadeRuntime { manifest },
            request: &request,
            process_env: &env,
            log_path: None,
            parent: ParentRuntimeEnvironment::rooted_at(dir),
            child: None,
        };
        assert_eq!(steps.select_backend().unwrap(), "vulkan");
        assert_eq!(
            std::fs::read_to_string(dir.join("xdg.txt")).unwrap(),
            dir.join("xdg-runtime").to_string_lossy()
        );
        // The listing said `installed`, so nothing was installed.
        assert_eq!(steps.runtime.manifest.backend_name, "vulkan");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn real_steps_send_the_router_name_not_the_requested_alias() {
        use crate::test_http::{Reply, fake_runtime, spawn_fixture};
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        // A name the alias `qwen-smoke` does not normalize to, so a step that used
        // the requested text could not pass by matching it.
        let router_name = "Registered-Model-GGUF";
        let (port, server) = spawn_fixture(Duration::from_secs(10), move |recorded| {
            let chat = recorded.method == "POST";
            Reply {
                body: if chat {
                    json!({"choices": [{"message": {"content": "ok"}}]})
                } else {
                    json!({"all_models_loaded": [{"model_name": router_name, "recipe_options": {"llamacpp_backend": "vulkan"}}]})
                }
                .to_string(),
                last: chat,
            }
        });
        let mut request = request(dir);
        request.port = port;
        assert_eq!(request.model_ref, "qwen-smoke");
        let env = LemonadeProcessEnvironment::default();
        let mut steps = LemondRouterSteps {
            runtime: LemonadeRuntime {
                manifest: fake_runtime::manifest(dir, 0, None),
            },
            request: &request,
            process_env: &env,
            log_path: None,
            parent: ParentRuntimeEnvironment::rooted_at(dir),
            child: None,
        };
        steps.load_model(router_name, "vulkan").unwrap();
        assert!(steps.model_loaded(router_name, "vulkan"));
        assert!(steps.chat_smoke(router_name));

        let load = fake_runtime::calls(dir);
        assert_eq!(
            load,
            [format!(
                "--host 127.0.0.1 --port {port} load {router_name} --llamacpp vulkan --save-options"
            )]
        );
        let seen = server.join().unwrap();
        let chat = seen.iter().find(|r| r.method == "POST").expect("chat sent");
        assert_eq!(chat.path, "/v1/chat/completions");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&chat.body).unwrap()["model"],
            router_name
        );
    }
}
