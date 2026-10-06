// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Cross-platform local smoke test for the built `rocm`/`rocmd`/engine binaries.
//!
//! Runs the freshly built binaries against a throwaway state root and asserts
//! their first-run output: the traceable version string, the examine/engines
//! inventory, telemetry config, the freeform command surface, the `rocmd`
//! bridge and sandbox tools, and — the reason this gate exists — that every
//! GPU-required path fails loudly instead of silently falling back to CPU.
//!
//! Replaces the former `scripts/smoke_local.py`, preserving each of its
//! assertions. Two deliberate differences from that script:
//!
//! * The target directory is resolved with [`crate::paths::target_dir`], so a
//!   configured `CARGO_TARGET_DIR` is honoured. The script always looked under
//!   `<root>/target` and so reported every binary missing when that variable
//!   pointed elsewhere.
//! * The script merged each child's stderr into its stdout pipe, truly
//!   interleaving them. `std::process` has no portable equivalent, so the two
//!   streams are captured separately and concatenated (stdout first). Every
//!   assertion here is a substring test or reads the first line of stdout, so
//!   the ordering between streams is not load-bearing.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::paths::{binary_name, target_dir, workspace_root};

/// Build profile whose binaries are smoked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Profile {
    Debug,
    Release,
}

impl Profile {
    const fn dir_name(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

/// One of the four binaries the smoke requires to be present.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Bin {
    Rocm,
    Rocmd,
    Lemonade,
    Vllm,
}

impl Bin {
    /// Short name used in the "missing smoke binary" failure.
    const fn key(self) -> &'static str {
        match self {
            Self::Rocm => "rocm",
            Self::Rocmd => "rocmd",
            Self::Lemonade => "lemonade",
            Self::Vllm => "vllm",
        }
    }

    /// File stem of the built artifact, before the platform `.exe` suffix.
    const fn stem(self) -> &'static str {
        match self {
            Self::Rocm => "rocm",
            Self::Rocmd => "rocmd",
            Self::Lemonade => "rocm-engine-lemonade",
            Self::Vllm => "rocm-engine-vllm",
        }
    }

    const ALL: [Self; 4] = [Self::Rocm, Self::Rocmd, Self::Lemonade, Self::Vllm];
}

/// Resolved paths to the binaries under test.
#[derive(Debug, PartialEq, Eq)]
struct Binaries {
    dir: PathBuf,
}

impl Binaries {
    const fn new(binary_dir: PathBuf) -> Self {
        Self { dir: binary_dir }
    }

    fn path(&self, bin: Bin) -> PathBuf {
        self.dir.join(binary_name(bin.stem()))
    }

    /// Fail naming the first binary that is not a file, as the gate is
    /// pointless against a partial build.
    fn require_all_present(&self) -> Result<()> {
        for bin in Bin::ALL {
            let path = self.path(bin);
            if !path.is_file() {
                bail!("missing smoke binary {}: {}", bin.key(), path.display());
            }
        }
        Ok(())
    }
}

/// A JSON-shaped reply whose payload identity is asserted, rather than a
/// substring of the rendered text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonCheck {
    /// Must parse as JSON; nothing further is asserted (`vllm detect`).
    Parses,
    /// An object whose `protocol` field equals this value.
    Protocol(&'static str),
    /// A sandbox reply naming `tool` and reporting `ok`. `noun` is the word the
    /// failure uses, which the script varied per tool.
    SandboxTool {
        tool: &'static str,
        noun: &'static str,
    },
    /// vLLM capabilities: OpenAI-compatible serving, and not CPU.
    VllmCapabilities,
}

impl JsonCheck {
    /// Assert the parsed reply, naming what was wrong with it.
    fn assert_value(self, value: &Value) -> Result<()> {
        match self {
            Self::Parses => Ok(()),
            Self::Protocol(expected) => {
                if value.is_object() && json_field_is(value, "protocol", expected) {
                    Ok(())
                } else {
                    bail!("unexpected bridge snapshot protocol: {value}")
                }
            }
            Self::SandboxTool { tool, noun } => {
                if sandbox_reply_ok(value, tool) {
                    Ok(())
                } else {
                    bail!("unexpected sandbox {noun} result: {value}")
                }
            }
            Self::VllmCapabilities => {
                if vllm_capabilities_ok(value) {
                    Ok(())
                } else {
                    bail!("vllm capabilities did not report GPU-only OpenAI serving: {value}")
                }
            }
        }
    }
}

/// A single command run plus the expectations its output must satisfy.
///
/// Holding these as data rather than a straight line of calls is what lets the
/// table be checked by unit tests (no duplicate labels, every expectation
/// non-empty) without spawning a single process.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Check {
    label: &'static str,
    bin: Bin,
    args: Vec<&'static str>,
    expect_failure: bool,
    contains: Vec<&'static str>,
    contains_any: Vec<&'static str>,
    not_contains: Vec<&'static str>,
    json: Option<JsonCheck>,
}

impl Check {
    fn new(label: &'static str, bin: Bin, args: &[&'static str]) -> Self {
        Self {
            label,
            bin,
            args: args.to_vec(),
            expect_failure: false,
            contains: Vec::new(),
            contains_any: Vec::new(),
            not_contains: Vec::new(),
            json: None,
        }
    }

    /// The command is expected to exit non-zero; a success is the failure.
    const fn failing(mut self) -> Self {
        self.expect_failure = true;
        self
    }

    fn contains(mut self, needle: &'static str) -> Self {
        self.contains.push(needle);
        self
    }

    /// At least one of `needles` must appear — for surfaces with more than one
    /// accepted wording.
    fn contains_any(mut self, needles: &[&'static str]) -> Self {
        self.contains_any.extend_from_slice(needles);
        self
    }

    fn not_contains(mut self, needle: &'static str) -> Self {
        self.not_contains.push(needle);
        self
    }

    /// The reply is parsed as JSON and asserted as a value.
    const fn json(mut self, check: JsonCheck) -> Self {
        self.json = Some(check);
        self
    }

    fn assert_output(&self, output: &str) -> Result<()> {
        for needle in &self.contains {
            assert_contains(output, needle, self.label)?;
        }
        if !self.contains_any.is_empty() {
            assert_contains_any(output, &self.contains_any, self.label)?;
        }
        for needle in &self.not_contains {
            assert_not_contains(output, needle, self.label)?;
        }
        if let Some(json) = self.json {
            json.assert_value(&parse_json(output, self.label)?)?;
        }
        Ok(())
    }
}

fn assert_contains(text: &str, needle: &str, label: &str) -> Result<()> {
    if text.contains(needle) {
        return Ok(());
    }
    bail!("{label} did not contain expected text: {needle}\n{text}")
}

fn assert_contains_any(text: &str, needles: &[&str], label: &str) -> Result<()> {
    if needles.iter().any(|needle| text.contains(needle)) {
        return Ok(());
    }
    let expected = needles
        .iter()
        .map(|needle| format!("'{needle}'"))
        .collect::<Vec<_>>()
        .join(" or ");
    bail!("{label} did not contain expected text: {expected}\n{text}")
}

fn assert_not_contains(text: &str, needle: &str, label: &str) -> Result<()> {
    if text.contains(needle) {
        bail!("{label} contained unexpected text: {needle}\n{text}");
    }
    Ok(())
}

fn assert_path_missing(path: &Path, label: &str) -> Result<()> {
    if path.exists() {
        bail!("{label} unexpectedly exists: {}", path.display());
    }
    Ok(())
}

/// Accept only the traceable build string `rocm-cli <version> (<ref>, <hash>)`,
/// and reject the placeholder `unknown` ref a build with no resolvable
/// tag/branch emits.
fn assert_version_string(text: &str, label: &str) -> Result<()> {
    let Some((version, git_ref, hash)) = parse_version_line(text) else {
        bail!("{label} did not match 'rocm-cli <version> (<ref>, <hash>)':\n{text}");
    };
    debug_assert!(!version.is_empty() && !hash.is_empty());
    if git_ref == "unknown" {
        bail!("{label} has an unresolved ref ('unknown'), expected a real tag/branch:\n{text}");
    }
    Ok(())
}

/// Pure parser behind [`assert_version_string`], mirroring the script's
/// `rocm-cli (\S+) \((\S+), ([0-9a-f]+)\)` full match.
fn parse_version_line(text: &str) -> Option<(&str, &str, &str)> {
    let rest = text.trim().strip_prefix("rocm-cli ")?;
    let (version, rest) = rest.split_once(' ')?;
    let inner = rest.strip_prefix('(')?.strip_suffix(')')?;
    let (git_ref, hash) = inner.split_once(", ")?;
    let fields_are_words = [version, git_ref]
        .iter()
        .all(|field| !field.is_empty() && !field.contains(char::is_whitespace));
    let hash_is_lower_hex = !hash.is_empty()
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    (fields_are_words && hash_is_lower_hex).then_some((version, git_ref, hash))
}

/// Python-style truthiness for a JSON field, so the sandbox/capability checks
/// accept exactly what the script accepted.
fn json_truthy(value: &Value, field: &str) -> bool {
    match value.get(field) {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|value| value != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(entries)) => !entries.is_empty(),
    }
}

fn json_field_is(value: &Value, field: &str, expected: &str) -> bool {
    value.get(field).and_then(Value::as_str) == Some(expected)
}

/// A sandbox tool reply is accepted only when it names the tool that was asked
/// for and reports success.
fn sandbox_reply_ok(value: &Value, tool: &str) -> bool {
    value.is_object() && json_field_is(value, "tool", tool) && json_truthy(value, "ok")
}

/// vLLM must advertise OpenAI-compatible serving and must not advertise CPU.
fn vllm_capabilities_ok(value: &Value) -> bool {
    value.is_object() && json_truthy(value, "openai_compatible") && !json_truthy(value, "cpu")
}

fn parse_json(text: &str, label: &str) -> Result<Value> {
    serde_json::from_str(text)
        .with_context(|| format!("{label} did not return valid JSON:\n{text}"))
}

/// Environment overrides that confine a smoke run to `smoke_root`, so it can
/// neither read nor write the developer's real ROCm CLI state.
fn isolated_env(smoke_root: &Path) -> BTreeMap<&'static str, OsString> {
    let mut env = BTreeMap::new();
    env.insert("ROCM_CLI_UPDATE_USER_PATH", OsString::from("0"));
    env.insert(
        "ROCM_CLI_CONFIG_DIR",
        smoke_root.join("rocm-config").into_os_string(),
    );
    env.insert(
        "ROCM_CLI_DATA_DIR",
        smoke_root.join("rocm-data").into_os_string(),
    );
    env.insert(
        "ROCM_CLI_CACHE_DIR",
        smoke_root.join("rocm-cache").into_os_string(),
    );
    if cfg!(windows) {
        env.insert("APPDATA", smoke_root.join("appdata").into_os_string());
        env.insert(
            "LOCALAPPDATA",
            smoke_root.join("localappdata").into_os_string(),
        );
    } else {
        env.insert(
            "XDG_CONFIG_HOME",
            smoke_root.join("xdg-config").into_os_string(),
        );
        env.insert(
            "XDG_DATA_HOME",
            smoke_root.join("xdg-data").into_os_string(),
        );
        env.insert(
            "XDG_CACHE_HOME",
            smoke_root.join("xdg-cache").into_os_string(),
        );
        env.insert("HOME", smoke_root.join("home").into_os_string());
    }
    env
}

/// Bind an ephemeral port and hand back the number, so the serve-rejection
/// checks name a port nothing else on the host is listening on.
fn free_tcp_port() -> Result<u16> {
    let listener =
        TcpListener::bind(("127.0.0.1", 0)).context("failed to bind an ephemeral port")?;
    Ok(listener
        .local_addr()
        .context("failed to read the bound port")?
        .port())
}

/// Spawn one child and return its combined output, failing when the exit status
/// does not match `expect_failure`.
///
/// Every child is handed a null stdin rather than inheriting this process's:
/// `rocm chat` without `--prompt` reads a piped stdin to EOF, so an inherited
/// pipe that the harness keeps open would hang the gate depending only on how
/// it was launched.
fn run_command(
    label: &str,
    program: &Path,
    args: &[&str],
    env: &BTreeMap<&'static str, OsString>,
    cwd: &Path,
    expect_failure: bool,
) -> Result<String> {
    println!("smoke: {label}");
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command
        .output()
        .with_context(|| format!("failed to run {}", program.display()))?;

    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    let combined = combined.trim().to_string();
    if !combined.is_empty() {
        println!("{combined}");
    }

    if expect_failure {
        if output.status.success() {
            bail!("{label} unexpectedly succeeded");
        }
    } else if !output.status.success() {
        bail!(
            "{label} exited with status {}",
            exit_code_text(&output.status)
        );
    }
    Ok(combined)
}

fn exit_code_text(status: &std::process::ExitStatus) -> String {
    status
        .code()
        .map_or_else(|| "signal".to_string(), |code| code.to_string())
}

/// Every check whose command line is fixed, in the exact order the script ran
/// them. Order is part of the contract: each runs against the same throwaway
/// state root, so a command moved earlier or later sees different state.
///
/// The two serve rejections are not here because they need a port chosen at run
/// time; they are driven at the end of [`run`], which is where they ran.
fn checks() -> Vec<Check> {
    let mut checks = checks_before_platform_gate();
    if cfg!(windows) {
        checks.extend(windows_only_checks());
    }
    checks.extend(checks_after_platform_gate());
    checks
}

fn checks_before_platform_gate() -> Vec<Check> {
    vec![
        Check::new("rocm examine", Bin::Rocm, &["examine"])
            .contains("rocm examine")
            .contains("default_engine:")
            .contains("managed_runtimes: 0")
            .contains("managed_services: 0"),
        Check::new("rocm engines list", Bin::Rocm, &["engines", "list"])
            .contains("lemonade")
            .contains("vllm"),
        Check::new(
            "rocm config set telemetry off",
            Bin::Rocm,
            &["config", "set-telemetry", "off"],
        )
        .contains("telemetry mode set to off")
        .contains("policy: disabled"),
        Check::new("rocm config show", Bin::Rocm, &["config", "show"])
            .contains("telemetry_mode: off")
            .contains("telemetry_policy: disabled"),
        Check::new(
            "rocm engines install requires exact runtime",
            Bin::Rocm,
            &["engines", "install", "vllm"],
        )
        .failing()
        .contains("no active ROCm runtime is configured"),
        Check::new(
            "rocm chat local status",
            Bin::Rocm,
            &["chat", "--provider", "local"],
        )
        .contains_any(&[
            "Provider: local",
            "Assistant source: local model on this computer",
        ]),
        Check::new(
            "rocm freeform installed status question",
            Bin::Rocm,
            &["is rocm installed?"],
        )
        .contains("ROCm status")
        .contains("Nothing was changed.")
        .not_contains("No ROCm action selected"),
        Check::new(
            "rocm freeform comfyui help question",
            Bin::Rocm,
            &["how do i setup comfyui"],
        )
        .contains("ComfyUI status")
        .contains("Nothing was changed.")
        .not_contains("No ROCm action selected"),
        Check::new(
            "rocm freeform comfyui install request",
            Bin::Rocm,
            &["can you setup comfyui for me"],
        )
        .contains("Install ComfyUI")
        .contains("approval: required"),
        Check::new(
            "rocm freeform vllm plan",
            Bin::Rocm,
            &["serve qwen with vllm"],
        )
        .contains("engine: vllm")
        .contains("no CPU fallback is implied"),
        Check::new(
            "rocm freeform tiny gpu recipe plan",
            Bin::Rocm,
            &["serve tiny-gpt2"],
        )
        .contains("model: sshleifer/tiny-gpt2")
        .contains("device_policy: gpu_required")
        .contains("--device gpu_required")
        .contains("approval: required"),
    ]
}

/// The Windows-only check: a tarball SDK install has no Windows path and must
/// be refused rather than half-attempted.
fn windows_only_checks() -> Vec<Check> {
    vec![
        Check::new(
            "windows tarball sdk rejection",
            Bin::Rocm,
            &["install", "sdk", "--format", "tarball", "--dry-run"],
        )
        .failing()
        .contains("TheRock tarball installs are not supported on Windows"),
    ]
}

fn checks_after_platform_gate() -> Vec<Check> {
    vec![
        Check::new("rocmd status", Bin::Rocmd, &["status"]).contains("rocmd status"),
        Check::new("rocmd bridge snapshot", Bin::Rocmd, &["bridge-snapshot"])
            .json(JsonCheck::Protocol("rocmd-codex-bridge-v0")),
        Check::new(
            "rocmd sandbox examine snapshot",
            Bin::Rocmd,
            &["sandbox-run", "examine_snapshot", "--allow-native-fallback"],
        )
        .json(JsonCheck::SandboxTool {
            tool: "examine_snapshot",
            noun: "examine",
        }),
        Check::new(
            "rocmd sandbox list servers",
            Bin::Rocmd,
            &["sandbox-run", "list_servers", "--allow-native-fallback"],
        )
        .json(JsonCheck::SandboxTool {
            tool: "list_servers",
            noun: "list",
        }),
        Check::new(
            "rocmd sandbox prefetch validation",
            Bin::Rocmd,
            &[
                "sandbox-run",
                "prefetch_artifact",
                "--allow-native-fallback",
            ],
        )
        .failing()
        .contains("prefetch_artifact requires"),
        Check::new("vllm detect", Bin::Vllm, &["detect"]).json(JsonCheck::Parses),
        Check::new("vllm capabilities", Bin::Vllm, &["capabilities"])
            .json(JsonCheck::VllmCapabilities),
        Check::new("vllm resolve qwen", Bin::Vllm, &["resolve-model", "qwen"]).contains("qwen"),
        Check::new(
            "vllm reject cpu",
            Bin::Vllm,
            &["resolve-model", "qwen", "--device-policy", "cpu_only"],
        )
        .failing()
        .contains("no CPU fallback is used"),
    ]
}

/// Build the workspace, then run every smoke check against the built binaries
/// in an isolated state root.
pub fn run(profile: Profile, skip_build: bool, target_dir_override: Option<PathBuf>) -> Result<()> {
    let root = workspace_root()?;
    let env_root = target_dir(&root).join("smoke-local");
    let env = isolated_env(&env_root);

    if !skip_build {
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
        let cargo = PathBuf::from(cargo);
        // `--all-targets` matches the script: the local gate also catches a
        // test target that no longer compiles, not just the four binaries.
        run_command(
            "build workspace all targets",
            &cargo,
            &["build", "--workspace", "--all-targets"],
            &env,
            &root,
            false,
        )?;
    }

    let binary_dir = match target_dir_override {
        Some(dir) if dir.is_absolute() => dir,
        Some(dir) => root.join(dir),
        None => target_dir(&root),
    }
    .join(profile.dir_name());
    let binaries = Binaries::new(binary_dir);
    binaries.require_all_present()?;

    if env_root.exists() {
        std::fs::remove_dir_all(&env_root)
            .with_context(|| format!("failed to clear {}", env_root.display()))?;
    }
    std::fs::create_dir_all(&env_root)
        .with_context(|| format!("failed to create {}", env_root.display()))?;

    let reject_port = free_tcp_port()?.to_string();
    let rocm = binaries.path(Bin::Rocm);

    // The three version surfaces must agree on the build string, and `rocm
    // version` reports the active SDK and driver on top of it.
    let version_flag = run_command("rocm --version", &rocm, &["--version"], &env, &root, false)?;
    let version_short = run_command("rocm -V", &rocm, &["-V"], &env, &root, false)?;
    if version_flag != version_short {
        bail!(
            "version flag surfaces returned different output: {version_flag:?} vs {version_short:?}"
        );
    }
    let version_command = run_command("rocm version", &rocm, &["version"], &env, &root, false)?;
    let build_line = version_command.lines().next().unwrap_or_default();
    if build_line != version_flag {
        bail!(
            "`rocm version`'s build line does not match `-V`/`--version`: {build_line:?} vs {version_flag:?}"
        );
    }
    assert_version_string(build_line, "rocm version")?;
    assert_contains(&version_command, "ROCm SDK:", "rocm version")?;
    assert_contains(&version_command, "GPU driver:", "rocm version")?;

    for check in &checks() {
        let program = binaries.path(check.bin);
        let output = run_command(
            check.label,
            &program,
            &check.args,
            &env,
            &root,
            check.expect_failure,
        )?;
        check.assert_output(&output)?;
    }

    // The point of the whole gate: a GPU-required serve must refuse rather than
    // quietly running on CPU, and an explicit CPU serve must be refused too.
    let gpu_required = run_command(
        "vllm reject required gpu",
        &rocm,
        &[
            "serve",
            "qwen",
            "--engine",
            "vllm",
            "--device",
            "gpu_required",
            "--foreground",
            "--port",
            &reject_port,
        ],
        &env,
        &root,
        true,
    )?;
    assert_contains(&gpu_required, "gpu_required", "vllm reject required gpu")?;
    assert_not_contains(&gpu_required, "CPU fallback", "vllm reject required gpu")?;

    let cpu_serve = run_command(
        "rocm vllm reject cpu serve",
        &rocm,
        &[
            "serve",
            "qwen",
            "--engine",
            "vllm",
            "--device",
            "cpu",
            "--foreground",
            "--port",
            &reject_port,
        ],
        &env,
        &root,
        true,
    )?;
    assert_contains(
        &cpu_serve,
        "CPU mode is not a fallback path",
        "rocm vllm reject cpu serve",
    )?;

    // Nothing above should have provisioned anything: a first run that quietly
    // created a uv cache or a runtime registry would mean a command took an
    // install path it was never asked to take.
    assert_path_missing(
        &env_root.join("rocm-cache").join("uv"),
        "first-run smoke uv cache",
    )?;
    assert_path_missing(
        &env_root.join("rocm-data").join("runtimes").join("registry"),
        "first-run smoke runtime registry",
    )?;

    println!("smoke: ok");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_names_cover_every_required_artifact() {
        let binaries = Binaries::new(PathBuf::from("/build/debug"));
        let stems: Vec<String> = Bin::ALL
            .iter()
            .map(|bin| {
                binaries
                    .path(*bin)
                    .file_name()
                    .expect("file name")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(
            stems,
            vec![
                binary_name("rocm"),
                binary_name("rocmd"),
                binary_name("rocm-engine-lemonade"),
                binary_name("rocm-engine-vllm"),
            ]
        );
    }

    #[test]
    fn missing_binary_is_named_with_its_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let binaries = Binaries::new(dir.path().to_path_buf());
        let error = binaries
            .require_all_present()
            .expect_err("an empty dir has no binaries");
        let message = format!("{error}");
        assert!(message.contains("missing smoke binary rocm"), "{message}");
        assert!(message.contains(&binary_name("rocm")), "{message}");
    }

    #[test]
    fn profile_selects_the_cargo_output_dir() {
        assert_eq!(Profile::Debug.dir_name(), "debug");
        assert_eq!(Profile::Release.dir_name(), "release");
    }

    /// Order is part of the contract, not an accident of how the table was
    /// written: every check shares one throwaway state root, so a command moved
    /// earlier or later sees different state. This pins the sequence the
    /// `scripts/smoke_local.py` gate ran, minus the platform-gated entry and the
    /// two port-using serve rejections that `run` drives at the end.
    #[test]
    fn the_check_sequence_matches_the_gate_it_replaces() {
        let labels: Vec<&str> = checks_before_platform_gate()
            .iter()
            .chain(checks_after_platform_gate().iter())
            .map(|check| check.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "rocm examine",
                "rocm engines list",
                "rocm config set telemetry off",
                "rocm config show",
                "rocm engines install requires exact runtime",
                "rocm chat local status",
                "rocm freeform installed status question",
                "rocm freeform comfyui help question",
                "rocm freeform comfyui install request",
                "rocm freeform vllm plan",
                "rocm freeform tiny gpu recipe plan",
                "rocmd status",
                "rocmd bridge snapshot",
                "rocmd sandbox examine snapshot",
                "rocmd sandbox list servers",
                "rocmd sandbox prefetch validation",
                "vllm detect",
                "vllm capabilities",
                "vllm resolve qwen",
                "vllm reject cpu",
            ]
        );
    }

    /// The platform-gated check runs between the freeform plans and the `rocmd`
    /// checks, where the script ran it — not appended at the end.
    #[test]
    fn the_windows_check_sits_between_the_two_halves() {
        let before = checks_before_platform_gate();
        assert_eq!(
            before.last().expect("a check").label,
            "rocm freeform tiny gpu recipe plan"
        );
        assert_eq!(
            checks_after_platform_gate().first().expect("a check").label,
            "rocmd status"
        );
        if cfg!(windows) {
            let labels: Vec<&str> = checks().iter().map(|check| check.label).collect();
            let gated = labels
                .iter()
                .position(|label| *label == "windows tarball sdk rejection")
                .expect("the windows check is in the table");
            assert_eq!(gated, before.len());
        }
    }

    #[test]
    fn check_labels_are_unique_across_the_table() {
        let mut labels: Vec<&str> = checks_before_platform_gate()
            .iter()
            .chain(windows_only_checks().iter())
            .chain(checks_after_platform_gate().iter())
            .map(|check| check.label)
            .collect();
        let total = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), total, "duplicate check labels: {labels:?}");
    }

    #[test]
    fn every_check_asserts_something() {
        for check in checks_before_platform_gate()
            .iter()
            .chain(windows_only_checks().iter())
            .chain(checks_after_platform_gate().iter())
        {
            assert!(
                !check.contains.is_empty()
                    || !check.contains_any.is_empty()
                    || !check.not_contains.is_empty()
                    || check.json.is_some(),
                "{} runs a command but asserts nothing",
                check.label
            );
            assert!(!check.args.is_empty(), "{} has no arguments", check.label);
        }
    }

    #[test]
    fn the_no_cpu_fallback_checks_are_in_the_table() {
        let labels: Vec<&str> = checks().iter().map(|check| check.label).collect();
        for required in [
            "rocm engines install requires exact runtime",
            "vllm reject cpu",
            "rocm freeform vllm plan",
        ] {
            assert!(labels.contains(&required), "{required} is not smoked");
        }
    }

    #[test]
    fn contains_failure_names_the_label_and_the_output() {
        let check = Check::new("demo", Bin::Rocm, &["examine"]).contains("expected text");
        let error = check
            .assert_output("actual output")
            .expect_err("the needle is absent");
        let message = format!("{error}");
        assert!(
            message.contains("demo did not contain expected text"),
            "{message}"
        );
        assert!(message.contains("actual output"), "{message}");
    }

    #[test]
    fn contains_any_accepts_either_wording_and_reports_both() {
        let check = Check::new("demo", Bin::Rocm, &["chat"]).contains_any(&["first", "second"]);
        check
            .assert_output("has second here")
            .expect("second matches");
        check
            .assert_output("has first here")
            .expect("first matches");

        let error = check
            .assert_output("neither")
            .expect_err("no wording matches");
        let message = format!("{error}");
        assert!(message.contains("'first' or 'second'"), "{message}");
    }

    #[test]
    fn not_contains_rejects_the_forbidden_text() {
        let check = Check::new("demo", Bin::Rocm, &["serve"]).not_contains("CPU fallback");
        check.assert_output("gpu_required").expect("absent is fine");

        let error = check
            .assert_output("used CPU fallback instead")
            .expect_err("the forbidden text is present");
        assert!(
            format!("{error}").contains("demo contained unexpected text: CPU fallback"),
            "{error}"
        );
    }

    #[test]
    fn version_line_parses_a_traceable_build_string() {
        assert_eq!(
            parse_version_line("rocm-cli 0.2.0 (main, a1b2c3d)"),
            Some(("0.2.0", "main", "a1b2c3d"))
        );
        assert_eq!(
            parse_version_line("  rocm-cli 1.0.0-rc.1 (v1.0.0, 0123456789abcdef)  "),
            Some(("1.0.0-rc.1", "v1.0.0", "0123456789abcdef"))
        );
    }

    #[test]
    fn version_line_rejects_malformed_and_uppercase_hashes() {
        for text in [
            "rocm-cli 0.2.0",
            "rocm 0.2.0 (main, a1b2c3d)",
            "rocm-cli 0.2.0 (main a1b2c3d)",
            "rocm-cli 0.2.0 (main, A1B2C3D)",
            "rocm-cli 0.2.0 (main, zzz)",
            "rocm-cli 0.2.0 (main, )",
            "rocm-cli  (main, a1b2c3d)",
        ] {
            assert_eq!(parse_version_line(text), None, "{text} should not parse");
        }
    }

    #[test]
    fn an_unresolved_ref_fails_even_though_it_parses() {
        assert!(parse_version_line("rocm-cli 0.2.0 (unknown, a1b2c3d)").is_some());
        let error = assert_version_string("rocm-cli 0.2.0 (unknown, a1b2c3d)", "rocm version")
            .expect_err("an unknown ref is rejected");
        assert!(format!("{error}").contains("unresolved ref"), "{error}");
        assert_version_string("rocm-cli 0.2.0 (main, a1b2c3d)", "rocm version")
            .expect("a real ref");
    }

    /// Each JSON variant must reject a wrong payload, and say which check and
    /// which payload was wrong — these are the only failures where the output
    /// is a blob rather than prose a reader can scan.
    #[test]
    fn each_json_check_rejects_a_wrong_payload_by_name() {
        let cases = [
            (
                JsonCheck::Protocol("rocmd-codex-bridge-v0"),
                serde_json::json!({"protocol": "something-else"}),
                "unexpected bridge snapshot protocol",
            ),
            (
                JsonCheck::SandboxTool {
                    tool: "examine_snapshot",
                    noun: "examine",
                },
                serde_json::json!({"tool": "examine_snapshot", "ok": false}),
                "unexpected sandbox examine result",
            ),
            (
                JsonCheck::SandboxTool {
                    tool: "list_servers",
                    noun: "list",
                },
                serde_json::json!({"tool": "list_servers", "ok": false}),
                "unexpected sandbox list result",
            ),
            (
                JsonCheck::VllmCapabilities,
                serde_json::json!({"openai_compatible": true, "cpu": true}),
                "vllm capabilities did not report GPU-only OpenAI serving",
            ),
        ];
        for (check, payload, expected) in cases {
            let error = check
                .assert_value(&payload)
                .expect_err("the payload should be rejected");
            let message = format!("{error}");
            assert!(message.contains(expected), "{message}");
        }
    }

    #[test]
    fn a_parses_only_check_accepts_any_valid_json() {
        JsonCheck::Parses
            .assert_value(&serde_json::json!({"anything": 1}))
            .expect("an object parses");
        JsonCheck::Parses
            .assert_value(&serde_json::json!([]))
            .expect("an array parses too — `vllm detect` only has to be JSON");
    }

    /// A reply that is valid JSON but not an object must not slip past the
    /// identity checks.
    #[test]
    fn a_non_object_reply_is_rejected() {
        for check in [
            JsonCheck::Protocol("rocmd-codex-bridge-v0"),
            JsonCheck::VllmCapabilities,
        ] {
            assert!(check.assert_value(&serde_json::json!("a string")).is_err());
            assert!(check.assert_value(&serde_json::json!([1, 2])).is_err());
        }
    }

    #[test]
    fn json_truthiness_matches_the_script() {
        let value = serde_json::json!({
            "yes": true,
            "no": false,
            "zero": 0,
            "one": 1,
            "empty_text": "",
            "text": "x",
            "empty_list": [],
            "list": [1],
            "null": null,
        });
        for field in ["yes", "one", "text", "list"] {
            assert!(json_truthy(&value, field), "{field} should be truthy");
        }
        for field in ["no", "zero", "empty_text", "empty_list", "null", "absent"] {
            assert!(!json_truthy(&value, field), "{field} should be falsy");
        }
    }

    #[test]
    fn a_sandbox_reply_must_name_its_own_tool_and_report_ok() {
        let good = serde_json::json!({"tool": "list_servers", "ok": true});
        assert!(sandbox_reply_ok(&good, "list_servers"));
        assert!(
            !sandbox_reply_ok(&good, "examine_snapshot"),
            "a reply for another tool is not an answer"
        );

        let not_ok = serde_json::json!({"tool": "list_servers", "ok": false});
        assert!(!sandbox_reply_ok(&not_ok, "list_servers"));

        let missing_ok = serde_json::json!({"tool": "list_servers"});
        assert!(!sandbox_reply_ok(&missing_ok, "list_servers"));

        assert!(!sandbox_reply_ok(&serde_json::json!([]), "list_servers"));
    }

    #[test]
    fn vllm_capabilities_must_be_openai_compatible_and_not_cpu() {
        assert!(vllm_capabilities_ok(&serde_json::json!({
            "openai_compatible": true,
            "cpu": false,
        })));
        assert!(vllm_capabilities_ok(
            &serde_json::json!({"openai_compatible": true})
        ));
        assert!(
            !vllm_capabilities_ok(&serde_json::json!({"openai_compatible": true, "cpu": true})),
            "advertising CPU is the regression this check exists for"
        );
        assert!(!vllm_capabilities_ok(&serde_json::json!({
            "openai_compatible": false,
            "cpu": false,
        })));
    }

    #[test]
    fn parse_json_failure_names_the_label() {
        let error = parse_json("not json", "vllm detect").expect_err("invalid JSON");
        assert!(
            format!("{error}").contains("vllm detect did not return valid JSON"),
            "{error}"
        );
    }

    #[test]
    fn isolated_env_confines_every_state_dir_to_the_smoke_root() {
        let root = Path::new("/tmp/smoke-local");
        let env = isolated_env(root);

        assert_eq!(env["ROCM_CLI_UPDATE_USER_PATH"], OsString::from("0"));
        for key in [
            "ROCM_CLI_CONFIG_DIR",
            "ROCM_CLI_DATA_DIR",
            "ROCM_CLI_CACHE_DIR",
        ] {
            let value = PathBuf::from(&env[key]);
            assert!(
                value.starts_with(root),
                "{key} escaped the smoke root: {value:?}"
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn isolated_env_redirects_the_windows_profile_dirs() {
        let env = isolated_env(Path::new("C:\\smoke"));
        for key in ["APPDATA", "LOCALAPPDATA"] {
            assert!(env.contains_key(key), "{key} is not redirected");
        }
        assert!(!env.contains_key("HOME"));
    }

    // Unix-only: the XDG/HOME arm is compiled out on Windows, where the branch
    // above applies instead. Supported hosts are Windows and Linux only.
    #[cfg(unix)]
    #[test]
    fn isolated_env_redirects_the_xdg_dirs_and_home() {
        let env = isolated_env(Path::new("/tmp/smoke-local"));
        for key in ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "HOME"] {
            assert!(env.contains_key(key), "{key} is not redirected");
        }
        assert!(!env.contains_key("APPDATA"));
    }

    /// A child killed by a signal has no exit code, and the gate's failure must
    /// still say something rather than render an empty status.
    #[cfg(unix)]
    #[test]
    fn a_signalled_child_is_reported_as_a_signal() {
        use std::os::unix::process::ExitStatusExt as _;

        let signalled = std::process::ExitStatus::from_raw(9);
        assert_eq!(exit_code_text(&signalled), "signal");
    }

    #[test]
    fn an_exit_code_is_reported_verbatim() {
        let status = Command::new(if cfg!(windows) { "cmd" } else { "sh" })
            .args(if cfg!(windows) {
                vec!["/C", "exit 3"]
            } else {
                vec!["-c", "exit 3"]
            })
            .status()
            .expect("spawn a shell");
        assert_eq!(exit_code_text(&status), "3");
    }

    #[test]
    fn a_free_port_is_outside_the_reserved_range() {
        let port = free_tcp_port().expect("an ephemeral port");
        assert!(port > 0);
    }

    #[test]
    fn path_missing_rejects_a_path_that_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_path_missing(&dir.path().join("absent"), "first-run smoke uv cache")
            .expect("an absent path passes");

        let error = assert_path_missing(dir.path(), "first-run smoke uv cache")
            .expect_err("an existing path fails");
        assert!(
            format!("{error}").contains("first-run smoke uv cache unexpectedly exists"),
            "{error}"
        );
    }
}
