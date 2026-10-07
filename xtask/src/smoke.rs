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
//! assertions, on the same commands, in the same order. Where it differs from
//! that script, the difference is deliberate:
//!
//! * The binaries are looked for with [`crate::paths::target_dir`], so a
//!   configured `CARGO_TARGET_DIR` is honoured. The script always looked under
//!   `<root>/target` and so reported every binary missing when that variable
//!   pointed elsewhere. The throwaway state root stays at
//!   `<root>/target/smoke-local`, as in the script, because it is wiped on every
//!   run and a target directory may be shared between checkouts.
//! * `--profile release` builds release binaries, and `--target-dir` is handed
//!   to the build as well as the lookup. The script always built debug, so both
//!   flags built one place and then looked in another; see [`build_args`].
//! * A child killed by a signal is reported with the negated signal number
//!   (`-9`), as the script's `returncode` was; see [`exit_code_text`].
//! * The build runs in this process's own environment, not the isolated one.
//!   The script built under the isolated environment too, where the redirected
//!   `HOME` hid cargo's registry and config from the build; no build script
//!   reads a variable the isolation sets. The build uses the `cargo` this task
//!   runs under (`$CARGO`, which `cargo xtask` always sets), where the script
//!   searched `PATH` and then `~/.cargo/bin`.
//! * The script merged each child's stderr into its stdout pipe, truly
//!   interleaving them. `std::process` has no portable equivalent, so the two
//!   streams are captured separately and concatenated (stdout first). No check
//!   depends on the seam: each asserts a substring, reads the first line of the
//!   captured output, compares whole outputs for equality (the version
//!   surfaces), or requires the whole of it to be one JSON value. (On
//!   Linux, each command smoked here was also measured writing to only one of
//!   the two.)
//!
//! Smaller differences that change no result today: output is decoded lossily
//! and keeps Windows' `\r` until the line comparisons trim it, where the script
//! decoded strictly and translated newlines; on Windows a crashed child's status
//! prints as a signed code (`-1073741819`), where Python printed it unsigned
//! (`3221225477`); the children inherit the variables `cargo xtask` adds to the
//! environment; and the `managed_runtimes`/`managed_services` failures name the
//! check, `rocm examine`, where the script said `rocm examine first-run state`.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::paths::{binary_name, workspace_root};

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
                if json_field_is(value, "protocol", expected) {
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

/// Token in a check's argv replaced with the ephemeral port chosen for this
/// run. The two serve rejections need a port nothing is listening on, and
/// keeping them in the table is what puts them under the order and expectation
/// pins — they are the checks the gate exists for.
const PORT_TOKEN: &str = "{port}";

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

    /// Argv with [`PORT_TOKEN`] resolved against the port chosen for this run.
    fn resolved_args(&self, port: &str) -> Vec<String> {
        self.args
            .iter()
            .map(|arg| {
                if *arg == PORT_TOKEN {
                    port.to_string()
                } else {
                    (*arg).to_string()
                }
            })
            .collect()
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
    let Some((_version, git_ref, _hash)) = parse_version_line(text) else {
        bail!("{label} did not match 'rocm-cli <version> (<ref>, <hash>)':\n{text}");
    };
    if git_ref == "unknown" {
        // The ref comes from the build script's `git describe`/`rev-parse`, so a
        // detached HEAD or a git-stripped tarball yields `unknown` even though
        // nothing is wrong with the binary. Name that, or the gate reads as a
        // regression during a bisect.
        bail!(
            "{label} has an unresolved ref ('unknown'), expected a real tag/branch. \
             A detached HEAD or a source tree without git metadata produces this; \
             build from a branch or tag:\n{text}"
        );
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
    // No `is_object` guard is needed: `Value::get` on any non-object is `None`,
    // so a bare string or array fails both fields below.
    json_field_is(value, "tool", tool) && json_truthy(value, "ok")
}

/// vLLM must advertise OpenAI-compatible serving and must not advertise CPU.
fn vllm_capabilities_ok(value: &Value) -> bool {
    json_truthy(value, "openai_compatible") && !json_truthy(value, "cpu")
}

fn parse_json(text: &str, label: &str) -> Result<Value> {
    serde_json::from_str(text)
        .with_context(|| format!("{label} did not return valid JSON:\n{text}"))
}

/// Environment overrides that point the CLI's own state directories, and the
/// platform profile directories under them, at `smoke_root`.
///
/// This is the script's allowlist, not a sandbox: a `ROCM_CLI_*` variable
/// outside the four below is still inherited, so an exported one can change
/// what the gate sees.
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

/// The cache and data directories the smoke children were pointed at.
fn state_dirs(env: &BTreeMap<&'static str, OsString>) -> (PathBuf, PathBuf) {
    (
        PathBuf::from(&env["ROCM_CLI_CACHE_DIR"]),
        PathBuf::from(&env["ROCM_CLI_DATA_DIR"]),
    )
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

/// The script printed Python's `returncode`, which is the negated signal number
/// for a child that was killed. Keep that, so an OOM kill and a segfault stay
/// distinguishable.
fn exit_code_text(status: &std::process::ExitStatus) -> String {
    if let Some(code) = status.code() {
        return code.to_string();
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        if let Some(signal) = status.signal() {
            return format!("-{signal}");
        }
    }
    "signal".to_string()
}

/// Every check the gate runs after the version comparison, in the exact order
/// the script ran them. Order is part of the contract: each runs against the
/// same throwaway state root, so a command moved earlier or later sees
/// different state.
///
/// The two serve rejections, which ran last, need a port chosen at run time;
/// their argv carries [`PORT_TOKEN`], which [`execute`] resolves.
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
        // The point of the whole gate: a GPU-required serve must refuse rather
        // than quietly running on CPU, and an explicit CPU serve must be
        // refused too.
        Check::new(
            "vllm reject required gpu",
            Bin::Rocm,
            &[
                "serve",
                "qwen",
                "--engine",
                "vllm",
                "--device",
                "gpu_required",
                "--foreground",
                "--port",
                PORT_TOKEN,
            ],
        )
        .failing()
        .contains("gpu_required")
        .not_contains("CPU fallback"),
        Check::new(
            "rocm vllm reject cpu serve",
            Bin::Rocm,
            &[
                "serve",
                "qwen",
                "--engine",
                "vllm",
                "--device",
                "cpu",
                "--foreground",
                "--port",
                PORT_TOKEN,
            ],
        )
        .failing()
        .contains("CPU mode is not a fallback path"),
    ]
}

/// `cargo build` argv for a smoke run, so the build lands where the binaries are
/// then looked for. The script hard-coded a debug workspace build, which made
/// `--profile release` build one profile and then smoke another.
///
/// The debug build keeps the script's `--workspace --all-targets`: as the local
/// gate it also catches a test target that no longer compiles. The release
/// build is narrowed to the four binaries the gate runs, for two reasons. A
/// release build of every test and bench target is a heavy price for four
/// executables. And `cargo xtask` is itself a release build of this crate, so a
/// release `--workspace` build would relink the very executable that is running
/// it — which Windows refuses.
///
/// `target_dir` is the already-resolved directory, so cargo and the binary
/// lookup cannot disagree about a relative path.
fn build_args(profile: Profile, target_dir: Option<&Path>) -> Vec<String> {
    let mut args = vec!["build".to_string()];
    match profile {
        Profile::Debug => {
            args.push("--workspace".to_string());
            args.push("--all-targets".to_string());
        }
        Profile::Release => {
            args.push("--release".to_string());
            for bin in Bin::ALL {
                args.push("-p".to_string());
                args.push(bin.stem().to_string());
            }
        }
    }
    if let Some(dir) = target_dir {
        args.push("--target-dir".to_string());
        args.push(dir.display().to_string());
    }
    args
}

/// Resolve a `--target-dir` override against the workspace root once, so the
/// build and the binary lookup are handed the same absolute path.
fn resolve_target_dir(root: &Path, dir: &Path) -> PathBuf {
    if dir.is_absolute() {
        dir.to_path_buf()
    } else {
        root.join(dir)
    }
}

/// Directory the built binaries are looked for in: the resolved override, or
/// the active cargo target directory.
fn binary_dir(root: &Path, profile: Profile, target_dir: Option<&Path>) -> PathBuf {
    target_dir
        .map_or_else(|| crate::paths::target_dir(root), Path::to_path_buf)
        .join(profile.dir_name())
}

/// Label printed for the build step, so the log says what was compiled.
const fn build_label(profile: Profile) -> &'static str {
    match profile {
        // The script's label, kept: this is the build it ran.
        Profile::Debug => "build workspace all targets",
        Profile::Release => "build release binaries",
    }
}

/// Everything a smoke run decides before it spawns anything, so those decisions
/// can be checked without building or running a binary.
#[derive(Debug)]
struct Plan {
    root: PathBuf,
    /// Throwaway state root, wiped at the start of every run.
    env_root: PathBuf,
    env: BTreeMap<&'static str, OsString>,
    binaries: Binaries,
    /// `cargo` argv for the build, or `None` with `--skip-build`.
    build: Option<Vec<String>>,
}

impl Plan {
    fn new(
        root: &Path,
        profile: Profile,
        skip_build: bool,
        target_dir_override: Option<&Path>,
    ) -> Self {
        // Resolved once, so the build and the binary lookup get the same path.
        let target_dir = target_dir_override.map(|dir| resolve_target_dir(root, dir));
        // Rooted at the checkout, not the active target dir: `docs/testing.md`
        // tells contributors to share one `CARGO_TARGET_DIR` across checkouts,
        // and this directory is wiped at the start of every run — two worktrees
        // sharing a target dir would delete each other's state mid-run.
        let env_root = root.join("target").join("smoke-local");
        Self {
            root: root.to_path_buf(),
            env: isolated_env(&env_root),
            env_root,
            binaries: Binaries::new(binary_dir(root, profile, target_dir.as_deref())),
            build: (!skip_build).then(|| build_args(profile, target_dir.as_deref())),
        }
    }
}

/// Runs one smoke command: `(label, program, argv, expect_failure)` to its
/// combined output. [`run`] passes [`run_command`]; tests pass a fake, which is
/// what lets [`execute`] — the gate's whole logic — be tested without binaries.
type Runner<'a> = dyn FnMut(&str, &Path, &[&str], bool) -> Result<String> + 'a;

/// The gate: the version comparison, every table check in order, and the
/// first-run assertions, against binaries the plan has located.
fn execute(plan: &Plan, port: &str, run: &mut Runner<'_>) -> Result<()> {
    plan.binaries.require_all_present()?;

    if plan.env_root.exists() {
        std::fs::remove_dir_all(&plan.env_root)
            .with_context(|| format!("failed to clear {}", plan.env_root.display()))?;
    }
    std::fs::create_dir_all(&plan.env_root)
        .with_context(|| format!("failed to create {}", plan.env_root.display()))?;

    let rocm = plan.binaries.path(Bin::Rocm);

    // The three version surfaces must agree on the build string, and `rocm
    // version` reports the active SDK and driver on top of it.
    let version_flag = run("rocm --version", &rocm, &["--version"], false)?;
    let version_short = run("rocm -V", &rocm, &["-V"], false)?;
    if version_flag != version_short {
        bail!(
            "version flag surfaces returned different output: {version_flag:?} vs {version_short:?}"
        );
    }
    let version_command = run("rocm version", &rocm, &["version"], false)?;
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
        let program = plan.binaries.path(check.bin);
        let args = check.resolved_args(port);
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        let output = run(check.label, &program, &argv, check.expect_failure)?;
        check.assert_output(&output)?;
    }

    // Nothing above should have provisioned anything: a first run that quietly
    // created a uv cache or a runtime registry would mean a command took an
    // install path it was never asked to take. The parent directories come from
    // the map the children were given, so renaming one in `isolated_env` cannot
    // leave this looking somewhere the CLI was never pointed.
    let (cache_dir, data_dir) = state_dirs(&plan.env);
    assert_path_missing(&cache_dir.join("uv"), "first-run smoke uv cache")?;
    assert_path_missing(
        &data_dir.join("runtimes").join("registry"),
        "first-run smoke runtime registry",
    )?;
    Ok(())
}

/// The runner [`run`] hands to [`execute`]: every child gets the plan's isolated
/// environment and runs in the workspace root.
///
/// Separate from `run` so a test can prove it. A runner that lost the isolation
/// would not fail the gate — the binaries would read and write the developer's
/// real state, and on a clean machine still print `smoke: ok` — so running the
/// gate cannot catch that; only a test of this function can.
fn child_runner(plan: &Plan) -> impl FnMut(&str, &Path, &[&str], bool) -> Result<String> + '_ {
    move |label, program, argv, expect_failure| {
        run_command(label, program, argv, &plan.env, &plan.root, expect_failure)
    }
}

/// Build, then run every smoke check against the built binaries in an isolated
/// state root.
///
/// Glue between the environment and the tested parts: the gate's logic is
/// [`execute`], the up-front decisions are [`Plan`], and the children's
/// isolation is [`child_runner`]. What is untested is this function's own
/// wiring — the build step, the choice of port, and the hand-off of
/// `child_runner` to `execute` — which runs only when the gate itself is run.
pub fn run(profile: Profile, skip_build: bool, target_dir_override: Option<PathBuf>) -> Result<()> {
    let root = workspace_root()?;
    let plan = Plan::new(&root, profile, skip_build, target_dir_override.as_deref());

    if let Some(args) = &plan.build {
        let cargo = PathBuf::from(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        // The build inherits this process's environment rather than the isolated
        // one. Isolation exists for the binaries under test; under it, a redirected
        // `HOME` hides cargo's registry and config from the build. No build
        // script reads the variables the isolation sets.
        run_command(
            build_label(profile),
            &cargo,
            &argv,
            &BTreeMap::new(),
            &root,
            false,
        )?;
    }

    let port = free_tcp_port()?.to_string();
    execute(&plan, &port, &mut child_runner(&plan))?;
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

    /// The gate's whole content, pinned.
    ///
    /// With `scripts/smoke_local.py` deleted, nothing else records what this
    /// gate asserts: the order and label tests below pass just as happily over
    /// a check whose expectations have been quietly dropped. Dropping
    /// `.failing()` from a rejection check — turning "this must be refused"
    /// into "this must succeed" — is the failure this exists to catch.
    ///
    /// Update it when a check genuinely changes, and say why in the commit.
    #[test]
    fn every_check_pins_its_command_and_its_expectations() {
        let rendered: Vec<String> = checks().iter().map(render_check).collect();
        let expected = vec![
            r#"rocm examine | rocm ["examine"] | =contains["rocm examine", "default_engine:", "managed_runtimes: 0", "managed_services: 0"]"#,
            r#"rocm engines list | rocm ["engines", "list"] | =contains["lemonade", "vllm"]"#,
            r#"rocm config set telemetry off | rocm ["config", "set-telemetry", "off"] | =contains["telemetry mode set to off", "policy: disabled"]"#,
            r#"rocm config show | rocm ["config", "show"] | =contains["telemetry_mode: off", "telemetry_policy: disabled"]"#,
            r#"rocm engines install requires exact runtime | rocm ["engines", "install", "vllm"] | !contains["no active ROCm runtime is configured"]"#,
            r#"rocm chat local status | rocm ["chat", "--provider", "local"] | =any["Provider: local", "Assistant source: local model on this computer"]"#,
            r#"rocm freeform installed status question | rocm ["is rocm installed?"] | =contains["ROCm status", "Nothing was changed."] absent["No ROCm action selected"]"#,
            r#"rocm freeform comfyui help question | rocm ["how do i setup comfyui"] | =contains["ComfyUI status", "Nothing was changed."] absent["No ROCm action selected"]"#,
            r#"rocm freeform comfyui install request | rocm ["can you setup comfyui for me"] | =contains["Install ComfyUI", "approval: required"]"#,
            r#"rocm freeform vllm plan | rocm ["serve qwen with vllm"] | =contains["engine: vllm", "no CPU fallback is implied"]"#,
            r#"rocm freeform tiny gpu recipe plan | rocm ["serve tiny-gpt2"] | =contains["model: sshleifer/tiny-gpt2", "device_policy: gpu_required", "--device gpu_required", "approval: required"]"#,
            #[cfg(windows)]
            r#"windows tarball sdk rejection | rocm ["install", "sdk", "--format", "tarball", "--dry-run"] | !contains["TheRock tarball installs are not supported on Windows"]"#,
            r#"rocmd status | rocmd ["status"] | =contains["rocmd status"]"#,
            r#"rocmd bridge snapshot | rocmd ["bridge-snapshot"] | =json(Protocol("rocmd-codex-bridge-v0"))"#,
            r#"rocmd sandbox examine snapshot | rocmd ["sandbox-run", "examine_snapshot", "--allow-native-fallback"] | =json(SandboxTool { tool: "examine_snapshot", noun: "examine" })"#,
            r#"rocmd sandbox list servers | rocmd ["sandbox-run", "list_servers", "--allow-native-fallback"] | =json(SandboxTool { tool: "list_servers", noun: "list" })"#,
            r#"rocmd sandbox prefetch validation | rocmd ["sandbox-run", "prefetch_artifact", "--allow-native-fallback"] | !contains["prefetch_artifact requires"]"#,
            r#"vllm detect | vllm ["detect"] | =json(Parses)"#,
            r#"vllm capabilities | vllm ["capabilities"] | =json(VllmCapabilities)"#,
            r#"vllm resolve qwen | vllm ["resolve-model", "qwen"] | =contains["qwen"]"#,
            r#"vllm reject cpu | vllm ["resolve-model", "qwen", "--device-policy", "cpu_only"] | !contains["no CPU fallback is used"]"#,
            r#"vllm reject required gpu | rocm ["serve", "qwen", "--engine", "vllm", "--device", "gpu_required", "--foreground", "--port", "{port}"] | !contains["gpu_required"] absent["CPU fallback"]"#,
            r#"rocm vllm reject cpu serve | rocm ["serve", "qwen", "--engine", "vllm", "--device", "cpu", "--foreground", "--port", "{port}"] | !contains["CPU mode is not a fallback path"]"#,
        ];
        assert_eq!(rendered, expected);
    }

    /// `=` expects success, `!` expects a non-zero exit.
    fn render_check(check: &Check) -> String {
        // Debug formatting quotes every element, so the rendering shows argv
        // boundaries: `["serve qwen with vllm"]` is one argument, and splitting
        // it into four would make `rocm` parse a real `serve` subcommand. The
        // JSON expectation is rendered whole for the same reason — its payload
        // is what the check asserts, not its variant name.
        let mut parts = vec![
            check.label.to_string(),
            format!("{} {:?}", check.bin.key(), check.args),
        ];
        let mut expectations = vec![if check.expect_failure {
            "!".to_string()
        } else {
            "=".to_string()
        }];
        if !check.contains.is_empty() {
            expectations.push(format!("contains{:?}", check.contains));
        }
        if !check.contains_any.is_empty() {
            expectations.push(format!("any{:?}", check.contains_any));
        }
        if !check.not_contains.is_empty() {
            expectations.push(format!("absent{:?}", check.not_contains));
        }
        if let Some(json) = check.json {
            expectations.push(format!("json({json:?})"));
        }
        let head = expectations.remove(0);
        parts.push(format!("{head}{}", expectations.join(" ")));
        parts.join(" | ")
    }

    /// An empty needle is satisfied by every string, so a check carrying one
    /// asserts nothing while looking like it does.
    #[test]
    fn no_check_carries_an_empty_needle() {
        for check in &checks() {
            for needle in check
                .contains
                .iter()
                .chain(check.contains_any.iter())
                .chain(check.not_contains.iter())
            {
                assert!(
                    !needle.trim().is_empty(),
                    "{} carries an empty needle",
                    check.label
                );
            }
        }
    }

    /// The port token is substituted, not passed through to the CLI as a
    /// literal — and only where it appears.
    #[test]
    fn the_port_token_is_resolved_in_argv() {
        let check = Check::new(
            "demo",
            Bin::Rocm,
            &["serve", "--port", PORT_TOKEN, "--device", "cpu"],
        );
        assert_eq!(
            check.resolved_args("54321"),
            vec!["serve", "--port", "54321", "--device", "cpu"]
        );
        // The token is substituted only as a whole argument, so one embedded in
        // a larger argument (`--port={port}`) would reach the CLI verbatim. Catch
        // that by substring, which the whole-argument substitution cannot hide.
        for check in &checks() {
            assert!(
                !check
                    .resolved_args("1")
                    .iter()
                    .any(|arg| arg.contains(PORT_TOKEN)),
                "{} passes the port token through unresolved",
                check.label
            );
        }
        let embedded = Check::new("demo", Bin::Rocm, &["--port={port}"]);
        assert!(
            embedded.resolved_args("1")[0].contains(PORT_TOKEN),
            "the substring check above must be able to see an embedded token"
        );
    }

    /// Order is part of the contract, not an accident of how the table was
    /// written: every check shares one throwaway state root, so a command moved
    /// earlier or later sees different state. This pins the sequence the
    /// `scripts/smoke_local.py` gate ran after its version comparison, minus the
    /// platform-gated entry, which the next test places.
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
                "vllm reject required gpu",
                "rocm vllm reject cpu serve",
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
        // Every other entry is a directory, and every one must sit under the
        // throwaway root. Asserting only that the keys are present would let any
        // one of them be pointed at the developer's real profile unnoticed.
        for (key, value) in &env {
            if *key == "ROCM_CLI_UPDATE_USER_PATH" {
                continue;
            }
            let value = PathBuf::from(value);
            assert!(
                value.starts_with(root),
                "{key} escaped the smoke root: {value:?}"
            );
        }
        for key in [
            "ROCM_CLI_CONFIG_DIR",
            "ROCM_CLI_DATA_DIR",
            "ROCM_CLI_CACHE_DIR",
        ] {
            assert!(env.contains_key(key), "{key} is not redirected");
        }
    }

    #[cfg(windows)]
    #[test]
    fn isolated_env_redirects_the_windows_profile_dirs() {
        let root = Path::new("C:\\smoke");
        let env = isolated_env(root);
        for key in ["APPDATA", "LOCALAPPDATA"] {
            let value = PathBuf::from(env.get(key).unwrap_or_else(|| panic!("{key} is not set")));
            assert!(
                value.starts_with(root),
                "{key} points outside the smoke root: {value:?}"
            );
        }
        assert!(!env.contains_key("HOME"));
    }

    // Unix-only: the XDG/HOME arm is compiled out on Windows, where the branch
    // above applies instead. Supported hosts are Windows and Linux only.
    #[cfg(unix)]
    #[test]
    fn isolated_env_redirects_the_xdg_dirs_and_home() {
        let root = Path::new("/tmp/smoke-local");
        let env = isolated_env(root);
        for key in ["XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "HOME"] {
            let value = PathBuf::from(env.get(key).unwrap_or_else(|| panic!("{key} is not set")));
            assert!(
                value.starts_with(root),
                "{key} points outside the smoke root: {value:?}"
            );
        }
        assert!(!env.contains_key("APPDATA"));
    }

    /// A child killed by a signal has no exit code. The script printed Python's
    /// negated signal number, which keeps an OOM kill (`-9`) distinguishable
    /// from a segfault (`-11`); a bare "signal" would not.
    #[cfg(unix)]
    #[test]
    fn a_signalled_child_keeps_its_signal_number() {
        use std::os::unix::process::ExitStatusExt as _;

        assert_eq!(exit_code_text(&std::process::ExitStatus::from_raw(9)), "-9");
        assert_eq!(
            exit_code_text(&std::process::ExitStatus::from_raw(11)),
            "-11"
        );
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

    /// Run a shell script through `run_command`, so it is driven for real rather
    /// than reasoned about. Each test supplies the script for both shells: `cmd`
    /// is not POSIX — `;` is not a separator there and there is no `cat` — so a
    /// shared script would test something different on each platform.
    fn run_script(
        label: &str,
        unix: &str,
        windows: &str,
        env: &BTreeMap<&'static str, OsString>,
        cwd: &Path,
        expect_failure: bool,
    ) -> Result<String> {
        let (program, flag, script) = if cfg!(windows) {
            ("cmd", "/C", windows)
        } else {
            ("sh", "-c", unix)
        };
        run_command(
            label,
            Path::new(program),
            &[flag, script],
            env,
            cwd,
            expect_failure,
        )
    }

    fn run_shell(label: &str, unix: &str, windows: &str, expect_failure: bool) -> Result<String> {
        let dir = tempfile::tempdir().expect("tempdir");
        run_script(
            label,
            unix,
            windows,
            &BTreeMap::new(),
            dir.path(),
            expect_failure,
        )
    }

    /// Lines with `cmd`'s `\r` and trailing spaces removed, so both shells'
    /// output compares the same way.
    fn lines_of(output: &str) -> Vec<String> {
        output.lines().map(|line| line.trim().to_string()).collect()
    }

    #[test]
    fn run_command_returns_the_trimmed_output_of_a_successful_child() {
        let output =
            run_shell("demo", "echo hello", "echo hello", false).expect("the child succeeds");
        assert_eq!(output, "hello");
    }

    /// Both streams reach the assertions, stdout first. The script merged them
    /// into one pipe; this concatenates them, so a check asserting on stderr must
    /// still see it. Exact lines, not substrings: under `cmd` a mis-quoted script
    /// echoes its own text, which would satisfy a substring test by accident.
    #[test]
    fn run_command_captures_stderr_after_stdout() {
        let output = run_shell(
            "demo",
            "echo to-stdout; echo to-stderr 1>&2",
            "echo to-stdout& 1>&2 echo to-stderr",
            false,
        )
        .expect("the child succeeds");
        assert_eq!(lines_of(&output), vec!["to-stdout", "to-stderr"]);
    }

    /// The failure names the check and the status, so a red gate says which
    /// command broke without the reader going to the transcript.
    #[test]
    fn run_command_fails_when_an_expected_success_exits_non_zero() {
        let error = run_shell("demo", "exit 7", "exit 7", false).expect_err("the child fails");
        let message = format!("{error}");
        assert!(message.contains("demo exited with status 7"), "{message}");
    }

    #[test]
    fn run_command_accepts_a_failure_it_was_told_to_expect() {
        let output = run_shell(
            "demo",
            "echo refused 1>&2; exit 1",
            "1>&2 echo refused& exit 1",
            true,
        )
        .expect("a failure was expected");
        assert_eq!(lines_of(&output), vec!["refused"]);
    }

    /// The inverse, and the one that matters: a rejection check whose command
    /// starts succeeding must fail the gate rather than pass quietly.
    #[test]
    fn run_command_fails_when_an_expected_failure_succeeds() {
        let error =
            run_shell("demo", "exit 0", "exit 0", true).expect_err("success is the failure here");
        assert!(
            format!("{error}").contains("demo unexpectedly succeeded"),
            "{error}"
        );
    }

    /// Children are handed a null stdin, so a command that reads to EOF returns
    /// instead of hanging the gate — the contract `rocm chat` relies on.
    ///
    /// This only has teeth where the test process's own stdin is an open pipe or
    /// a terminal, as under a local `cargo test`. Where the harness already gives
    /// tests a null stdin, an inherited one would also read EOF and this passes
    /// either way; nothing in `std::process::Command` exposes the configured
    /// stdin for a direct assertion.
    #[test]
    fn run_command_hands_the_child_a_closed_stdin() {
        let output = run_shell("demo", "cat; echo done", "sort& echo done", false)
            .expect("the reader sees EOF at once");
        assert_eq!(lines_of(&output), vec!["done"]);
    }

    /// The isolated environment and the working directory both reach the child.
    /// The directory is proved by a file only it contains, not by comparing
    /// paths: a Windows temp directory can come back in 8.3 short form.
    #[test]
    fn run_command_passes_the_env_and_working_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("marker-file"), b"").expect("marker");
        let mut env = BTreeMap::new();
        env.insert("SMOKE_PROBE", OsString::from("from-env"));

        let output = run_script(
            "demo",
            r#"echo "$SMOKE_PROBE" && ls marker-file"#,
            "echo %SMOKE_PROBE%& dir /b marker-file",
            &env,
            dir.path(),
            false,
        )
        .expect("the probe and the marker are both visible");
        assert_eq!(lines_of(&output), vec!["from-env", "marker-file"]);
    }

    /// The JSON branch of `assert_output` is reached through a real check, not
    /// only by calling `JsonCheck::assert_value` directly.
    #[test]
    fn assert_output_runs_the_json_check_it_carries() {
        let check = Check::new("demo", Bin::Rocmd, &["bridge-snapshot"])
            .json(JsonCheck::Protocol("rocmd-codex-bridge-v0"));

        check
            .assert_output(r#"{"protocol":"rocmd-codex-bridge-v0"}"#)
            .expect("the protocol matches");

        let wrong = check
            .assert_output(r#"{"protocol":"other"}"#)
            .expect_err("the protocol does not match");
        assert!(
            format!("{wrong}").contains("unexpected bridge snapshot protocol"),
            "{wrong}"
        );

        let unparsable = check
            .assert_output("not json")
            .expect_err("not JSON at all");
        assert!(
            format!("{unparsable}").contains("demo did not return valid JSON"),
            "{unparsable}"
        );
    }

    /// `--profile release` must build the profile it then smokes, and a
    /// `--target-dir` must reach the build too: otherwise the flags build one
    /// place and look in another.
    #[test]
    fn the_build_lands_where_the_binaries_are_looked_for() {
        assert_eq!(
            build_args(Profile::Debug, None),
            vec!["build", "--workspace", "--all-targets"],
            "the debug build keeps the script's whole-workspace build"
        );
        assert_eq!(
            build_args(Profile::Release, None),
            vec![
                "build",
                "--release",
                "-p",
                "rocm",
                "-p",
                "rocmd",
                "-p",
                "rocm-engine-lemonade",
                "-p",
                "rocm-engine-vllm",
            ],
            "the release build names only the binaries the gate runs"
        );
        assert!(
            !build_args(Profile::Release, None).contains(&"--workspace".to_string()),
            "a release --workspace build would relink the running `cargo xtask`"
        );

        // An absolute path from a real directory, so it is absolute on Windows
        // too — `/tmp/...` has no drive and is relative there.
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("elsewhere");
        let args = build_args(Profile::Release, Some(&out));
        let target_flag = args
            .iter()
            .position(|arg| arg == "--target-dir")
            .expect("the build is told where to put the binaries");
        assert_eq!(args[target_flag + 1], out.display().to_string());

        let root = dir.path().join("repo");
        assert_eq!(
            binary_dir(&root, Profile::Release, Some(&out)),
            out.join("release")
        );
    }

    /// A relative `--target-dir` is resolved once, against the workspace root,
    /// so the build and the lookup are given the same path whatever directory
    /// cargo happens to run in.
    #[test]
    fn a_relative_target_dir_resolves_against_the_workspace_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("repo");
        let absolute = dir.path().join("abs");

        assert_eq!(
            resolve_target_dir(&root, Path::new("rel")),
            root.join("rel")
        );
        assert_eq!(
            resolve_target_dir(&root, &absolute),
            absolute,
            "an absolute path is used as given"
        );

        let resolved = Some(resolve_target_dir(&root, Path::new("rel")));
        let args = build_args(Profile::Debug, resolved.as_deref());
        assert_eq!(
            args.last().map(String::as_str),
            Some(root.join("rel").display().to_string().as_str()),
            "the build receives the resolved path, not the relative one"
        );
        assert_eq!(
            binary_dir(&root, Profile::Debug, resolved.as_deref()),
            root.join("rel").join("debug")
        );
    }

    /// The first-run assertions look where the CLI was told to write.
    #[test]
    fn the_first_run_checks_look_in_the_isolated_state_dirs() {
        let root = Path::new("/tmp/smoke-local");
        let env = isolated_env(root);
        let (cache_dir, data_dir) = state_dirs(&env);
        assert_eq!(cache_dir, PathBuf::from(&env["ROCM_CLI_CACHE_DIR"]));
        assert_eq!(data_dir, PathBuf::from(&env["ROCM_CLI_DATA_DIR"]));
        assert!(cache_dir.starts_with(root) && data_dir.starts_with(root));
    }

    const FAKE_VERSION: &str = "rocm-cli 0.1.0 (main, a1b2c3d)";
    const FAKE_PORT: &str = "45678";

    /// Output that satisfies a check's own expectations, built from the check —
    /// so a test can make one check fail by overriding only that one.
    fn satisfying_output(check: &Check) -> String {
        if let Some(json) = check.json {
            return match json {
                JsonCheck::Parses => "{}".to_string(),
                JsonCheck::Protocol(protocol) => format!(r#"{{"protocol":"{protocol}"}}"#),
                JsonCheck::SandboxTool { tool, .. } => {
                    format!(r#"{{"tool":"{tool}","ok":true}}"#)
                }
                JsonCheck::VllmCapabilities => {
                    r#"{"openai_compatible":true,"cpu":false}"#.to_string()
                }
            };
        }
        let mut lines: Vec<&str> = check.contains.clone();
        lines.extend(check.contains_any.first());
        lines.join("\n")
    }

    /// One recorded call: label, program file name, argv, `expect_failure`.
    type Call = (String, String, Vec<String>, bool);

    /// A plan over a throwaway tree, with the four binaries present as empty
    /// files so `execute` can locate them.
    fn fake_plan(dir: &Path) -> Plan {
        let target = dir.join("target-dir");
        let bin_dir = target.join("debug");
        std::fs::create_dir_all(&bin_dir).expect("bin dir");
        for bin in Bin::ALL {
            std::fs::write(bin_dir.join(binary_name(bin.stem())), b"").expect("stub binary");
        }
        Plan::new(&dir.join("repo"), Profile::Debug, true, Some(&target))
    }

    /// Drive `execute` with a fake runner that answers every command with
    /// passing output, except where `overrides` says otherwise. `on_call` runs
    /// before each answer, so a test can simulate a command's side effects.
    fn execute_with(
        plan: &Plan,
        overrides: &BTreeMap<&str, String>,
        on_call: &mut dyn FnMut(&str),
    ) -> (Result<()>, Vec<Call>) {
        let table = checks();
        let mut calls = Vec::new();
        let result = execute(
            plan,
            FAKE_PORT,
            &mut |label, program, argv, expect_failure| {
                on_call(label);
                calls.push((
                    label.to_string(),
                    program
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    argv.iter().map(|arg| (*arg).to_string()).collect(),
                    expect_failure,
                ));
                if let Some(output) = overrides.get(label) {
                    return Ok(output.clone());
                }
                Ok(match label {
                    "rocm --version" | "rocm -V" => FAKE_VERSION.to_string(),
                    "rocm version" => format!("{FAKE_VERSION}\nROCm SDK: none\nGPU driver: none"),
                    _ => satisfying_output(
                        table
                            .iter()
                            .find(|check| check.label == label)
                            .unwrap_or_else(|| panic!("unexpected command {label}")),
                    ),
                })
            },
        );
        (result, calls)
    }

    /// The gate runs the version surfaces, then every table check in table
    /// order, each against its own binary, with the port resolved and its
    /// expected exit status passed through.
    #[test]
    fn execute_runs_every_check_in_order_against_the_right_binary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let plan = fake_plan(dir.path());
        let (result, calls) = execute_with(&plan, &BTreeMap::new(), &mut |_| {});
        result.expect("every check is satisfied");

        let table = checks();
        let mut expected: Vec<Call> = vec![
            (
                "rocm --version".into(),
                binary_name("rocm"),
                vec!["--version".into()],
                false,
            ),
            (
                "rocm -V".into(),
                binary_name("rocm"),
                vec!["-V".into()],
                false,
            ),
            (
                "rocm version".into(),
                binary_name("rocm"),
                vec!["version".into()],
                false,
            ),
        ];
        expected.extend(table.iter().map(|check| {
            (
                check.label.to_string(),
                binary_name(check.bin.stem()),
                check.resolved_args(FAKE_PORT),
                check.expect_failure,
            )
        }));
        assert_eq!(calls, expected);
        assert!(
            calls
                .iter()
                .any(|(_, _, argv, _)| argv.iter().any(|arg| arg == FAKE_PORT)),
            "the serve rejections receive the chosen port"
        );
    }

    /// The failure that matters most: a check whose output does not meet its
    /// expectations must fail the gate. Without this, a gate that stopped
    /// asserting would still print `smoke: ok`.
    #[test]
    fn execute_fails_a_check_whose_output_misses_its_expectation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let plan = fake_plan(dir.path());
        let overrides = BTreeMap::from([("rocm config show", "telemetry_mode: on".to_string())]);
        let (result, _) = execute_with(&plan, &overrides, &mut |_| {});
        let error = result.expect_err("the expectation is unmet");
        assert!(
            format!("{error}").contains("rocm config show did not contain expected text"),
            "{error}"
        );
    }

    #[test]
    fn execute_fails_when_the_version_surfaces_disagree() {
        let dir = tempfile::tempdir().expect("tempdir");
        let plan = fake_plan(dir.path());

        let short = BTreeMap::from([("rocm -V", "rocm-cli 0.1.0 (main, ffffff)".to_string())]);
        let (result, _) = execute_with(&plan, &short, &mut |_| {});
        assert!(format!("{}", result.expect_err("-V disagrees")).contains("version flag surfaces"),);

        let long = BTreeMap::from([(
            "rocm version",
            "rocm-cli 0.1.0 (main, ffffff)\nROCm SDK: none\nGPU driver: none".to_string(),
        )]);
        let (result, _) = execute_with(&plan, &long, &mut |_| {});
        assert!(
            format!("{}", result.expect_err("the build line disagrees"))
                .contains("build line does not match"),
        );
    }

    /// `rocm version` must carry a traceable build string and report the SDK
    /// and driver. Each override keeps the three version surfaces equal, so the
    /// earlier equality checks pass and only the assertion under test can fail.
    #[test]
    fn execute_checks_what_rocm_version_reports() {
        let unknown = "rocm-cli 0.1.0 (unknown, a1b2c3d)";
        let cases = [
            (
                unknown,
                format!("{unknown}\nROCm SDK: none\nGPU driver: none"),
                "unresolved ref",
            ),
            (
                FAKE_VERSION,
                format!("{FAKE_VERSION}\nGPU driver: none"),
                "did not contain expected text: ROCm SDK:",
            ),
            (
                FAKE_VERSION,
                format!("{FAKE_VERSION}\nROCm SDK: none"),
                "did not contain expected text: GPU driver:",
            ),
        ];
        for (flag, version, expected) in cases {
            let dir = tempfile::tempdir().expect("tempdir");
            let plan = fake_plan(dir.path());
            let overrides = BTreeMap::from([
                ("rocm --version", flag.to_string()),
                ("rocm -V", flag.to_string()),
                ("rocm version", version),
            ]);
            let (result, _) = execute_with(&plan, &overrides, &mut |_| {});
            let error = result.expect_err("the version report is incomplete");
            assert!(format!("{error}").contains(expected), "{error}");
        }
    }

    /// A command that provisions a uv cache or a runtime registry fails the
    /// gate. The paths are written here independently of the production code,
    /// so a typo in either subpath there leaves the check looking somewhere
    /// nothing is created, and this test catches it.
    #[test]
    fn execute_fails_when_a_command_provisions_state() {
        for (subpath, expected) in [
            (&["uv"][..], "first-run smoke uv cache"),
            (
                &["runtimes", "registry"][..],
                "first-run smoke runtime registry",
            ),
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let plan = fake_plan(dir.path());
            let parent = if subpath[0] == "uv" {
                PathBuf::from(&plan.env["ROCM_CLI_CACHE_DIR"])
            } else {
                PathBuf::from(&plan.env["ROCM_CLI_DATA_DIR"])
            };
            let provisioned = subpath.iter().fold(parent, |path, part| path.join(part));
            let (result, _) = execute_with(&plan, &BTreeMap::new(), &mut |label| {
                if label == "rocm examine" {
                    std::fs::create_dir_all(&provisioned).expect("simulate provisioning");
                }
            });
            let error = result.expect_err("provisioned state is a failure");
            assert!(format!("{error}").contains(expected), "{error}");
        }
    }

    /// The state root is wiped before any command runs, so leftovers from a
    /// previous run — including a uv cache — neither leak in nor fail the gate.
    #[test]
    fn execute_starts_from_an_empty_state_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        let plan = fake_plan(dir.path());
        let stale_cache = PathBuf::from(&plan.env["ROCM_CLI_CACHE_DIR"]).join("uv");
        std::fs::create_dir_all(&stale_cache).expect("stale cache");
        let stale_file = plan.env_root.join("stale");
        std::fs::write(&stale_file, b"").expect("stale file");

        let mut seen_stale = false;
        let (result, _) = execute_with(&plan, &BTreeMap::new(), &mut |_| {
            seen_stale |= stale_file.exists();
        });
        result.expect("a previous run's leftovers are cleared, not reported");
        assert!(!seen_stale, "a command ran before the state root was wiped");
    }

    #[test]
    fn execute_refuses_a_partial_build() {
        let dir = tempfile::tempdir().expect("tempdir");
        let plan = fake_plan(dir.path());
        std::fs::remove_file(plan.binaries.path(Bin::Vllm)).expect("remove one binary");
        let (result, calls) = execute_with(&plan, &BTreeMap::new(), &mut |_| {});
        assert!(
            format!("{}", result.expect_err("a binary is missing"))
                .contains("missing smoke binary vllm"),
        );
        assert!(calls.is_empty(), "nothing runs against a partial build");
    }

    /// Every child runs with the plan's isolated environment and in the
    /// workspace root. This is what keeps the gate off the developer's real
    /// state, and running the gate cannot detect its loss — so it is proved here,
    /// through a real shell. Values are compared as the strings that were set,
    /// not as resolved paths, which Windows can return in 8.3 short form.
    #[test]
    fn child_runner_isolates_every_child_and_runs_it_in_the_workspace_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        let plan = fake_plan(dir.path());
        std::fs::create_dir_all(&plan.root).expect("workspace root");
        std::fs::write(plan.root.join("marker-file"), b"").expect("marker");

        let (program, flag, script, profile_var) = if cfg!(windows) {
            (
                "cmd",
                "/C",
                "echo %LOCALAPPDATA%& echo %ROCM_CLI_DATA_DIR%& dir /b marker-file",
                "LOCALAPPDATA",
            )
        } else {
            (
                "sh",
                "-c",
                r#"echo "$HOME"; echo "$ROCM_CLI_DATA_DIR"; ls marker-file"#,
                "HOME",
            )
        };
        let output = child_runner(&plan)("probe", Path::new(program), &[flag, script], false)
            .expect("the probe runs in the workspace root");

        let expected = vec![
            plan.env[profile_var].to_string_lossy().into_owned(),
            plan.env["ROCM_CLI_DATA_DIR"].to_string_lossy().into_owned(),
            "marker-file".to_string(),
        ];
        assert_eq!(lines_of(&output), expected);
        assert!(
            PathBuf::from(&expected[0]).starts_with(&plan.env_root),
            "the profile directory the child saw is the throwaway one"
        );
    }

    /// The plan builds the profile it then smokes, in the directory it then
    /// looks in, and skips the build only when asked.
    #[test]
    fn the_plan_builds_and_looks_in_the_same_place() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("repo");

        let release = Plan::new(&root, Profile::Release, false, Some(Path::new("rel")));
        assert_eq!(
            release.build,
            Some(build_args(Profile::Release, Some(&root.join("rel"))))
        );
        assert!(
            release
                .build
                .as_ref()
                .is_some_and(|args| args.contains(&"--release".to_string()))
        );
        assert_eq!(release.binaries.dir, root.join("rel").join("release"));

        let skipped = Plan::new(&root, Profile::Release, true, None);
        assert_eq!(skipped.build, None);

        assert_eq!(release.env_root, root.join("target").join("smoke-local"));
        assert!(PathBuf::from(&release.env["ROCM_CLI_DATA_DIR"]).starts_with(&release.env_root));
    }

    #[test]
    fn the_build_label_names_what_is_built() {
        assert_eq!(build_label(Profile::Debug), "build workspace all targets");
        assert_eq!(build_label(Profile::Release), "build release binaries");
    }

    #[test]
    fn a_free_port_is_outside_the_reserved_range() {
        let port = free_tcp_port().expect("an ephemeral port");
        assert!(port > 1023, "{port} is in the reserved range");
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
