// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Regenerate (or verify) the recorded `rocm` output the Doctor skill evals replay.
//!
//! The skill's behavioural evals run on grading machines that have no `rocm`
//! binary, so an agent following the skill stops at the install offer and never
//! reads a diagnosis. The eval hook (`skills/rocm-doctor/evals/hooks.py`) puts a
//! stub `rocm` on `PATH` that replays what this module writes, so the skill can
//! be graded on how it reads a real answer.
//!
//! A replay is only worth grading against while it still says what the CLI says.
//! What a current CLI would say about the catalog is therefore rendered from the
//! compiled catalog, not typed in: the manifest comes from
//! [`rocm_core::catalog_manifest_json`] and a diagnosis from
//! [`rocm_core::run_diagnose`], run over a recorded examination. `--check` fails
//! when the committed copy differs, so a catalog change cannot leave a stale
//! recording behind, the same guard `cargo xtask catalog --check` gives the manifest.
//!
//! A case may also name a `host`, which makes the stub answer `rocm examine --json`.
//! The reply is the recorded examination with the few fields the host implies
//! edited, and its `status` recomputed by [`rocm_core::Examination::compute_status`],
//! the rule the probe itself uses. The real command prints those fields plus a
//! `summary` of the host that only the `rocm` binary can build, so the replay
//! leaves `summary` out; the skill reads `status` and `has_amd_gpu` and nothing in it.
//!
//! A case may also name an `apply` fix id, which makes the stub answer `rocm fix <id>`
//! with `--dry-run` and with no flag. The fix must be `auto` on the host's platform
//! (`wsl` for the WSL host, else `linux`), and the case's symptom and host must make
//! the replayed diagnosis match it, or the generator fails (a fix the agent could not
//! be pointed at has nothing to replay). The plan is the CLI's own
//! ([`rocm_core::fix_plan_text`]), so it carries the command's wording; only the
//! `Will run:` line the runner adds is not there, because it names the user and the
//! machine. The layout around the plan (the blank line, then the notice) is written
//! here by hand, copying the runner, and is checked only against the committed
//! replay. The refusal's exit code, its message and the dry-run notice are the CLI's
//! own constants ([`rocm_core::fix_exit`], [`rocm_core::NON_INTERACTIVE_REFUSAL`],
//! [`rocm_core::DRY_RUN_NOTICE`]).
//!
//! Some replies are not rendered from the catalog. What an OLDER CLI printed was
//! recorded once from a real older binary and is committed under `recorded/`. The
//! current CLI's `--version` line is fixed text ([`CURRENT_VERSION`]) that carries a
//! marker only the replay prints, so an eval can tell that the agent met the replay;
//! nothing in the skill reads it. The failure message of `examine` is fixed text too.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::paths;

/// Where the fixtures live, relative to the workspace root.
const FIXTURES_DIR: &str = "skills/rocm-doctor/evals/fixtures";
/// The case list: which eval cases replay, against which kind of CLI.
const CASES_FILE: &str = "cases.json";
/// The file written under each case's directory.
const REPLIES_FILE: &str = "replies.json";
/// Hand-recorded output from a real older binary. Never regenerated.
const RECORDED_DIR: &str = "recorded";
const OLDER_FIX_JSON_STDERR: &str = "older-cli-fix-json.stderr";
const OLDER_VERSION: &str = "older-cli-version.txt";
/// The examination every replayed diagnosis runs over. A recorded Linux host
/// with no GPU, so the replay names no real machine.
const EXAMINATION: &str = "tests/remote-ssh/examination.json";
/// The eval dataset the cases belong to, relative to the workspace root.
const EVALS_FILE: &str = "skills/rocm-doctor/evals/evals.json";
/// What the current-CLI replay prints for `--version`. Fixed text, so a rebuild
/// never makes a recording differ. The parenthesised marker is what the
/// `rocm-current-cli-carries-on` eval pins in `logs_contain`; no real build
/// prints that marker, and the skill does not read the version string.
const CURRENT_VERSION: &str = "rocm-cli 0.1.0 (recorded for evals)\n";
/// An older CLI rejects `--json` with a usage error, and clap's usage-error code is 2.
const USAGE_ERROR_RC: i32 = 2;
/// What the replay prints when `host` says `rocm examine` itself fails. Fixed
/// text, not recorded from a real failure: the skill reacts to the non-zero exit
/// and relays whatever was printed, and the marker is what the eval pins in
/// `logs_contain`, since only the replay prints it.
const EXAMINE_FAILS_STDERR: &str = "error: failed to read the host (recorded for evals)\n";
/// A genuine inability to examine propagates as an error, which exits 1.
const EXAMINE_FAILS_RC: i32 = 1;
/// The platform family a replayed host is, which is what `rocm fix` reports for it:
/// the plan's class and argument are per platform. The WSL host is `wsl` there, not
/// `linux`, so a fix that applies on Linux only is not applicable on it.
fn replayed_os(host: Option<Host>) -> &'static str {
    if host == Some(Host::WslHostGpuHidden) {
        "wsl"
    } else {
        "linux"
    }
}

/// The part of the dataset this module reads: which prompt each case id has.
#[derive(Debug, Deserialize)]
struct Evals {
    evaluations: Vec<Eval>,
}

#[derive(Debug, Deserialize)]
struct Eval {
    id: String,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct Cases {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    /// The eval case id the replay belongs to; also the directory name.
    id: String,
    /// Which CLI the stub imitates.
    cli: Cli,
    /// The symptom the diagnosis replay was run with.
    symptom: String,
    /// The machine `rocm examine --json` describes, for a case that needs it. A
    /// case without one gets no examine reply, and the stub refuses the command.
    #[serde(default)]
    host: Option<Host>,
    /// A fix id the case's agent is expected to preview and apply. The stub then
    /// answers `rocm fix <id>` with `--dry-run` and with no flag, and never with
    /// `--yes`, so an agent that applies before the user said OK meets a refusal.
    #[serde(default)]
    apply: Option<String>,
}

/// The host a replayed `rocm examine --json` describes. Each is the recorded
/// examination with only the fields the host implies changed.
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum Host {
    /// Linux with no AMD GPU found: status `no-amd-gpu`.
    NoAmdGpu,
    /// WSL2 where the Windows host has no AMD GPU: status `wsl`, `has_amd_gpu`
    /// false. On WSL the status never says `no-amd-gpu`, so the skill reads the field.
    WslHostGpuHidden,
    /// Linux with an AMD GPU found: status `ok`.
    AmdFound,
    /// `rocm examine` itself fails and exits non-zero.
    ExamineFails,
    /// Linux with an AMD GPU found, inside a Docker container whose
    /// `/dev/shm` is the 64 MiB default. Two entries then score from host
    /// state alone, whatever the symptom is.
    ContainerSmallShm,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Cli {
    /// A CLI that publishes the catalog manifest.
    Current,
    /// A CLI that predates `rocm fix --json`.
    Older,
}

/// One command's reply. The field order is the file's order.
#[derive(Debug, Serialize, PartialEq, Eq)]
struct Reply {
    stdout: String,
    stderr: String,
    rc: i32,
}

impl Reply {
    const fn ok(stdout: String) -> Self {
        Self {
            stdout,
            stderr: String::new(),
            rc: 0,
        }
    }
}

/// Every command the stub answers.
#[derive(Debug, Serialize, PartialEq, Eq)]
struct Replies {
    version: Reply,
    fix_json: Reply,
    diagnose_json: Reply,
    diagnose_text: Reply,
    /// Only for a case that names a `host`; left out of the file otherwise, so a
    /// case without one is recorded exactly as it was before this existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    examine_json: Option<Reply>,
    /// Only for a case that names an `apply` fix id, for the same reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    apply: Option<Apply>,
}

/// What the stub answers for `rocm fix <fix_id>`: the preview, and the refusal a
/// shell with no terminal gets when `--yes` is missing.
#[derive(Debug, Serialize, PartialEq, Eq)]
struct Apply {
    fix_id: String,
    dry_run: Reply,
    bare: Reply,
}

/// Regenerate (or, with `check`, verify) the committed fixtures.
///
/// # Errors
/// When the workspace root cannot be found, an input cannot be read, a file cannot
/// be written, or `check` is set and a fixture is missing, stale, or unexpected.
pub fn run(check: bool) -> Result<()> {
    let root = paths::workspace_root()?;
    if check {
        verify(&root)
    } else {
        generate(&root)
    }
}

fn fixtures_dir(root: &Path) -> PathBuf {
    root.join(FIXTURES_DIR)
}

fn load_cases(root: &Path) -> Result<Vec<Case>> {
    let path = fixtures_dir(root).join(CASES_FILE);
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let cases: Cases = serde_json::from_str(&text)
        .with_context(|| format!("{} is not a valid case list", path.display()))?;
    let mut seen = std::collections::BTreeSet::new();
    for case in &cases.cases {
        if !seen.insert(case.id.as_str()) {
            bail!("{} lists the case id {:?} twice", path.display(), case.id);
        }
        if case.id.contains(['/', '\\']) || case.id.starts_with('.') || case.id == RECORDED_DIR {
            bail!(
                "{:?} in {} cannot be a directory name here",
                case.id,
                path.display()
            );
        }
    }
    Ok(cases.cases)
}

fn read_recorded(root: &Path, name: &str) -> Result<String> {
    let path = fixtures_dir(root).join(RECORDED_DIR).join(name);
    fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))
}

fn load_examination(root: &Path) -> Result<rocm_core::Examination> {
    let path = root.join(EXAMINATION);
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("{} is not a valid examination", path.display()))
}

/// The recorded examination, edited to be the machine `host` names. The host sets
/// the fields it implies whatever the recording held, and `status` is recomputed
/// from them by the probe's own rule, never typed.
fn examination_for(base: &rocm_core::Examination, host: Host) -> rocm_core::Examination {
    let mut examination = base.clone();
    // Every host states each field `compute_status` and the replay depend on, so
    // what comes out never depends on what the recorded examination happened to
    // hold for it. `os_family` and `probe_failures` count: a recording captured
    // with a failed probe would otherwise turn `amd-found` into `degraded`. So do
    // the container and `/dev/shm` fields the diagnosis reads: a recording made in
    // a container would otherwise put every host in one.
    examination.os_family = "linux".to_owned();
    examination.probe_failures = Vec::new();
    examination.is_wsl = host == Host::WslHostGpuHidden;
    examination.wsl = None;
    let amd_found = matches!(host, Host::AmdFound | Host::ContainerSmallShm);
    examination.has_amd_gpu = amd_found;
    examination.has_discrete_amd = amd_found;
    examination.gpus = Vec::new();
    examination.in_container = false;
    examination.container_kind = String::new();
    examination.shm_total_bytes = None;
    examination.shm_available_bytes = None;
    match host {
        Host::NoAmdGpu | Host::ExamineFails => {}
        Host::WslHostGpuHidden => {
            // A WSL2 distro whose Windows host exposes a GPU to it (`/dev/dxg`,
            // DXCore) but not an AMD one, so nothing ROCm-side is installed. Typed,
            // not probed: the probe fills this from the distro it runs in.
            examination.wsl = Some(rocm_core::examine::WslFacts {
                version: 2,
                dxg_device: true,
                dxcore: true,
                wsl_lib_dir: true,
                ..rocm_core::examine::WslFacts::default()
            });
        }
        Host::AmdFound | Host::ContainerSmallShm => {
            // `pci_id` is what the Linux probe reads: the bus address that starts
            // the `lspci` line.
            examination.gpus = vec![rocm_core::examine::Gpu {
                name: "AMD Radeon RX 7900 XTX".to_owned(),
                gfx_target: "gfx1100".to_owned(),
                pci_id: "0000:c3:00.0".to_owned(),
                is_apu: Some(false),
                is_amd: true,
            }];
            if host == Host::ContainerSmallShm {
                const DEFAULT_SHM_BYTES: u64 = 64 * 1024 * 1024;
                examination.in_container = true;
                examination.container_kind = "docker".to_owned();
                examination.shm_total_bytes = Some(DEFAULT_SHM_BYTES);
                examination.shm_available_bytes = Some(DEFAULT_SHM_BYTES);
            }
        }
    }
    examination.status = examination.compute_status();
    examination
}

/// Everything the stub says for one case.
fn replies_for(root: &Path, case: &Case, recorded: &rocm_core::Examination) -> Result<Replies> {
    let examination = &case
        .host
        .map_or_else(|| recorded.clone(), |host| examination_for(recorded, host));
    let report = rocm_core::run_diagnose(examination, &case.symptom);
    let diagnose_json = Reply::ok(
        serde_json::to_string_pretty(&report).context("failed to render the diagnosis")? + "\n",
    );
    let diagnose_text = Reply::ok(rocm_core::render_diagnose_text(&report, 5));
    let (version, fix_json) = match case.cli {
        Cli::Current => (
            Reply::ok(CURRENT_VERSION.to_owned()),
            Reply::ok(
                rocm_core::catalog_manifest_json()
                    .context("failed to render the catalog manifest")?,
            ),
        ),
        Cli::Older => (
            Reply::ok(read_recorded(root, OLDER_VERSION)?),
            Reply {
                stdout: String::new(),
                stderr: read_recorded(root, OLDER_FIX_JSON_STDERR)?,
                rc: USAGE_ERROR_RC,
            },
        ),
    };
    let examine_json = match case.host {
        None => None,
        Some(Host::ExamineFails) => Some(Reply {
            stdout: String::new(),
            stderr: EXAMINE_FAILS_STDERR.to_owned(),
            rc: EXAMINE_FAILS_RC,
        }),
        Some(_) => Some(Reply::ok(
            serde_json::to_string_pretty(examination)
                .context("failed to render the examination")?
                + "\n",
        )),
    };
    let apply = case
        .apply
        .as_deref()
        .map(|fix_id| apply_for(fix_id, replayed_os(case.host), &report))
        .transpose()?;
    Ok(Replies {
        version,
        fix_json,
        diagnose_json,
        diagnose_text,
        examine_json,
        apply,
    })
}

/// What the stub answers for `rocm fix <fix_id>` in the case's replay, on the platform
/// `os`. The plan is the CLI's own, from [`rocm_core::fix_plan_text`], so its wording
/// cannot drift from the command's. Only an `auto` entry is modelled: what the command
/// prints after the plan differs by class, and the stub copies none of it. And the
/// replayed diagnosis must match the fix, that is list it at or above
/// `min_score_for_match`, or the agent has no way to be pointed at its id.
fn apply_for(fix_id: &str, os: &str, report: &rocm_core::DiagnoseReport) -> Result<Apply> {
    let plan = rocm_core::fix_plan_text(fix_id, os)
        .with_context(|| format!("the catalog has no fix {fix_id:?}"))?;
    if rocm_core::fix::class_on(fix_id, os) != Some(rocm_core::fix::FixClass::Auto) {
        bail!("the apply replay models only a fix that is `auto` on {os}, and {fix_id:?} is not");
    }
    if !report.matched.iter().any(|m| {
        m.score >= report.min_score_for_match
            && m.fix.as_ref().is_some_and(|fix| fix.fix_id == fix_id)
    }) {
        bail!("the replayed diagnosis carries no match for {fix_id:?} that clears the threshold");
    }
    Ok(Apply {
        fix_id: fix_id.to_owned(),
        dry_run: Reply::ok(format!("{plan}\n{}\n", rocm_core::DRY_RUN_NOTICE)),
        bare: Reply {
            stdout: format!("{plan}\n"),
            stderr: format!("{}\n", rocm_core::NON_INTERACTIVE_REFUSAL),
            rc: rocm_core::fix_exit::DECLINED,
        },
    })
}

/// Relative path (under the fixtures directory) and the exact content of every
/// generated file.
fn render(root: &Path) -> Result<Vec<(PathBuf, String)>> {
    let examination = load_examination(root)?;
    load_cases(root)?
        .iter()
        .map(|case| {
            let replies = replies_for(root, case, &examination)?;
            let text =
                serde_json::to_string_pretty(&replies).context("failed to render a replay")? + "\n";
            Ok((PathBuf::from(&case.id).join(REPLIES_FILE), text))
        })
        .collect()
}

/// Where `cases.json` and the eval dataset disagree. The hook finds a replay by
/// case id, so an id the dataset does not have is a recording nothing will ever
/// replay, and the case it was meant for runs with no `rocm` and still passes.
///
/// A symptom is the error text the agent would pass to `rocm diagnose`. A user
/// pastes only its head (the part before the first `:`) into a prompt, so that is
/// what must appear: otherwise the recorded diagnosis answers a different
/// question from the one the case asks.
fn dataset_problems(root: &Path, cases: &[Case]) -> Result<Vec<String>> {
    let path = root.join(EVALS_FILE);
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let evals: Evals = serde_json::from_str(&text)
        .with_context(|| format!("{} is not a valid eval dataset", path.display()))?;
    let mut problems = Vec::new();
    for case in cases {
        let Some(eval) = evals.evaluations.iter().find(|e| e.id == case.id) else {
            problems.push(format!(
                "{CASES_FILE} has the case {:?}, which {EVALS_FILE} does not",
                case.id
            ));
            continue;
        };
        let head = case.symptom.split(':').next().unwrap_or_default().trim();
        if head.is_empty() || !eval.prompt.contains(head) {
            problems.push(format!(
                "the symptom {head:?} of the case {:?} does not appear in its prompt in {EVALS_FILE}",
                case.id
            ));
        }
    }
    Ok(problems)
}

/// Directories under the fixtures directory that no case owns.
fn stray_directories(root: &Path, expected: &[(PathBuf, String)]) -> Result<Vec<String>> {
    let dir = fixtures_dir(root);
    let mut strays = Vec::new();
    for entry in fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == RECORDED_DIR {
            continue;
        }
        if !expected.iter().any(|(rel, _)| rel.starts_with(&name)) {
            strays.push(name);
        }
    }
    strays.sort();
    Ok(strays)
}

fn generate(root: &Path) -> Result<()> {
    let expected = render(root)?;
    let dir = fixtures_dir(root);
    for (relative, content) in &expected {
        let path = dir.join(relative);
        if fs::read_to_string(&path).unwrap_or_default() == *content {
            continue;
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(&path, content).with_context(|| format!("failed to write {}", path.display()))?;
    }
    // A directory no case owns is reported, never deleted: removing files is a
    // decision for the person running this, and `--check` fails until they do.
    for stray in stray_directories(root, &expected)? {
        eprintln!(
            "{} belongs to no case in {CASES_FILE}; remove it by hand",
            dir.join(&stray).display()
        );
    }
    Ok(())
}

fn verify(root: &Path) -> Result<()> {
    let expected = render(root)?;
    let dir = fixtures_dir(root);
    let mut problems = Vec::new();
    for (relative, content) in &expected {
        let path = dir.join(relative);
        match fs::read_to_string(&path) {
            Ok(current) if current == *content => {}
            Ok(_) => problems.push(format!("{} is out of date", relative.display())),
            Err(_) => problems.push(format!("{} is missing", relative.display())),
        }
    }
    for stray in stray_directories(root, &expected)? {
        problems.push(format!("{stray}/ belongs to no case in {CASES_FILE}"));
    }
    problems.extend(dataset_problems(root, &load_cases(root)?)?);
    if problems.is_empty() {
        return Ok(());
    }
    bail!(
        "the recorded `rocm` output the skill evals replay no longer matches the catalog or the dataset:\n  {}\n\
         For a stale or missing recording, run `cargo xtask eval-fixtures` to record it \
         again, then read the diff: a replay that changed is output the skill will now be \
         graded against. A case id or symptom that disagrees with {EVALS_FILE} is fixed by \
         hand in {CASES_FILE} or there.",
        problems.join("\n  ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch workspace holding only what the generator reads, copied from the
    /// real tree, so a test can break a fixture without touching the repository.
    fn scratch() -> (tempfile::TempDir, PathBuf) {
        let real = paths::workspace_root().expect("workspace root");
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().to_path_buf();
        let copy = |relative: &str| {
            let to = root.join(relative);
            fs::create_dir_all(to.parent().expect("parent")).expect("mkdir");
            fs::copy(real.join(relative), &to).expect("copy");
        };
        copy(EXAMINATION);
        copy(EVALS_FILE);
        for file in [
            format!("{FIXTURES_DIR}/{CASES_FILE}"),
            format!("{FIXTURES_DIR}/{RECORDED_DIR}/{OLDER_FIX_JSON_STDERR}"),
            format!("{FIXTURES_DIR}/{RECORDED_DIR}/{OLDER_VERSION}"),
        ] {
            copy(&file);
        }
        (tmp, root)
    }

    fn cases(root: &Path) -> Vec<Case> {
        load_cases(root).expect("cases")
    }

    #[test]
    fn what_is_committed_is_what_the_catalog_says() {
        // The same check CI runs, here so `cargo test` catches a stale recording too.
        let root = paths::workspace_root().expect("workspace root");
        verify(&root).expect("fixtures are out of date; run `cargo xtask eval-fixtures`");
    }

    #[test]
    fn generating_then_checking_passes() {
        let (_tmp, root) = scratch();
        generate(&root).expect("generate");
        verify(&root).expect("a freshly generated tree must verify");
    }

    #[test]
    fn a_host_replays_the_status_the_probe_would_compute() {
        let root = paths::workspace_root().expect("workspace root");
        let recorded = load_examination(&root).expect("examination");
        let mut seen = Vec::new();
        for case in cases(&root) {
            let Some(host) = case.host else { continue };
            seen.push(host);
            let replies = replies_for(&root, &case, &recorded).expect("replies");
            let examine = replies
                .examine_json
                .unwrap_or_else(|| panic!("{} names a host but replays no examine", case.id));
            let (status, has_amd_gpu) = match host {
                Host::ExamineFails => {
                    assert_eq!(examine.rc, EXAMINE_FAILS_RC, "{}", case.id);
                    assert!(examine.stdout.is_empty(), "{}", case.id);
                    assert_eq!(examine.stderr, EXAMINE_FAILS_STDERR, "{}", case.id);
                    continue;
                }
                Host::NoAmdGpu => ("no-amd-gpu", false),
                Host::WslHostGpuHidden => ("wsl", false),
                Host::AmdFound | Host::ContainerSmallShm => ("ok", true),
            };
            let value: serde_json::Value =
                serde_json::from_str(&examine.stdout).expect("an examination is JSON");
            assert_eq!(value["status"], status, "{}", case.id);
            assert_eq!(value["has_amd_gpu"], has_amd_gpu, "{}", case.id);
        }
        for host in [
            Host::NoAmdGpu,
            Host::WslHostGpuHidden,
            Host::AmdFound,
            Host::ExamineFails,
            Host::ContainerSmallShm,
        ] {
            assert!(seen.contains(&host), "no case replays the host {host:?}");
        }
    }

    #[test]
    fn a_host_does_not_depend_on_what_the_recorded_examination_held() {
        /// What one host must come out as, whatever the base held.
        struct Expected {
            host: Host,
            is_wsl: bool,
            has_amd_gpu: bool,
            gpus: usize,
            wsl_facts: bool,
            status: &'static str,
            /// The `/dev/shm` allowance in bytes: only the container host measures one.
            shm_bytes: Option<u64>,
        }

        // A base that says the opposite of every host, so a host that forgot to
        // state a field would show the base's value instead of its own. The
        // recorded examination alone cannot tell: it already is a Linux host with
        // no AMD GPU, so emptying the `NoAmdGpu` arm changed nothing there.
        let root = paths::workspace_root().expect("workspace root");
        let mut hostile = load_examination(&root).expect("examination");
        hostile.is_wsl = true;
        hostile.has_amd_gpu = true;
        hostile.has_discrete_amd = true;
        hostile.gpus = vec![rocm_core::examine::Gpu::default()];
        hostile.wsl = Some(rocm_core::examine::WslFacts::default());
        // `compute_status` reads these two as well, so the base disagrees on them.
        hostile.os_family = "other".to_owned();
        hostile.probe_failures = vec!["lspci missing".to_owned()];
        hostile.status = "degraded".to_owned();
        // A recording made in a container with a measured `/dev/shm`.
        hostile.in_container = true;
        hostile.container_kind = "podman".to_owned();
        hostile.shm_total_bytes = Some(1);
        hostile.shm_available_bytes = Some(1);

        let expected = |host, is_wsl, has_amd_gpu, gpus, wsl_facts, status, shm_bytes| Expected {
            host,
            is_wsl,
            has_amd_gpu,
            gpus,
            wsl_facts,
            status,
            shm_bytes,
        };
        for want in [
            expected(Host::NoAmdGpu, false, false, 0, false, "no-amd-gpu", None),
            // Shares an arm with `NoAmdGpu` today; listed so a later split of that
            // arm is still pinned.
            expected(
                Host::ExamineFails,
                false,
                false,
                0,
                false,
                "no-amd-gpu",
                None,
            ),
            expected(Host::WslHostGpuHidden, true, false, 0, true, "wsl", None),
            expected(Host::AmdFound, false, true, 1, false, "ok", None),
            expected(
                Host::ContainerSmallShm,
                false,
                true,
                1,
                false,
                "ok",
                Some(64 * 1024 * 1024),
            ),
        ] {
            let host = want.host;
            let examination = examination_for(&hostile, host);
            assert_eq!(examination.is_wsl, want.is_wsl, "{host:?}");
            assert_eq!(examination.has_amd_gpu, want.has_amd_gpu, "{host:?}");
            assert_eq!(examination.has_discrete_amd, want.has_amd_gpu, "{host:?}");
            assert_eq!(examination.gpus.len(), want.gpus, "{host:?}");
            assert_eq!(examination.wsl.is_some(), want.wsl_facts, "{host:?}");
            assert_eq!(examination.os_family, "linux", "{host:?}");
            assert!(examination.probe_failures.is_empty(), "{host:?}");
            assert_eq!(examination.status, want.status, "{host:?}");
            // Only the container host is in a container, with the default allowance.
            let in_container = want.shm_bytes.is_some();
            assert_eq!(examination.in_container, in_container, "{host:?}");
            assert_eq!(
                examination.container_kind,
                if in_container { "docker" } else { "" },
                "{host:?}"
            );
            assert_eq!(examination.shm_total_bytes, want.shm_bytes, "{host:?}");
            assert_eq!(examination.shm_available_bytes, want.shm_bytes, "{host:?}");
        }
    }

    #[test]
    fn a_case_with_no_host_records_no_examine_reply() {
        // So the cases recorded before `host` existed are byte-for-byte what they
        // were, and the stub keeps refusing `examine` for them.
        let root = paths::workspace_root().expect("workspace root");
        let mut checked = 0;
        for case in cases(&root).iter().filter(|case| case.host.is_none()) {
            let path = fixtures_dir(&root).join(&case.id).join(REPLIES_FILE);
            let text = fs::read_to_string(&path).expect("replies");
            assert!(!text.contains("examine_json"), "{}", case.id);
            checked += 1;
        }
        assert!(checked > 0, "no case without a host to check");
    }

    #[test]
    fn a_changed_fixture_fails_the_check_and_names_it() {
        let (_tmp, root) = scratch();
        generate(&root).expect("generate");
        let id = &cases(&root)[0].id;
        let path = fixtures_dir(&root).join(id).join(REPLIES_FILE);
        fs::write(&path, "{}\n").expect("overwrite");
        let message = format!(
            "{:#}",
            verify(&root).expect_err("a changed fixture must fail")
        );
        // The path is printed with the platform's separator.
        let named = Path::new(id).join(REPLIES_FILE);
        assert!(
            message.contains(&format!("{} is out of date", named.display())),
            "{message}"
        );
        assert!(message.contains("cargo xtask eval-fixtures"), "{message}");
    }

    #[test]
    fn a_missing_fixture_fails_the_check() {
        let (_tmp, root) = scratch();
        let message = format!("{:#}", verify(&root).expect_err("nothing generated yet"));
        assert!(message.contains("is missing"), "{message}");
    }

    #[test]
    fn a_directory_no_case_owns_fails_the_check_and_generate_leaves_it_alone() {
        let (_tmp, root) = scratch();
        generate(&root).expect("generate");
        let stray = fixtures_dir(&root).join("left-over");
        fs::create_dir_all(&stray).expect("mkdir");
        let message = format!(
            "{:#}",
            verify(&root).expect_err("a stray directory must fail")
        );
        assert!(
            message.contains("left-over/ belongs to no case"),
            "{message}"
        );
        generate(&root).expect("generate again");
        assert!(stray.exists(), "generate must never delete a directory");
        // The person running it removes the directory, and the check passes again.
        fs::remove_dir_all(&stray).expect("remove");
        verify(&root).expect("clean again");
    }

    #[test]
    fn the_recorded_directory_is_never_treated_as_stray() {
        let (_tmp, root) = scratch();
        generate(&root).expect("generate");
        assert!(fixtures_dir(&root).join(RECORDED_DIR).is_dir());
        verify(&root).expect("recorded/ is an input, not a stray");
    }

    #[test]
    fn a_current_cli_replays_the_manifest_the_catalog_publishes() {
        let (_tmp, root) = scratch();
        let examination = load_examination(&root).expect("examination");
        let case = cases(&root)
            .into_iter()
            .find(|c| c.cli == Cli::Current)
            .expect("a current-CLI case");
        let replies = replies_for(&root, &case, &examination).expect("replies");
        assert_eq!(replies.fix_json.rc, 0);
        let manifest: serde_json::Value =
            serde_json::from_str(&replies.fix_json.stdout).expect("manifest is JSON");
        assert_eq!(
            manifest["contract_version"],
            serde_json::json!(rocm_core::CATALOG_CONTRACT_VERSION),
            "the replayed manifest must carry the contract version the catalog declares"
        );
    }

    #[test]
    fn an_older_cli_replays_the_usage_error_a_real_one_printed() {
        let (_tmp, root) = scratch();
        let examination = load_examination(&root).expect("examination");
        let case = cases(&root)
            .into_iter()
            .find(|c| c.cli == Cli::Older)
            .expect("an older-CLI case");
        let replies = replies_for(&root, &case, &examination).expect("replies");
        assert_eq!(replies.fix_json.rc, USAGE_ERROR_RC);
        assert!(
            replies.fix_json.stdout.is_empty(),
            "an older CLI printed nothing on stdout"
        );
        assert!(
            replies
                .fix_json
                .stderr
                .contains("unexpected argument '--json'"),
            "{}",
            replies.fix_json.stderr
        );
    }

    #[test]
    fn a_replayed_diagnosis_is_the_one_the_catalog_gives_for_the_symptom() {
        let (_tmp, root) = scratch();
        let examination = load_examination(&root).expect("examination");
        let case = &cases(&root)[0];
        let replies = replies_for(&root, case, &examination).expect("replies");
        let report: serde_json::Value =
            serde_json::from_str(&replies.diagnose_json.stdout).expect("diagnosis is JSON");
        assert!(
            report.get("has_match").is_some(),
            "the skill gates on `has_match`"
        );
        assert!(report.get("route_when_no_match").is_some());
    }

    /// What the replayed diagnosis of one case says.
    struct Replayed {
        has_match: bool,
        /// Ranked `(id, score, fix class)`.
        matches: Vec<(String, u64, String)>,
        threshold: u64,
        /// The score from which a match is high confidence.
        high_threshold: u64,
        /// Where `route_when_no_match` points.
        route_target: String,
        /// The preview the stub gives for the case's `apply` fix, if it names one.
        apply_plan: Option<String>,
    }

    fn replayed(case_id: &str) -> Replayed {
        let root = paths::workspace_root().expect("workspace root");
        let examination = load_examination(&root).expect("examination");
        let case = cases(&root)
            .into_iter()
            .find(|case| case.id == case_id)
            .unwrap_or_else(|| panic!("no case {case_id}"));
        let replies = replies_for(&root, &case, &examination).expect("replies");
        let report: serde_json::Value =
            serde_json::from_str(&replies.diagnose_json.stdout).expect("diagnosis is JSON");
        let matches = report["matched"]
            .as_array()
            .expect("matched is a list")
            .iter()
            .map(|m| {
                (
                    m["id"].as_str().expect("id").to_owned(),
                    m["score"].as_u64().expect("score"),
                    m["fix"]["class"].as_str().expect("class").to_owned(),
                )
            })
            .collect();
        Replayed {
            has_match: report["has_match"].as_bool().expect("has_match"),
            matches,
            threshold: report["min_score_for_match"].as_u64().expect("threshold"),
            high_threshold: report["high_confidence_threshold"]
                .as_u64()
                .expect("high threshold"),
            route_target: report["route_when_no_match"]["target"]
                .as_str()
                .expect("route target")
                .to_owned(),
            apply_plan: replies.apply.map(|apply| apply.dry_run.stdout),
        }
    }

    /// The judged cases rest on what their replayed diagnosis holds: a case that
    /// grades "the agent did not act on a sub-threshold match" is void if the
    /// recording stops containing one, and the dataset's notes say so in prose
    /// nothing else checks. This pins each such premise.
    #[test]
    fn the_premise_a_judged_case_rests_on_is_what_its_replay_records() {
        // The failing app is not on the host, so nothing was established.
        for id in [
            "rocm-lemonade-routes-upstream",
            "rocm-serve-failure-routes-to-this-project",
            "rocm-ollama-routes-upstream",
            "rocm-lm-studio-routes-upstream",
        ] {
            let r = replayed(id);
            assert!(!r.has_match, "{id}: the replay must establish no cause");
            assert!(r.matches.is_empty(), "{id}: nothing is listed either");
            // The tracker the diagnosis hands back is never the app's own, which
            // is what lets these cases tell the skill's routing from `route_when_no_match`.
            assert_eq!(r.route_target, "rocm-core", "{id}");
        }

        // `has_match` false while `matched` is not empty: the trap an agent gating
        // on emptiness falls into. The route is the ROCm core tracker although the
        // prompt names PyTorch, which is what the case's routing item turns on.
        let r = replayed("rocm-current-cli-carries-on");
        assert!(!r.has_match);
        assert!(
            !r.matches.is_empty(),
            "a sub-threshold match must be listed"
        );
        assert_eq!(r.route_target, "rocm-core");

        // A vLLM failure described in words matches neither vLLM entry, which score
        // only the symptom text, so the case's agent has nothing to route on until it
        // has asked for the error line.
        let r = replayed("rocm-vllm-failure-in-words-asks-for-the-error-line");
        assert!(!r.has_match);
        assert!(
            r.matches
                .iter()
                .all(|m| m.0 != "fix-16-vllm-oom" && m.0 != "fix-17-torch-dlpack"),
            "{:?}",
            r.matches
        );
        assert_eq!(r.route_target, "rocm-core");

        // The vLLM import error is a match, but a likely one, not a high-confidence one:
        // the case grades "likely rather than certain".
        let r = replayed("rocm-vllm-import-error");
        assert!(r.has_match);
        assert_eq!(r.matches[0].0, "fix-17-torch-dlpack");
        assert!(
            (r.threshold..r.high_threshold).contains(&r.matches[0].1),
            "{:?}",
            r.matches[0]
        );
        // The vLLM out-of-memory error is the matched cause, and print-only.
        let r = replayed("rocm-vllm-oom-at-startup");
        assert!(r.has_match);
        assert_eq!(r.matches[0].0, "fix-16-vllm-oom");
        assert_eq!(r.matches[0].2, "print-only");

        // An `auto` entry is the cause, so the apply step is reached.
        let r = replayed("rocm-apply-auto-fix-needs-consent");
        assert!(r.has_match);
        assert_eq!(r.matches[0].0, "fix-4-render-group");
        assert_eq!(r.matches[0].2, "auto");
        // Two of the case's expected items (log out and back in, what happens when
        // `sudo` runs with no input) rest on these flags, which show in the plan the
        // stub prints.
        let plan = r.apply_plan.expect("the case names a fix to apply");
        assert!(plan.contains("requires sudo"), "{plan}");
        assert!(plan.contains("requires re-login"), "{plan}");
        assert!(!plan.contains("requires reboot"), "{plan}");
        // A host-only entry outranks the one that explains the paste, and both
        // clear the threshold.
        let r = replayed("rocm-match-does-not-explain-the-paste");
        assert!(r.has_match);
        assert_eq!(r.matches[0].0, "fix-19-shm-too-small");
        // 60 for the measured allowance plus 10 only inside a container
        // (`check_19_shm_too_small`), so a host that lost its container flag
        // scores 60 and no longer tells the case's story.
        assert_eq!(r.matches[0].1, 70, "the shared-memory entry in a container");
        let container = r
            .matches
            .iter()
            .find(|m| m.0 == "fix-10-container")
            .expect("a container entry is listed");
        assert!(container.1 < r.threshold, "and it is below the threshold");
        let torch = r
            .matches
            .iter()
            .find(|m| m.0 == "fix-17-torch-dlpack")
            .expect("the entry explaining the paste is listed");
        assert!(
            torch.1 < r.matches[0].1,
            "it ranks below the host-only entry"
        );
        assert!(torch.1 >= r.threshold, "it clears the threshold");

        // Everything that clears the threshold comes from host state: the
        // complaint is mentioned by nothing, and the route is ROCm core.
        let r = replayed("rocm-host-only-match-is-not-the-cause");
        assert!(r.has_match);
        let cleared: Vec<&str> = r
            .matches
            .iter()
            .filter(|m| m.1 >= r.threshold)
            .map(|m| m.0.as_str())
            .collect();
        assert_eq!(cleared, ["fix-19-shm-too-small"]);
        assert_eq!(r.route_target, "rocm-core");
    }

    #[test]
    fn an_apply_replay_needs_the_replayed_diagnosis_to_match_the_fix() {
        let root = paths::workspace_root().expect("workspace root");
        let examination = load_examination(&root).expect("examination");
        // This paste makes `fix-4-render-group` clear the threshold, so a diagnosis
        // that matches something is not enough: it has to be the asked-for fix.
        let report =
            rocm_core::run_diagnose(&examination, "Unable to open /dev/kfd: Permission denied");
        assert!(
            report.has_match,
            "the premise: a match cleared the threshold"
        );
        assert!(
            report
                .matched
                .iter()
                .any(|m| m.score >= report.min_score_for_match
                    && m.fix
                        .as_ref()
                        .is_some_and(|f| f.fix_id == "fix-4-render-group")),
            "the premise: it is fix-4-render-group"
        );
        // `fix-6-path` is `auto` on Linux and is not what this diagnosis matched.
        let message = format!(
            "{:#}",
            apply_for("fix-6-path", "linux", &report)
                .expect_err("a fix the diagnosis does not match is refused")
        );
        assert!(message.contains("carries no match"), "{message}");
    }

    #[test]
    fn an_apply_replay_is_the_catalogs_plan_followed_by_the_dry_run_notice() {
        let root = paths::workspace_root().expect("workspace root");
        let examination = load_examination(&root).expect("examination");
        let case = cases(&root)
            .into_iter()
            .find(|case| case.apply.is_some())
            .expect("a case names a fix to apply");
        let apply = replies_for(&root, &case, &examination)
            .expect("replies")
            .apply
            .expect("an `apply` case replays the fix commands");
        let plan =
            rocm_core::fix_plan_text(&apply.fix_id, replayed_os(case.host)).expect("the plan");
        assert_eq!(
            apply.dry_run.stdout,
            format!("{plan}\n{}\n", rocm_core::DRY_RUN_NOTICE)
        );
        // What the plan says is the CLI's, not a second rendering of it. How it is laid
        // out around the notice is this generator's own, so the equality above checks
        // the replay against the generator and not against the command.
        for line in [
            "Fix:        ",
            "OS scope:   ",
            "Rationale:  ",
            "Commands:",
            "Flags:      ",
            "Verify:     ",
        ] {
            assert!(plan.contains(line), "{plan}");
        }
    }

    #[test]
    fn an_apply_replay_needs_the_fix_to_clear_the_threshold_not_just_be_listed() {
        let root = paths::workspace_root().expect("workspace root");
        let examination = load_examination(&root).expect("examination");
        // This paste lists `fix-4-render-group` below `min_score_for_match`.
        let report = rocm_core::run_diagnose(&examination, "/dev/kfd: Permission denied");
        assert!(
            !report.has_match,
            "the premise: nothing cleared the threshold"
        );
        assert!(
            report.matched.iter().any(|m| m
                .fix
                .as_ref()
                .is_some_and(|f| f.fix_id == "fix-4-render-group")),
            "the premise: the fix is listed"
        );
        let message = format!(
            "{:#}",
            apply_for("fix-4-render-group", "linux", &report)
                .expect_err("a listed but sub-threshold fix is refused")
        );
        assert!(message.contains("clears the threshold"), "{message}");
    }

    #[test]
    fn an_apply_replay_names_an_unknown_fix_and_refuses_one_the_platform_lacks() {
        let root = paths::workspace_root().expect("workspace root");
        let examination = load_examination(&root).expect("examination");
        let report = rocm_core::run_diagnose(&examination, "hipErrorNoBinaryForGpu");
        let message = format!(
            "{:#}",
            apply_for("fix-0-nope", "linux", &report).expect_err("an unknown fix is refused")
        );
        assert!(message.contains("the catalog has no fix"), "{message}");
        // `fix-4-render-group` is `auto` on Linux only, and the WSL host is `wsl`.
        let message = format!(
            "{:#}",
            apply_for(
                "fix-4-render-group",
                replayed_os(Some(Host::WslHostGpuHidden)),
                &report
            )
            .expect_err("a fix that does not apply on WSL is refused")
        );
        assert!(message.contains("`auto` on wsl"), "{message}");
    }

    #[test]
    fn an_apply_replay_is_refused_for_a_fix_that_is_not_auto() {
        let root = paths::workspace_root().expect("workspace root");
        let examination = load_examination(&root).expect("examination");
        let host = examination_for(&examination, Host::ContainerSmallShm);
        let report = rocm_core::run_diagnose(&host, "ImportError: libtorch_cuda.so");
        // `fix-19-shm-too-small` is in this diagnosis and is print-only.
        let message = format!(
            "{:#}",
            apply_for("fix-19-shm-too-small", "linux", &report)
                .expect_err("a print-only fix is refused")
        );
        assert!(message.contains("`auto` on linux"), "{message}");
    }

    #[test]
    fn an_apply_case_replays_the_plan_the_catalog_gives_and_the_refusal_of_a_shell_with_no_terminal()
     {
        let root = paths::workspace_root().expect("workspace root");
        let examination = load_examination(&root).expect("examination");
        let case = cases(&root)
            .into_iter()
            .find(|case| case.apply.is_some())
            .expect("a case names a fix to apply");
        let apply = replies_for(&root, &case, &examination)
            .expect("replies")
            .apply
            .expect("an `apply` case replays the fix commands");
        let title = rocm_core::catalog_manifest()
            .entries
            .into_iter()
            .find(|entry| entry.id == apply.fix_id)
            .expect("the catalog has the fix")
            .title;
        assert!(apply.dry_run.stdout.contains(&title));
        assert!(apply.dry_run.stdout.contains(rocm_core::DRY_RUN_NOTICE));
        assert_eq!(apply.dry_run.rc, rocm_core::fix_exit::OK);
        assert!(apply.bare.stdout.contains(&title));
        assert!(!apply.bare.stdout.contains(rocm_core::DRY_RUN_NOTICE));
        assert_eq!(apply.bare.rc, rocm_core::fix_exit::DECLINED);
        assert_eq!(
            apply.bare.stderr,
            format!("{}\n", rocm_core::NON_INTERACTIVE_REFUSAL)
        );
    }

    #[test]
    fn a_case_id_cannot_escape_the_fixtures_directory() {
        let (_tmp, root) = scratch();
        let path = fixtures_dir(&root).join(CASES_FILE);
        for bad in ["../x", "a/b", "a\\b", ".hidden", "recorded"] {
            // Built with serde_json so the backslash is escaped in the JSON text.
            let text =
                serde_json::json!({"cases": [{"id": bad, "cli": "current", "symptom": "x"}]});
            fs::write(&path, text.to_string()).expect("write");
            assert!(load_cases(&root).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn a_case_id_listed_twice_is_refused() {
        let (_tmp, root) = scratch();
        let path = fixtures_dir(&root).join(CASES_FILE);
        fs::write(
            &path,
            r#"{"cases":[{"id":"a","cli":"current","symptom":"x"},{"id":"a","cli":"older","symptom":"y"}]}"#,
        )
        .expect("write");
        assert!(load_cases(&root).is_err());
    }

    /// Rewrite the scratch copy of the dataset with `edit` applied to its text.
    fn edit_evals(root: &Path, edit: impl FnOnce(String) -> String) {
        let path = root.join(EVALS_FILE);
        let text = fs::read_to_string(&path).expect("read evals");
        fs::write(&path, edit(text)).expect("write evals");
    }

    #[test]
    fn a_case_id_the_dataset_does_not_have_fails_the_check() {
        let (_tmp, root) = scratch();
        generate(&root).expect("generate");
        let id = cases(&root)[0].id.clone();
        edit_evals(&root, |t| {
            t.replace(&format!("\"id\": \"{id}\""), "\"id\": \"renamed\"")
        });
        let message = format!("{:#}", verify(&root).expect_err("a renamed eval must fail"));
        assert!(
            message.contains(&format!("the case {id:?}, which")),
            "{message}"
        );
    }

    #[test]
    fn a_symptom_missing_from_the_prompt_fails_the_check() {
        let (_tmp, root) = scratch();
        generate(&root).expect("generate");
        edit_evals(&root, |t| {
            t.replace("hipErrorNoBinaryForGpu on my AMD", "an error on my AMD")
        });
        let message = format!(
            "{:#}",
            verify(&root).expect_err("a drifted prompt must fail")
        );
        assert!(
            message.contains("does not appear in its prompt"),
            "{message}"
        );
    }

    #[test]
    fn the_committed_cases_all_name_a_prompt_that_carries_their_symptom() {
        let root = paths::workspace_root().expect("workspace root");
        let problems = dataset_problems(&root, &cases(&root)).expect("dataset");
        assert!(problems.is_empty(), "{problems:?}");
    }

    /// The skill's own prose, which an agent reads and which a pin must never quote.
    const SKILL_TEXT: [&str; 2] = [
        "skills/rocm-doctor/SKILL.md",
        "skills/rocm-doctor/reference.md",
    ];

    /// What is wrong with the `logs_contain` pins of the replayed cases. A pin
    /// must be text the replay prints (or it fails every real run), and must not
    /// be in the skill's own text (or it passes whenever the skill loads, and
    /// stops proving the replay ran). skillscope matches a pin as a
    /// case-insensitive substring of the whole transcript, so that is how it is
    /// compared here.
    fn pin_problems(root: &Path) -> Vec<String> {
        let text = fs::read_to_string(root.join(EVALS_FILE)).expect("evals");
        let evals: serde_json::Value = serde_json::from_str(&text).expect("json");
        let skill_text: Vec<(&str, String)> = SKILL_TEXT
            .iter()
            .map(|f| {
                (
                    *f,
                    fs::read_to_string(root.join(f))
                        .expect("skill text")
                        .to_lowercase(),
                )
            })
            .collect();
        let mut problems = Vec::new();
        for case in cases(root) {
            let replies = fs::read_to_string(fixtures_dir(root).join(&case.id).join(REPLIES_FILE))
                .expect("replies")
                .to_lowercase();
            let eval = evals["evaluations"]
                .as_array()
                .and_then(|all| all.iter().find(|e| e["id"] == case.id.as_str()))
                .expect("the case is in the dataset");
            let pins = eval["logs_contain"].as_array().cloned().unwrap_or_default();
            if pins.is_empty() {
                problems.push(format!("{} pins nothing", case.id));
            }
            for pin in pins {
                let pin = pin.as_str().expect("string pin").to_lowercase();
                if !replies.contains(&pin) {
                    problems.push(format!("{}: {pin:?} is not in its replay", case.id));
                }
                for (file, body) in &skill_text {
                    if body.contains(&pin) {
                        problems.push(format!("{}: {pin:?} is quoted in {file}", case.id));
                    }
                }
            }
        }
        problems
    }

    #[test]
    fn a_replayed_case_pins_only_text_its_replay_prints_and_the_skill_does_not() {
        let root = paths::workspace_root().expect("workspace root");
        let problems = pin_problems(&root);
        assert!(problems.is_empty(), "{problems:#?}");
    }

    /// A scratch copy that `pin_problems` can read: generated replays, and the skill
    /// text it compares the pins with. It starts clean.
    fn scratch_for_pins() -> (tempfile::TempDir, PathBuf) {
        let (tmp, root) = scratch();
        generate(&root).expect("generate");
        let real = paths::workspace_root().expect("workspace root");
        for file in SKILL_TEXT {
            let to = root.join(file);
            fs::create_dir_all(to.parent().expect("parent")).expect("mkdir");
            fs::copy(real.join(file), &to).expect("copy");
        }
        assert!(pin_problems(&root).is_empty(), "the copy starts clean");
        (tmp, root)
    }

    /// Set the `logs_contain` pins of one case in the scratch copy of the dataset.
    fn set_pins(root: &Path, case_id: &str, pins: &[&str]) {
        edit_evals(root, |text| {
            let mut evals: serde_json::Value = serde_json::from_str(&text).expect("json");
            let eval = evals["evaluations"]
                .as_array_mut()
                .and_then(|all| all.iter_mut().find(|e| e["id"] == case_id))
                .expect("the case is in the dataset");
            eval["logs_contain"] = serde_json::json!(pins);
            serde_json::to_string_pretty(&evals).expect("json")
        });
    }

    #[test]
    fn a_replayed_case_that_pins_nothing_is_caught() {
        let (_tmp, root) = scratch_for_pins();
        set_pins(&root, "rocm-current-cli-carries-on", &[]);
        let problems = pin_problems(&root);
        assert!(
            problems
                .iter()
                .any(|p| p == "rocm-current-cli-carries-on pins nothing"),
            "{problems:?}"
        );
    }

    #[test]
    fn a_pin_the_replay_does_not_print_is_caught() {
        let (_tmp, root) = scratch_for_pins();
        set_pins(
            &root,
            "rocm-current-cli-carries-on",
            &["text no replay prints"],
        );
        let problems = pin_problems(&root);
        assert!(
            problems.iter().any(|p| p.contains("is not in its replay")),
            "{problems:?}"
        );
    }

    #[test]
    fn a_pin_the_skill_text_quotes_is_caught() {
        let (_tmp, root) = scratch_for_pins();
        // A later skill edit that quotes a pin, in a different case.
        let mut skill = fs::OpenOptions::new()
            .append(true)
            .open(root.join(SKILL_TEXT[0]))
            .expect("open");
        std::io::Write::write_all(&mut skill, b"\nExpect: RECORDED FOR EVALS\n").expect("append");
        let problems = pin_problems(&root);
        assert!(
            problems
                .iter()
                .any(|p| p.contains("is quoted in skills/rocm-doctor/SKILL.md")),
            "{problems:?}"
        );
    }

    /// A Python that can run the eval hook's own tests, or `None`.
    fn python() -> Option<&'static str> {
        ["python3", "python"].into_iter().find(|name| {
            std::process::Command::new(name)
                .args(["-I", "-c", "import unittest"])
                .output()
                .is_ok_and(|out| out.status.success())
        })
    }

    #[test]
    fn the_eval_hook_tests_pass() {
        // `hooks.py` is what keeps the recorded `rocm` from running anything but a
        // replay, so its tests (replay, refusal, PATH and temp-dir cleanup) run
        // with the rest of this crate. Python is on every lane that runs `cargo
        // test` in CI, so on CI a missing Python is a failure; on a machine
        // without one the test says so and stops.
        let Some(python) = python() else {
            assert!(
                std::env::var_os("CI").is_none(),
                "no python3/python on PATH: the eval hook tests cannot run on CI"
            );
            eprintln!(
                "SKIPPED: no python3/python on PATH, so skills/rocm-doctor/evals/test_hooks.py did not run"
            );
            return;
        };
        let root = paths::workspace_root().expect("workspace root");
        let script = root.join("skills/rocm-doctor/evals/test_hooks.py");
        let out = std::process::Command::new(python)
            .arg("-I")
            .arg(&script)
            .output()
            .expect("run python");
        assert!(
            out.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
