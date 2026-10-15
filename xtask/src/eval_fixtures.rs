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
//! Everything a current CLI would print is therefore rendered from the compiled
//! catalog, never typed in: the manifest comes from
//! [`rocm_core::catalog_manifest_json`] and a diagnosis from
//! [`rocm_core::run_diagnose`], run over a recorded examination. `--check` fails
//! when the committed copy differs, so a catalog change cannot leave a stale
//! recording behind, the same guard `cargo xtask catalog --check` gives the manifest.
//!
//! The one thing that cannot be rendered is what an OLDER CLI printed. That was
//! recorded once from a real older binary and is committed under `recorded/`.

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
/// What a current CLI prints for `--version` in a replay. Fixed text, so a
/// rebuild never makes a recording differ.
const CURRENT_VERSION: &str = "rocm-cli 0.1.0 (recorded for evals)\n";
/// An older CLI rejects `--json` with a usage error, and clap's usage-error code is 2.
const USAGE_ERROR_RC: i32 = 2;

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

/// Everything the stub says for one case.
fn replies_for(root: &Path, case: &Case, examination: &rocm_core::Examination) -> Result<Replies> {
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
    Ok(Replies {
        version,
        fix_json,
        diagnose_json,
        diagnose_text,
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
    if problems.is_empty() {
        return Ok(());
    }
    bail!(
        "the recorded `rocm` output the skill evals replay no longer matches the catalog:\n  {}\n\
         Run `cargo xtask eval-fixtures` to record it again, then read the diff: a replay \
         that changed is output the skill will now be graded against.",
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

    #[test]
    fn a_case_id_cannot_escape_the_fixtures_directory() {
        let (_tmp, root) = scratch();
        let path = fixtures_dir(&root).join(CASES_FILE);
        for bad in ["../x", "a/b", ".hidden", "recorded"] {
            fs::write(
                &path,
                format!(r#"{{"cases":[{{"id":"{bad}","cli":"current","symptom":"x"}}]}}"#),
            )
            .expect("write");
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
}
