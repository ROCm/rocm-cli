// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! A stand-in for the GitHub CLI, for scenarios that install a pull request's CI
//! build through the installers' `pr-<number>` channel.
//!
//! The installers make three kinds of call, and this answers each from files in
//! the directory `FAKE_GH_DIR` names, so a scenario decides what GitHub "has":
//!
//! - `head`: the pull request's head commit; missing or empty makes
//!   `gh pr view` fail the way an unknown pull request does
//! - `runs`: one `<id> <status> <conclusion> [<workflow path>]` line per run,
//!   newest first; the path defaults to the CI workflow
//! - `artifacts-<id>`: how many unexpired artifacts run `<id>` still has, and
//!   `expired-<id>`: how many expired ones it also lists
//! - `bundle/`: the directory whose files `gh run download` copies into `-D`
//! - `calls`: every invocation is appended here, one line each, and a download
//!   adds `downloaded <id>`
//!
//! `install.sh` asks `gh` to filter with `--jq` and `install.ps1` parses JSON
//! itself, so each `api` answer is the filtered text when `--jq` is present and
//! the JSON object the real API returns otherwise.
//!
//! It is as strict as the real API, so an installer that drops an argument
//! fails here instead of passing against a lenient stand-in: a call naming
//! another repository than `FAKE_GH_REPO` (default `ROCm/rocm-cli`), a run
//! listing for any commit but the pull request's head or not limited to pull
//! request runs, an artifact lookup or download for another artifact than this
//! platform's pull request bundle, and a download that does not name it all
//! fail.
//!
//! A Rust binary rather than a shell script so the scenarios run on Windows too.

use std::io::Write as _;
use std::path::{Path, PathBuf};

/// The artifact CI uploads the pull request bundle as on this platform.
const ARTIFACT: &str = if cfg!(windows) {
    "rocm-cli-pr-windows-amd64"
} else {
    "rocm-cli-pr-linux-amd64"
};

fn die(message: &str) -> ! {
    eprintln!("fake-gh: {message}");
    std::process::exit(1);
}

/// The word after `flag`, as `gh` reads `-R owner/repo`.
fn arg_after<'a>(words: &[&'a str], flag: &str) -> Option<&'a str> {
    let at = words.iter().position(|word| *word == flag)?;
    words.get(at + 1).copied()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir =
        PathBuf::from(std::env::var_os("FAKE_GH_DIR").expect("FAKE_GH_DIR names the stub state"));
    append_line(&dir.join("calls"), &args.join(" "));
    let repo = std::env::var("FAKE_GH_REPO").unwrap_or_else(|_| "ROCm/rocm-cli".to_owned());

    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    match words.as_slice() {
        ["pr", "view", ..] => {
            if arg_after(&words, "-R") != Some(repo.as_str()) {
                die(&format!("pr view without -R {repo}: {args:?}"));
            }
            // Without these a real gh prints a human summary, which would
            // become the commit.
            if arg_after(&words, "--json") != Some("headRefOid")
                || arg_after(&words, "--jq") != Some(".headRefOid")
            {
                die(&format!(
                    "pr view without --json headRefOid --jq .headRefOid: {args:?}"
                ));
            }
            pr_view(&dir);
        }
        ["api", path, ..] => api(&dir, &repo, path, arg_after(&words, "--jq")),
        ["run", "download", run, ..] => {
            if arg_after(&words, "-R") != Some(repo.as_str()) {
                die(&format!("run download without -R {repo}: {args:?}"));
            }
            if arg_after(&words, "-n") != Some(ARTIFACT) {
                die(&format!("run download without -n {ARTIFACT}: {args:?}"));
            }
            download(&dir, run, &words);
        }
        _ => die(&format!("unexpected invocation: {args:?}")),
    }
}

fn append_line(path: &Path, line: &str) {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open the call log");
    writeln!(file, "{line}").expect("write the call log");
}

fn read_trimmed(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn pr_view(dir: &Path) {
    let Some(head) = read_trimmed(&dir.join("head")) else {
        eprintln!("GraphQL: Could not resolve to a PullRequest with that number.");
        std::process::exit(1);
    };
    println!("{head}");
}

const CI_PATH: &str = ".github/workflows/ci.yml";

/// The number in the stub file `name`, 0 when it is absent.
fn count_in(dir: &Path, name: &str) -> usize {
    read_trimmed(&dir.join(name))
        .and_then(|text| text.parse().ok())
        .unwrap_or(0)
}

/// `jq` is the `--jq` expression when the caller asked `gh` to filter, and then
/// the answer is what that expression would leave: only when it carries the
/// expiry or the workflow filter are expired artifacts and other workflows'
/// runs left out, so a caller that drops either is handed them, as it would be
/// by the real API. Without `--jq` the full JSON is printed and the caller
/// filters it itself.
fn api(dir: &Path, repo: &str, path: &str, jq: Option<&str>) {
    if !path.starts_with(&format!("repos/{repo}/")) {
        die(&format!("api path outside {repo}: {path}"));
    }
    if let Some(rest) = path.split("/actions/runs/").nth(1) {
        if !path.ends_with(&format!("/artifacts?name={ARTIFACT}")) {
            die(&format!("artifact lookup for another artifact: {path}"));
        }
        let run = rest.split('/').next().unwrap_or_default();
        let live = count_in(dir, &format!("artifacts-{run}"));
        let expired = count_in(dir, &format!("expired-{run}"));
        match jq {
            Some(expr) if expr.contains("select(.expired | not)") => println!("{live}"),
            Some(_) => println!("{}", live + expired),
            None => {
                let entries = [
                    vec![r#"{"expired":true}"#; expired],
                    vec![r#"{"expired":false}"#; live],
                ]
                .concat()
                .join(",");
                println!(r#"{{"artifacts":[{entries}]}}"#);
            }
        }
    } else if path.contains("/actions/runs?") {
        let head = read_trimmed(&dir.join("head")).unwrap_or_default();
        if !path.contains(&format!("head_sha={head}")) {
            die(&format!(
                "run listing for another commit than {head}: {path}"
            ));
        }
        // GitHub lists 30 runs by default; a commit with more could hide the
        // one that has a build.
        if !path.contains("per_page=100") {
            die(&format!("run listing without per_page=100: {path}"));
        }
        if !path.contains("event=pull_request") {
            die(&format!(
                "run listing not limited to pull request runs: {path}"
            ));
        }
        let runs = read_trimmed(&dir.join("runs")).unwrap_or_default();
        // `<id> <status> <conclusion> [<workflow path>]`, the path defaulting
        // to the CI workflow.
        let runs: Vec<Vec<&str>> = runs
            .lines()
            .map(|line| line.split_whitespace().collect())
            .collect();
        let workflow_of = |run: &Vec<&str>| run.get(3).copied().unwrap_or(CI_PATH).to_owned();
        if let Some(expr) = jq {
            let ci_only = expr.contains(&format!(".path == \"{CI_PATH}\""));
            for run in runs
                .iter()
                .filter(|r| !ci_only || workflow_of(r) == CI_PATH)
            {
                println!("{} {} {}", run[0], run[1], run[2]);
            }
        } else {
            let json: Vec<String> = runs
                .iter()
                .map(|run| {
                    let conclusion = match run[2] {
                        "-" => "null".to_owned(),
                        other => format!("\"{other}\""),
                    };
                    format!(
                        r#"{{"id":{},"path":"{}","status":"{}","conclusion":{conclusion}}}"#,
                        run[0],
                        workflow_of(run),
                        run[1]
                    )
                })
                .collect();
            println!(r#"{{"workflow_runs":[{}]}}"#, json.join(","));
        }
    } else {
        die(&format!("unexpected api path: {path}"));
    }
}

fn download(dir: &Path, run: &str, rest: &[&str]) {
    let dest = rest
        .iter()
        .position(|arg| *arg == "-D")
        .and_then(|at| rest.get(at + 1))
        .expect("gh run download -D <dir>");
    std::fs::create_dir_all(dest).expect("create the download directory");
    for entry in std::fs::read_dir(dir.join("bundle")).expect("read the stub bundle") {
        let entry = entry.expect("stub bundle entry");
        std::fs::copy(entry.path(), Path::new(dest).join(entry.file_name()))
            .expect("copy a bundle file");
    }
    append_line(&dir.join("calls"), &format!("downloaded {run}"));
}
