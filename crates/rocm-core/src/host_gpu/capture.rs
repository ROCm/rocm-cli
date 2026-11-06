// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Optional-command capture for the host probes.
//!
//! Runs a tool that may not be installed under a short timeout and keeps its
//! stdout only on success, resolving the program the way the platform does
//! (`PATHEXT` and the fixed Windows system locations included). A missing,
//! slow or failing tool reads as `None`, never as an error.

use crate::runtime_is_windows;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub(super) const OPTIONAL_COMMAND_TIMEOUT: Duration = Duration::from_millis(1_500);

pub(super) fn capture_optional_command(program: &str, args: &[&str]) -> Option<String> {
    capture_optional_command_with_timeout(program, args, OPTIONAL_COMMAND_TIMEOUT)
}

pub(super) fn capture_optional_command_with_timeout(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    for candidate in tool_path_candidates(program) {
        if let Some(output) =
            capture_optional_command_candidate_with_timeout(Path::new(&candidate), args, timeout)
        {
            return Some(output);
        }
    }
    None
}

fn capture_optional_command_candidate_with_timeout(
    program: &Path,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    let mut child = match Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            debug_command_capture_failure(program, "spawn", &error.to_string());
            return None;
        }
    };
    let mut stdout_reader = child.stdout.take().map(|mut stdout| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stdout.read_to_end(&mut bytes);
            bytes
        })
    });

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let bytes = stdout_reader
                    .take()
                    .map(|reader| reader.join().unwrap_or_default())
                    .unwrap_or_default();
                if status.success() {
                    return String::from_utf8(bytes).ok();
                }
                debug_command_capture_failure(program, "exit", &format!("status {status}"));
                return None;
            }
            Ok(None) if start.elapsed() < timeout => {
                thread::sleep(Duration::from_millis(25));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = stdout_reader.take() {
                    let _ = reader.join();
                }
                debug_command_capture_failure(program, "timeout", "timed out");
                return None;
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = stdout_reader.take() {
                    let _ = reader.join();
                }
                debug_command_capture_failure(program, "wait", "failed to wait");
                return None;
            }
        }
    }
}

fn debug_command_capture_failure(program: &Path, stage: &str, detail: &str) {
    if !crate::runtime::env_flag("ROCM_CLI_DEBUG_COMMAND_CAPTURE") {
        return;
    }
    eprintln!(
        "rocm debug: command capture {stage} failed for {}: {detail}",
        program.display()
    );
}

pub(super) fn tool_on_path(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|dir| {
            tool_path_candidates(program)
                .into_iter()
                .any(|name| dir.join(name).is_file())
        })
    })
}

pub(super) fn tool_path_candidates(program: &str) -> Vec<String> {
    let path = Path::new(program);
    if path.extension().is_some() || !runtime_is_windows() {
        return vec![program.to_owned()];
    }
    let mut names = Vec::new();
    names.push(program.to_owned());
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_owned());
    for ext in pathext
        .split(';')
        .map(str::trim)
        .filter(|ext| !ext.is_empty())
    {
        names.push(format!("{program}{ext}"));
        names.push(format!("{program}{}", ext.to_ascii_lowercase()));
    }
    names.extend(windows_absolute_tool_candidates(program));
    names.sort();
    names.dedup();
    names
}

fn windows_absolute_tool_candidates(program: &str) -> Vec<String> {
    if !runtime_is_windows() {
        return Vec::new();
    }
    let program = program.trim().to_ascii_lowercase();
    let system_root = std::env::var("SystemRoot")
        .or_else(|_| std::env::var("WINDIR"))
        .unwrap_or_else(|_| r"C:\Windows".to_owned());
    match program.as_str() {
        "cmd" | "cmd.exe" => vec![format!(r"{system_root}\System32\cmd.exe")],
        "pnputil" | "pnputil.exe" => vec![format!(r"{system_root}\System32\pnputil.exe")],
        "powershell" | "powershell.exe" => vec![
            format!(r"{system_root}\System32\WindowsPowerShell\v1.0\powershell.exe"),
            "powershell.exe".to_owned(),
        ],
        "pwsh" | "pwsh.exe" => vec!["pwsh.exe".to_owned()],
        _ => Vec::new(),
    }
}

/// Characterisation of the two PATH lookups the host probes carry: `tool_on_path`
/// here (the summary stack) and `which` in [`crate::probe_exec`] (the examination
/// stack). They answer the same question, "is this program installed", and agree
/// on a `PATH` of plain directories; the tests below pin that agreement and,
/// separately, each copy's own answer where they differ. This is the first pair
/// of the host-probing note (`docs/host-probing.md`, "PATH lookup").
///
/// Both lookups read the process's own `PATH` (and, on Windows, `PATHEXT`), and
/// one case needs a different working directory. Neither can be changed on a test
/// thread without disturbing every other test in the process, which spawn
/// children by bare name and call `which` themselves. So each case runs in a
/// child: this test binary again, started with the `PATH`, `PATHEXT` and working
/// directory the case needs, and asked to run only
/// [`path_lookup_child_reports_both_answers`], which prints what each lookup
/// returns for one program. In a normal run that test finds no program to ask
/// about and passes without doing anything.
#[cfg(test)]
mod tests {
    use super::tool_on_path;
    use std::ffi::{OsStr, OsString};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const PROGRAM_ENV: &str = "ROCM_CORE_TEST_PATH_LOOKUP_PROGRAM";
    const ANSWER_PREFIX: &str = "path-lookup-answers:";
    /// A name no host has installed, so only the planted file can answer for it.
    const PROGRAM: &str = "rocm-core-pathprobe-tool";

    #[derive(Debug, PartialEq, Eq)]
    struct Answers {
        tool_on_path: bool,
        which: bool,
    }

    const BOTH: Answers = Answers {
        tool_on_path: true,
        which: true,
    };
    const NEITHER: Answers = Answers {
        tool_on_path: false,
        which: false,
    };
    /// What the summary's lookup finds and the examination's does not.
    const ONLY_SUMMARY: Answers = Answers {
        tool_on_path: true,
        which: false,
    };

    /// The child's side of [`ask_for`]: print both answers for the program named
    /// in the environment. Does nothing in a normal test run.
    #[test]
    fn path_lookup_child_reports_both_answers() {
        let Some(program) = std::env::var_os(PROGRAM_ENV) else {
            return;
        };
        let program = program.to_string_lossy();
        println!(
            "{ANSWER_PREFIX} tool_on_path={} which={}",
            tool_on_path(&program),
            crate::probe_exec::which(&program)
        );
    }

    /// Run both lookups for `program` in a child whose `PATH` is `path` (unset
    /// when `None`), working directory is `cwd` and extra environment is `env`.
    fn ask_for(program: &str, path: Option<&OsStr>, cwd: &Path, env: &[(&str, &str)]) -> Answers {
        let this_test = module_path!()
            .split_once("::")
            .expect("module path has a crate prefix")
            .1;
        let mut child = Command::new(std::env::current_exe().expect("path of the test binary"));
        child
            .args([
                "--exact",
                &format!("{this_test}::path_lookup_child_reports_both_answers"),
                "--nocapture",
                "--test-threads=1",
            ])
            .env(PROGRAM_ENV, program)
            .current_dir(cwd);
        match path {
            Some(path) => child.env("PATH", path),
            None => child.env_remove("PATH"),
        };
        for (key, value) in env {
            child.env(key, value);
        }
        let output = child.output().expect("start the test binary again");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout
            .lines()
            .find_map(|line| line.split_once(ANSWER_PREFIX).map(|(_, answers)| answers))
            .unwrap_or_else(|| {
                panic!(
                    "the child printed no answers (status {}):\n{stdout}\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                )
            });
        let flag = |name: &str| line.contains(&format!("{name}=true"));
        Answers {
            tool_on_path: flag("tool_on_path"),
            which: flag("which"),
        }
    }

    fn ask(path: Option<&OsStr>, cwd: &Path) -> Answers {
        ask_for(PROGRAM, path, cwd, &[])
    }

    fn join(dirs: &[&Path]) -> OsString {
        std::env::join_paths(dirs).expect("no directory name holds a path separator")
    }

    /// Create `name` under `parent`, holding a file called `file_name`.
    fn dir_holding(parent: &Path, name: &OsStr, file_name: &str) -> PathBuf {
        let dir = parent.join(name);
        std::fs::create_dir_all(&dir).expect("create the planted directory");
        crate::fake_executable::write_fake_executable(&dir.join(file_name), "#!/bin/sh\n")
            .expect("plant the program");
        dir
    }

    fn empty_dir(parent: &Path, name: &str) -> PathBuf {
        let dir = parent.join(name);
        std::fs::create_dir_all(&dir).expect("create the empty directory");
        dir
    }

    #[test]
    fn both_lookups_find_a_program_in_a_path_directory_and_not_otherwise() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let present = dir_holding(scratch.path(), OsStr::new("present"), PROGRAM);
        let absent = empty_dir(scratch.path(), "absent");
        let cwd = empty_dir(scratch.path(), "cwd");

        assert_eq!(
            ask(Some(&join(&[&absent, &present])), &cwd),
            BOTH,
            "a program in the second PATH directory is found by both"
        );
        assert_eq!(
            ask(Some(&join(&[&absent])), &cwd),
            NEITHER,
            "a PATH without the program finds nothing in either"
        );
        assert_eq!(
            ask(None, &cwd),
            NEITHER,
            "an unset PATH finds nothing in either"
        );
    }

    #[test]
    fn a_directory_named_like_the_program_is_not_the_program() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let bin = empty_dir(scratch.path(), "bin");
        std::fs::create_dir_all(bin.join(PROGRAM)).expect("plant a directory with the name");
        assert_eq!(ask(Some(&join(&[&bin])), scratch.path()), NEITHER);
    }

    /// An empty `PATH` entry means the current directory to `std::env::split_paths`
    /// (and to a shell or `execvp`). The summary's lookup follows it; the
    /// examination's skips empty entries. A file in the current directory is
    /// found through an empty entry by the first and not by the second, and by
    /// neither when `PATH` has no empty entry.
    #[cfg(unix)]
    #[test]
    fn a_file_in_the_current_directory_is_found_through_an_empty_entry_by_the_summary_only() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let cwd = dir_holding(scratch.path(), OsStr::new("cwd"), PROGRAM);
        let bin = empty_dir(scratch.path(), "bin");
        let bin_text = bin.to_str().expect("UTF-8 scratch path");

        for (shape, path) in [
            ("leading", format!(":{bin_text}")),
            ("middle", format!("{bin_text}::{bin_text}")),
            ("trailing", format!("{bin_text}:")),
            ("the whole PATH", String::new()),
        ] {
            assert_eq!(
                ask(Some(OsStr::new(&path)), &cwd),
                ONLY_SUMMARY,
                "an empty entry ({shape}) reads the current directory in the summary only"
            );
        }
        assert_eq!(
            ask(Some(&join(&[&bin])), &cwd),
            NEITHER,
            "without an empty entry, a file in the current directory is not on PATH"
        );
    }

    /// A `PATH` that is not valid Unicode still lists directories. The summary's
    /// lookup splits it as an `OsString` and finds the program; the examination's
    /// reads it with `std::env::var`, which fails, and so finds nothing in any
    /// directory, valid ones included.
    #[cfg(unix)]
    #[test]
    fn a_path_that_is_not_unicode_is_searched_by_the_summary_only() {
        use std::os::unix::ffi::OsStrExt as _;

        let scratch = tempfile::tempdir().expect("scratch directory");
        let cwd = empty_dir(scratch.path(), "cwd");
        let valid = dir_holding(scratch.path(), OsStr::new("valid"), PROGRAM);
        // A directory whose own name is not Unicode (it holds the byte 0xff).
        let odd = dir_holding(scratch.path(), OsStr::from_bytes(b"odd-\xff"), PROGRAM);
        let missing_odd = scratch.path().join(OsStr::from_bytes(b"missing-\xff"));

        assert_eq!(
            ask(Some(&join(&[&odd])), &cwd),
            ONLY_SUMMARY,
            "the program sits in a directory whose name is not Unicode"
        );
        assert_eq!(
            ask(Some(&join(&[&missing_odd, &valid])), &cwd),
            ONLY_SUMMARY,
            "a Unicode directory after an entry that is not Unicode is still searched by the summary, and not by the examination"
        );
    }

    /// On Windows the summary's lookup tries every `PATHEXT` entry, in both
    /// cases, where the examination's tries `.exe`, `.bat`, `.cmd` and the bare
    /// name. A `.ps1` listed in `PATHEXT` is found by the first only.
    #[cfg(windows)]
    #[test]
    fn a_pathext_extension_outside_the_examinations_list_is_found_by_the_summary_only() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let ps1 = dir_holding(scratch.path(), OsStr::new("ps1"), &format!("{PROGRAM}.ps1"));
        let exe = dir_holding(scratch.path(), OsStr::new("exe"), &format!("{PROGRAM}.exe"));
        let pathext = [("PATHEXT", ".COM;.EXE;.BAT;.CMD;.PS1")];

        assert_eq!(
            ask_for(PROGRAM, Some(&join(&[&ps1])), scratch.path(), &pathext),
            ONLY_SUMMARY
        );
        assert_eq!(
            ask_for(PROGRAM, Some(&join(&[&exe])), scratch.path(), &pathext),
            BOTH
        );
    }

    /// `std::env::split_paths` on Windows keeps a quoted entry whole, including a
    /// `;` inside it; the examination's lookup splits on every `;` and so reads
    /// the quoted directory as two halves that are not directories.
    #[cfg(windows)]
    #[test]
    fn a_quoted_path_entry_holding_a_semicolon_is_searched_by_the_summary_only() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let dir = dir_holding(
            scratch.path(),
            OsStr::new("semi;colon"),
            &format!("{PROGRAM}.exe"),
        );
        let quoted = format!("\"{}\"", dir.display());
        assert_eq!(ask(Some(OsStr::new(&quoted)), scratch.path()), ONLY_SUMMARY);
    }

    /// The summary's candidate list names the fixed System32 location of `cmd`,
    /// and its lookup joins each candidate onto each `PATH` directory, so an
    /// absolute candidate is found wherever `PATH` points. The examination's
    /// lookup knows nothing of it.
    #[cfg(windows)]
    #[test]
    fn a_system32_tool_is_found_by_the_summary_without_being_on_path() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let empty = empty_dir(scratch.path(), "empty");
        assert_eq!(
            ask_for("cmd", Some(&join(&[&empty])), scratch.path(), &[]),
            ONLY_SUMMARY
        );
    }
}
