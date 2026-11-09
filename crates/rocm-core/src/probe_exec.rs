// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Subprocess and file-read helpers for the host probes.
//!
//! [`run`]/[`run_with_env`] run a command under a timeout and never panic, `which`
//! resolves a program on `PATH` without executing it, and [`read_text`]/
//! [`read_host_text`] read a file as a possibly-empty string. They sit apart from
//! [`crate::examine`] so `fix`, `process` and `host_gpu` can reach them without
//! importing the module that owns the `Examination` document.

use crate::runtime_is_windows;
use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Run a command with a timeout. Returns `(rc, stdout, stderr)`. `rc` is `127`
/// when the program can't be spawned and `124` on timeout.
pub(crate) fn run(program: &str, args: &[&str], timeout: Duration) -> (i32, String, String) {
    run_with_env(program, args, &[], timeout)
}

/// [`run`] with extra environment variables overlaid on the inherited ones.
pub(crate) fn run_with_env(
    program: &str,
    args: &[&str],
    envs: &[(String, OsString)],
    timeout: Duration,
) -> (i32, String, String) {
    let (rc, stdout, stderr) = run_raw(program, args, envs, timeout);
    (
        rc,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

/// [`run`] without the UTF-8 assumption, for output that is not UTF-8.
fn run_raw(
    program: &str,
    args: &[&str],
    envs: &[(String, OsString)],
    timeout: Duration,
) -> (i32, Vec<u8>, Vec<u8>) {
    let Ok(mut child) = Command::new(program)
        .args(args)
        .envs(envs.iter().map(|(key, value)| (key, value)))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    else {
        return (127, Vec::new(), Vec::new());
    };
    // Bytes, then a lossy conversion at the end. `read_to_string` FAILS on
    // invalid UTF-8 and the error was discarded, so a single stray byte silently
    // emptied the whole capture — which reads downstream as "the command printed
    // nothing", not as "the output could not be decoded".
    let stdout_handle = child.stdout.take().map(|mut stdout| {
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf);
            buf
        })
    });
    let stderr_handle = child.stderr.take().map(|mut stderr| {
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        })
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(_) => break None,
        }
    };
    let stdout: Vec<u8> = stdout_handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let stderr: Vec<u8> = stderr_handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let rc = match status {
        Some(status) => status.code().unwrap_or(-1),
        None => 124,
    };
    (rc, stdout, stderr)
}

/// Run a command whose output is UTF-16LE, as `wsl.exe`'s is.
///
/// Decoding is by declaration, not detection. Sniffing the encoding cannot work
/// here: UTF-16LE text in a Latin or Cyrillic script is made entirely of bytes
/// below 0x80, so it is *valid UTF-8* and decodes without error straight into
/// mojibake — no NUL-density or validity test can tell the two apart. The one
/// reliable fact is which program produced the bytes.
pub(crate) fn run_utf16le(program: &str, args: &[&str], timeout: Duration) -> (i32, String) {
    let (rc, stdout, _) = run_raw(program, args, &[], timeout);
    (rc, decode_utf16le(&stdout))
}

pub(crate) fn decode_utf16le(bytes: &[u8]) -> String {
    let body = bytes.strip_prefix(&[0xFF, 0xFE]).unwrap_or(bytes);
    // `chunks_exact` drops a trailing odd byte rather than panicking on it.
    let units: Vec<u16> = body
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    String::from_utf16_lossy(&units)
}

pub(crate) fn read_text(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

/// [`read_text`] for an absolute host path, read where [`crate::host_path`]
/// says the host is.
pub(crate) fn read_host_text(path: &str) -> String {
    std::fs::read_to_string(crate::host_path(path)).unwrap_or_default()
}

/// Whether `program` resolves on `PATH` (best-effort, no execution).
///
/// The one PATH lookup of the host probes: the examination stack's capability
/// checks and the summary stack's WSL facts both ask it, so a tool reads as
/// installed or not the same way in every part of `rocm examine`. It tries
/// each spelling [`tool_path_candidates`] gives, in each directory of `PATH`,
/// and answers `true` for the first one that is a file.
///
/// - An empty `PATH` entry is skipped, not read as the current directory as a
///   shell or `execvp` would. A file named like the tool in the directory the
///   CLI was started from therefore does not read as the tool being installed
///   through an empty entry, which matters for the checks that gate a
///   privileged step (`sudo`, `usermod`). An explicit `.` or other relative
///   entry is an ordinary directory and is searched. The cost is that a
///   program the CLI later starts by bare name can still resolve from the
///   current directory when `PATH` has an empty entry, because the operating
///   system resolves it, not this lookup.
/// - A `PATH` that is not valid Unicode is searched: it is split as an
///   `OsString`, and the entries that are not Unicode are looked in as they are.
/// - The directories are split the way [`std::env::split_paths`] splits them on
///   this platform, so on Windows an entry may be quoted to hold a `;`.
/// - On Windows a program without an extension is tried with every `PATHEXT`
///   entry in both cases, and `cmd`, `pnputil`, `powershell` and `pwsh` are
///   also found at their fixed System32 locations. An absolute candidate is
///   tested once as it is, not joined onto each `PATH` directory, so it is found
///   whether or not `PATH` is set, on `PATH` or not: the lookup then agrees with
///   what the capture runner, which starts those locations, can run. No caller
///   asks for those four today; they stay in the shared candidate list because
///   the runner needs them and one list cannot drift from the other.
pub(crate) fn which(program: &str) -> bool {
    let dirs = std::env::var_os("PATH")
        .map(|path| path_dirs(&path))
        .unwrap_or_default();
    found_in(&dirs, &tool_path_candidates(program), Path::is_file)
}

/// The directories of a `PATH` value, empty entries dropped.
fn path_dirs(path: &OsStr) -> Vec<PathBuf> {
    std::env::split_paths(path)
        .filter(|dir| !dir.as_os_str().is_empty())
        .collect()
}

/// Whether any of `candidates` is a file according to `is_file`.
///
/// A relative candidate is joined onto each of `dirs`: directories are tried in
/// order, and within a directory the candidates in order. An absolute candidate
/// names its own location, so it is tested once as it is, after the directories,
/// and does not depend on `dirs` at all. `is_file` is injected so a test can say
/// which files exist without a file system.
fn found_in(dirs: &[PathBuf], candidates: &[String], is_file: impl Fn(&Path) -> bool) -> bool {
    let in_a_dir = dirs.iter().any(|dir| {
        candidates
            .iter()
            .filter(|candidate| !Path::new(candidate).is_absolute())
            .any(|candidate| is_file(&dir.join(candidate)))
    });
    in_a_dir
        || candidates
            .iter()
            .map(Path::new)
            .filter(|candidate| candidate.is_absolute())
            .any(is_file)
}

/// The spellings of `program` to try, in the order to try them.
///
/// On Linux, just `program`. On Windows a program with no extension is also
/// spelled with each `PATHEXT` entry, in both cases, and `cmd`, `pnputil`,
/// `powershell` and `pwsh` add their fixed locations. Used by [`which`] and by
/// the capture runner, which starts each spelling in turn.
pub(crate) fn tool_path_candidates(program: &str) -> Vec<String> {
    path_candidates(
        program,
        runtime_is_windows(),
        std::env::var("PATHEXT").ok().as_deref(),
        &system_root_from_env(),
    )
}

/// `SystemRoot`, else `WINDIR`, else `C:\Windows`.
fn system_root_from_env() -> String {
    std::env::var("SystemRoot")
        .or_else(|_| std::env::var("WINDIR"))
        .unwrap_or_else(|_| r"C:\Windows".to_owned())
}

/// [`tool_path_candidates`] with the platform, `PATHEXT` (`None` when unset or
/// not Unicode) and system root passed in, so the Windows spellings can be
/// checked on every host.
fn path_candidates(
    program: &str,
    windows: bool,
    pathext: Option<&str>,
    system_root: &str,
) -> Vec<String> {
    if Path::new(program).extension().is_some() || !windows {
        return vec![program.to_owned()];
    }
    let mut names = vec![program.to_owned()];
    let pathext = pathext.unwrap_or(".COM;.EXE;.BAT;.CMD");
    for ext in pathext
        .split(';')
        .map(str::trim)
        .filter(|ext| !ext.is_empty())
    {
        names.push(format!("{program}{ext}"));
        names.push(format!("{program}{}", ext.to_ascii_lowercase()));
    }
    names.extend(windows_absolute_tool_candidates(program, system_root));
    names.sort();
    names.dedup();
    names
}

/// The fixed locations of the Windows tools the host probes run.
fn windows_absolute_tool_candidates(program: &str, system_root: &str) -> Vec<String> {
    let program = program.trim().to_ascii_lowercase();
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

pub(crate) const SHORT: Duration = Duration::from_secs(5);
pub(crate) const MEDIUM: Duration = Duration::from_secs(8);

/// Truncate `value` to at most `max_chars` characters, appending a marker when
/// truncated. Slices on char boundaries (matching Python's `value[:n]`); a byte
/// slice would panic when the cut lands inside a multibyte character.
///
/// `pub(crate)` because two other places bound a quote with the same marker and
/// a second copy here would be free to drift from them: [`crate::process`] for
/// its timeout quote, and [`crate::amd_smi`] for the `amd-smi` failure quote,
/// which bounds raw child output for the same reason the examination bounds
/// captured text. Those two also share one character budget, declared once in
/// [`crate::process`].
/// (`host_gpu/diagnostics.rs`'s `truncate_diagnostic_line` is a different function with a
/// different marker — plain `...` — not a third copy of this one.)
pub(crate) fn truncate_to_chars(value: String, max_chars: usize) -> String {
    if value.chars().count() > max_chars {
        let truncated: String = value.chars().take(max_chars).collect();
        format!("{truncated}...[truncated]")
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_output_that_is_not_utf8_is_kept_rather_than_dropped() {
        // `read_to_string` fails on invalid UTF-8, and the error was discarded --
        // so one stray byte emptied a whole capture, which every caller then read
        // as "the command printed nothing".
        let (rc, out, _) = run("printf", &["ok\\xffdone"], SHORT);
        if rc == 127 {
            return; // no `printf` binary on this host
        }
        assert!(
            out.starts_with("ok") && out.ends_with("done"),
            "the undecodable byte must not take the rest of the output with it: {out:?}"
        );
    }

    #[test]
    fn env_truncation_is_char_safe_across_multibyte_boundary() {
        // 5000 'é' chars = 10000 bytes; byte 4000 lands inside a char, which a
        // byte slice would panic on. Must truncate cleanly to 4000 chars.
        let value = "é".repeat(5000);
        let out = truncate_to_chars(value, 4000);
        assert!(out.ends_with("...[truncated]"));
        assert_eq!(out.chars().filter(|&c| c == 'é').count(), 4000);
    }

    #[test]
    fn env_truncation_leaves_short_values_untouched() {
        assert_eq!(truncate_to_chars("short".to_owned(), 4000), "short");
    }
}

/// The PATH lookup ([`which`]) and the spellings it tries
/// ([`tool_path_candidates`]), which the summary stack and the examination stack
/// share. Before they were one, each stack had its own copy and the copies
/// answered differently for an empty `PATH` entry, a `PATH` that is not Unicode
/// and the Windows spellings (see `docs/host-probing.md`, "PATH lookup"); the
/// tests here pin the one answer they now give.
///
/// The pure functions are tested on every host with the platform, `PATHEXT` and
/// the files that exist passed in. The lookup as a whole reads the process's own
/// `PATH`, and one case needs a different working directory, which a test thread
/// cannot change without disturbing every other test in the process (they spawn
/// children by bare name and call `which` themselves). So those cases run in a
/// child: this test binary again, started with the `PATH`, `PATHEXT` and working
/// directory the case needs and asked to run only
/// [`which_child_reports_its_answer`]. In a normal run that test finds no
/// program to ask about and passes without doing anything.
#[cfg(test)]
mod path_lookup_tests {
    use super::{found_in, path_candidates, path_dirs, which};
    use std::cell::RefCell;
    use std::ffi::{OsStr, OsString};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const PROGRAM_ENV: &str = "ROCM_CORE_TEST_WHICH_PROGRAM";
    const ANSWER_PREFIX: &str = "which-answer:";
    /// A name no host has installed, so only the planted file can answer for it.
    const PROGRAM: &str = "rocm-core-which-tool";

    fn names(candidates: &[&str]) -> Vec<String> {
        candidates.iter().map(|name| (*name).to_owned()).collect()
    }

    #[test]
    fn candidates_off_windows_are_the_program_alone_whatever_pathext_says() {
        assert_eq!(
            path_candidates("tool", false, Some(".COM;.EXE"), r"C:\Windows"),
            names(&["tool"])
        );
    }

    #[test]
    fn candidates_on_windows_for_a_name_with_an_extension_are_the_name_alone() {
        // The early return also keeps the System32 spelling of `cmd.exe` out of
        // the list: only the extension-less `cmd` reaches it.
        assert_eq!(
            path_candidates("cmd.exe", true, Some(".COM;.EXE"), r"C:\Windows"),
            names(&["cmd.exe"])
        );
        assert_eq!(
            path_candidates("tool.ps1", true, Some(".PS1"), r"C:\Windows"),
            names(&["tool.ps1"])
        );
    }

    #[test]
    fn candidates_on_windows_try_every_pathext_entry_in_both_cases() {
        // Sorted as bytes, so the upper-case spelling of an extension comes
        // before its lower-case one, and the bare name before both. `.PS1` is
        // outside the old examination list (`.exe`, `.bat`, `.cmd`).
        assert_eq!(
            path_candidates("tool", true, Some(".COM;.EXE;.PS1"), r"C:\Windows"),
            names(&[
                "tool", "tool.COM", "tool.EXE", "tool.PS1", "tool.com", "tool.exe", "tool.ps1",
            ])
        );
    }

    #[test]
    fn candidates_on_windows_trim_pathext_entries_and_skip_empty_ones() {
        assert_eq!(
            path_candidates("tool", true, Some(" .BAT ;; ;.CMD;"), r"C:\Windows"),
            names(&["tool", "tool.BAT", "tool.CMD", "tool.bat", "tool.cmd"])
        );
    }

    #[test]
    fn candidates_on_windows_fall_back_to_the_default_pathext_when_it_is_unset() {
        assert_eq!(
            path_candidates("tool", true, None, r"C:\Windows"),
            names(&[
                "tool", "tool.BAT", "tool.CMD", "tool.COM", "tool.EXE", "tool.bat", "tool.cmd",
                "tool.com", "tool.exe",
            ])
        );
    }

    #[test]
    fn candidates_on_windows_add_the_fixed_system32_locations() {
        let root = r"C:\Win";
        let with = |program: &str| path_candidates(program, true, Some(".EXE"), root);

        assert!(
            with("cmd").contains(&r"C:\Win\System32\cmd.exe".to_owned()),
            "{:?}",
            with("cmd")
        );
        assert!(with("pnputil").contains(&r"C:\Win\System32\pnputil.exe".to_owned()));
        let powershell = with("powershell");
        assert!(
            powershell
                .contains(&r"C:\Win\System32\WindowsPowerShell\v1.0\powershell.exe".to_owned())
        );
        assert!(powershell.contains(&"powershell.exe".to_owned()));
        assert!(with("pwsh").contains(&"pwsh.exe".to_owned()));
        // Matched without regard to case, like the file system.
        assert!(with("CMD").contains(&r"C:\Win\System32\cmd.exe".to_owned()));
        // The location comes first when it sorts first, which is the order the
        // capture runner starts them in.
        assert_eq!(
            with("cmd").first().map(String::as_str),
            Some(r"C:\Win\System32\cmd.exe")
        );
    }

    #[test]
    fn candidates_on_windows_for_other_tools_have_no_fixed_location() {
        assert_eq!(
            path_candidates("python", true, Some(".EXE"), r"C:\Win"),
            names(&["python", "python.EXE", "python.exe"])
        );
    }

    #[test]
    fn candidates_off_windows_never_name_a_system32_location() {
        assert_eq!(
            path_candidates("cmd", false, Some(".EXE"), r"C:\Win"),
            names(&["cmd"])
        );
    }

    #[test]
    fn path_dirs_drop_every_empty_entry() {
        let path = std::env::join_paths([
            OsStr::new(""),
            OsStr::new("first"),
            OsStr::new(""),
            OsStr::new("second"),
            OsStr::new(""),
        ])
        .expect("no entry holds a separator");
        assert_eq!(
            path_dirs(&path),
            vec![PathBuf::from("first"), PathBuf::from("second")]
        );
        assert!(path_dirs(OsStr::new("")).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn path_dirs_keep_an_entry_that_is_not_unicode() {
        use std::os::unix::ffi::OsStrExt as _;

        let odd = OsStr::from_bytes(b"/odd-\xff");
        let path = std::env::join_paths([OsStr::new("/usr/bin"), odd]).expect("no separator");
        assert_eq!(
            path_dirs(&path),
            vec![PathBuf::from("/usr/bin"), PathBuf::from(odd)]
        );
    }

    #[test]
    fn found_in_asks_each_candidate_in_each_directory_in_order_and_stops_at_the_first_file() {
        let dirs = [PathBuf::from("first"), PathBuf::from("second")];
        let candidates = names(&["tool", "tool.exe"]);
        let asked = RefCell::new(Vec::new());

        let found = found_in(&dirs, &candidates, |file| {
            asked.borrow_mut().push(file.to_owned());
            file == Path::new("second").join("tool")
        });

        assert!(found);
        assert_eq!(
            asked.into_inner(),
            vec![
                Path::new("first").join("tool"),
                Path::new("first").join("tool.exe"),
                Path::new("second").join("tool"),
            ]
        );
        assert!(!found_in(&dirs, &candidates, |_| false));
        assert!(!found_in(&[], &candidates, |_| true));
    }

    #[test]
    fn found_in_asks_about_an_absolute_candidate_once_as_it_is_whatever_the_directories() {
        // This is why a System32 location is found "without being on PATH", and
        // found even when PATH is unset or all empty: it is tested on its own,
        // once, not joined onto each directory.
        let absolute = std::env::temp_dir().join("rocm-core-absolute-candidate");
        let candidates = [absolute.to_string_lossy().into_owned()];

        for dirs in [
            Vec::new(),
            vec![PathBuf::from("anywhere")],
            vec![PathBuf::from("anywhere"), PathBuf::from("elsewhere")],
        ] {
            let asked = RefCell::new(Vec::new());
            let found = found_in(&dirs, &candidates, |file| {
                asked.borrow_mut().push(file.to_owned());
                true
            });
            // With directories listed the absolute candidate is never joined
            // onto them, so the only thing asked about is the candidate itself.
            assert!(found, "{} directories", dirs.len());
            assert_eq!(
                asked.into_inner(),
                vec![absolute.clone()],
                "{} directories",
                dirs.len()
            );
        }
        assert!(!found_in(&[], &candidates, |_| false));
    }

    #[test]
    fn found_in_still_tries_relative_candidates_in_directories_beside_an_absolute_one() {
        let absolute = std::env::temp_dir().join("rocm-core-absolute-candidate");
        let candidates = [
            absolute.to_string_lossy().into_owned(),
            "tool.exe".to_owned(),
        ];
        let dirs = [PathBuf::from("first")];

        assert!(found_in(&dirs, &candidates, |file| file
            == Path::new("first").join("tool.exe")));
        assert!(!found_in(&[], &candidates, |file| file
            == Path::new("first").join("tool.exe")));
    }

    /// The child's side of [`ask_for`]: print the answer for the program named in
    /// the environment. Does nothing in a normal test run.
    #[test]
    fn which_child_reports_its_answer() {
        let Some(program) = std::env::var_os(PROGRAM_ENV) else {
            return;
        };
        println!("{ANSWER_PREFIX} {}", which(&program.to_string_lossy()));
    }

    /// Run [`which`] for `program` in a child whose `PATH` is `path` (unset when
    /// `None`), working directory is `cwd` and extra environment is `env`.
    fn ask_for(program: &str, path: Option<&OsStr>, cwd: &Path, env: &[(&str, &str)]) -> bool {
        let this_test = module_path!()
            .split_once("::")
            .expect("module path has a crate prefix")
            .1;
        let mut child = Command::new(std::env::current_exe().expect("path of the test binary"));
        child
            .args([
                "--exact",
                &format!("{this_test}::which_child_reports_its_answer"),
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
        let answer = stdout
            .lines()
            .find_map(|line| line.split_once(ANSWER_PREFIX).map(|(_, answer)| answer))
            .unwrap_or_else(|| {
                panic!(
                    "the child printed no answer (status {}):\n{stdout}\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                )
            });
        answer.trim() == "true"
    }

    fn ask(path: Option<&OsStr>, cwd: &Path) -> bool {
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
    fn which_finds_a_program_in_a_path_directory_and_not_otherwise() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let present = dir_holding(scratch.path(), OsStr::new("present"), PROGRAM);
        let absent = empty_dir(scratch.path(), "absent");
        let cwd = empty_dir(scratch.path(), "cwd");

        assert!(
            ask(Some(&join(&[&absent, &present])), &cwd),
            "a program in the second PATH directory is found"
        );
        assert!(
            !ask(Some(&join(&[&absent])), &cwd),
            "a PATH without the program finds nothing"
        );
        assert!(!ask(None, &cwd), "an unset PATH finds nothing");
    }

    #[test]
    fn which_does_not_take_a_directory_named_like_the_program_for_the_program() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let bin = empty_dir(scratch.path(), "bin");
        std::fs::create_dir_all(bin.join(PROGRAM)).expect("plant a directory with the name");
        assert!(!ask(Some(&join(&[&bin])), scratch.path()));
    }

    /// The decision this lookup was merged on: an empty `PATH` entry is skipped,
    /// so a file in the directory the CLI runs from is not "installed". The
    /// summary stack's old copy read the entry as the current directory and
    /// answered `true` here; the control below shows the same file does count
    /// once a real entry names its directory.
    #[cfg(unix)]
    #[test]
    fn which_does_not_read_an_empty_path_entry_as_the_current_directory() {
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
            assert!(
                !ask(Some(OsStr::new(&path)), &cwd),
                "an empty entry ({shape}) must not read the current directory"
            );
        }
        assert!(
            ask(Some(&join(&[&cwd])), &cwd),
            "the same file is found when a PATH entry names its directory"
        );
    }

    /// The other half of the decision: a `PATH` that is not valid Unicode is
    /// still searched. The examination stack's old copy read `PATH` with
    /// `std::env::var`, which fails on it, and answered `false` for every tool.
    #[cfg(unix)]
    #[test]
    fn which_searches_a_path_that_is_not_unicode() {
        use std::os::unix::ffi::OsStrExt as _;

        let scratch = tempfile::tempdir().expect("scratch directory");
        let cwd = empty_dir(scratch.path(), "cwd");
        let valid = dir_holding(scratch.path(), OsStr::new("valid"), PROGRAM);
        // A directory whose own name is not Unicode (it holds the byte 0xff).
        let odd = dir_holding(scratch.path(), OsStr::from_bytes(b"odd-\xff"), PROGRAM);
        let missing_odd = scratch.path().join(OsStr::from_bytes(b"missing-\xff"));

        assert!(
            ask(Some(&join(&[&odd])), &cwd),
            "the program sits in a directory whose name is not Unicode"
        );
        assert!(
            ask(Some(&join(&[&missing_odd, &valid])), &cwd),
            "a Unicode directory after an entry that is not Unicode is still searched"
        );
        assert!(
            !ask(Some(&join(&[&missing_odd])), &cwd),
            "a PATH that is not Unicode does not find a program that is not on it"
        );
    }

    /// A `.ps1` listed in `PATHEXT` is found, where the examination stack's old
    /// copy tried only `.exe`, `.bat`, `.cmd` and the bare name.
    #[cfg(windows)]
    #[test]
    fn which_tries_a_pathext_extension_outside_exe_bat_and_cmd() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let ps1 = dir_holding(scratch.path(), OsStr::new("ps1"), &format!("{PROGRAM}.ps1"));
        let pathext = [("PATHEXT", ".COM;.EXE;.BAT;.CMD;.PS1")];
        assert!(ask_for(
            PROGRAM,
            Some(&join(&[&ps1])),
            scratch.path(),
            &pathext
        ));
    }

    /// An entry in quotes keeps a `;` inside it, as `std::env::split_paths`
    /// reads it on Windows; the examination stack's old copy split on every `;`.
    #[cfg(windows)]
    #[test]
    fn which_searches_a_quoted_path_entry_that_holds_a_semicolon() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let dir = dir_holding(
            scratch.path(),
            OsStr::new("semi;colon"),
            &format!("{PROGRAM}.exe"),
        );
        let quoted = format!("\"{}\"", dir.display());
        assert!(ask(Some(OsStr::new(&quoted)), scratch.path()));
    }

    /// `cmd` is found at its System32 location without being on `PATH`, as the
    /// summary stack's old copy found it; the examination stack's did not.
    #[cfg(windows)]
    #[test]
    fn which_finds_cmd_at_its_system32_location_without_it_being_on_path() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let empty = empty_dir(scratch.path(), "empty");
        assert!(ask_for("cmd", Some(&join(&[&empty])), scratch.path(), &[]));
        // And with no PATH at all, as the capture runner could still start it.
        assert!(ask_for("cmd", None, scratch.path(), &[]));
    }
}
