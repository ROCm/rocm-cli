// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The shell script a scenario plants as a stand-in `stat`.
//!
//! The CLI learns a device node's mode and owning group by running
//! `stat -c %A|%U|%G <path>`. A device whose owning group the machine cannot name
//! (a container, where `stat` prints the literal `UNKNOWN`) is a property of the
//! machine a test cannot arrange, so the stand-in answers for ONE path and hands
//! every other call to the real `stat`.
//!
//! Both the format and the path have to match before it answers: keying on the
//! path alone would hand the `mode|owner|group` line to any other `stat` of the
//! same path, in whatever format it asked for, and the caller would parse the
//! wrong shape. The values are quoted for the shell, because the path lives
//! under a scenario's isolated root, whose name the harness does not control.

use std::path::Path;

use crate::shell_quote::single_quoted;

/// The format argument the CLI's device probe passes.
pub const DEVICE_STAT_FORMAT: &str = "%A|%U|%G";

/// The text of the stand-in `stat`: for a call that asks `device` for
/// [`DEVICE_STAT_FORMAT`], print `mode|owner|group` and succeed; for anything
/// else, run `real_stat` with the same arguments.
#[must_use]
pub fn stand_in_stat_script(
    device: &Path,
    mode: &str,
    owner: &str,
    group: &str,
    real_stat: &Path,
) -> String {
    let device = single_quoted(&device.to_string_lossy());
    let answer = single_quoted(&format!("{mode}|{owner}|{group}"));
    let real_stat = single_quoted(&real_stat.to_string_lossy());
    format!(
        "#!/bin/sh\n\
         fmt=\n\
         want=\n\
         for arg in \"$@\"; do\n\
         \x20 if [ \"$arg\" = '{DEVICE_STAT_FORMAT}' ]; then fmt=1; fi\n\
         \x20 if [ \"$arg\" = {device} ]; then want=1; fi\n\
         done\n\
         if [ -n \"$fmt\" ] && [ -n \"$want\" ]; then\n\
         \x20 printf '%s\n' {answer}\n\
         \x20 exit 0\n\
         fi\n\
         exec {real_stat} \"$@\"\n"
    )
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::Command;

    const REAL_STAT: &str = "/usr/bin/stat";

    fn plant(dir: &Path, device: &Path, group: &str) -> std::path::PathBuf {
        let script = dir.join("stat");
        rocm_core::fake_executable::write_fake_executable(
            &script,
            stand_in_stat_script(device, "crw-rw----", "root", group, Path::new(REAL_STAT)),
        )
        .expect("plant the stand-in");
        script
    }

    fn run(script: &Path, args: &[&std::ffi::OsStr]) -> (String, bool) {
        let output = Command::new(script).args(args).output().expect("run it");
        (
            String::from_utf8_lossy(&output.stdout).into_owned(),
            output.status.success(),
        )
    }

    #[test]
    fn it_answers_the_device_probe_for_the_named_path() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let device = scratch.path().join("kfd");
        std::fs::write(&device, "").expect("plant the device file");
        let script = plant(scratch.path(), &device, "UNKNOWN");
        let (out, ok) = run(
            &script,
            &[
                "-c".as_ref(),
                DEVICE_STAT_FORMAT.as_ref(),
                device.as_os_str(),
            ],
        );
        assert!(ok);
        assert_eq!(out, "crw-rw----|root|UNKNOWN\n");
    }

    #[test]
    fn a_quote_and_a_space_in_the_path_or_the_group_survive_the_shell() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let dir = scratch.path().join("it's a dir");
        std::fs::create_dir_all(&dir).expect("create the awkward directory");
        let device = dir.join("kfd node");
        std::fs::write(&device, "").expect("plant the device file");
        let script = plant(&dir, &device, "it's a group; echo injected");
        let (out, ok) = run(
            &script,
            &[
                "-c".as_ref(),
                DEVICE_STAT_FORMAT.as_ref(),
                device.as_os_str(),
            ],
        );
        assert!(ok);
        assert_eq!(out, "crw-rw----|root|it's a group; echo injected\n");
    }

    #[test]
    fn another_format_for_the_same_path_reaches_the_real_stat() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let device = scratch.path().join("kfd");
        std::fs::write(&device, "").expect("plant the device file");
        let script = plant(scratch.path(), &device, "UNKNOWN");
        let (out, ok) = run(&script, &["-c".as_ref(), "%F".as_ref(), device.as_os_str()]);
        assert!(ok);
        assert_eq!(out.trim(), "regular empty file");
    }

    #[test]
    fn the_device_format_for_another_path_reaches_the_real_stat() {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let device = scratch.path().join("kfd");
        let other = scratch.path().join("other");
        std::fs::write(&device, "").expect("plant the device file");
        std::fs::write(&other, "").expect("plant another file");
        let script = plant(scratch.path(), &device, "UNKNOWN");
        let (out, ok) = run(
            &script,
            &[
                "-c".as_ref(),
                DEVICE_STAT_FORMAT.as_ref(),
                other.as_os_str(),
            ],
        );
        assert!(ok);
        assert!(!out.contains("UNKNOWN"), "{out}");
    }
}

/// The script text itself, asserted on every host: the behaviour tests above
/// need a shell and run on unix only.
#[cfg(test)]
mod script_text {
    use super::*;

    #[test]
    fn every_value_is_quoted_for_the_shell() {
        let script = stand_in_stat_script(
            Path::new("/tmp/it's/kfd"),
            "crw-rw----",
            "root",
            "it's",
            Path::new("/usr/bin/real stat"),
        );
        assert!(script.contains(r"'/tmp/it'\''s/kfd'"), "{script}");
        assert!(script.contains(r"'crw-rw----|root|it'\''s'"), "{script}");
        assert!(
            script.contains("exec '/usr/bin/real stat' \"$@\""),
            "{script}"
        );
    }

    #[test]
    fn it_answers_only_when_both_the_format_and_the_path_match() {
        let script = stand_in_stat_script(
            Path::new("/dev/kfd"),
            "crw-rw----",
            "root",
            "render",
            Path::new("/usr/bin/stat"),
        );
        assert!(
            script.contains(r#"if [ -n "$fmt" ] && [ -n "$want" ]; then"#),
            "{script}"
        );
        assert!(script.contains(DEVICE_STAT_FORMAT), "{script}");
    }
}
