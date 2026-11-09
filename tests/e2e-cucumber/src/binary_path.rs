// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! How the harness names the `rocm` binary when it starts the child somewhere
//! else.
//!
//! `ROCM_CLI_BINARY` is either a path (`target/debug/rocm`, `./rocm`, an
//! absolute one) or a bare name (`rocm`) the operating system finds on `PATH`.
//! A scenario that moves the child to another working directory must pin a
//! relative path first, because it was written relative to the harness's own
//! directory. A bare name must be left alone: canonicalising it would find a
//! file called `rocm` in the harness's directory, if there is one, and run that
//! instead of the program `PATH` resolves.

use std::io;
use std::path::PathBuf;

/// Whether `binary` names a path rather than a bare program name: it is
/// absolute, or contains a separator (so `./rocm`, `bin/rocm` and `../x/rocm`
/// are paths, and `rocm` is not).
#[must_use]
pub fn names_a_path(binary: &str) -> bool {
    std::path::Path::new(binary).is_absolute() || binary.contains(['/', std::path::MAIN_SEPARATOR])
}

/// The program to start for `binary` when the child runs from another directory.
///
/// A path is made absolute with `canonicalize` (and kept as written when that
/// fails); a bare name is returned unchanged without asking.
pub fn program_for_moved_child(
    binary: &str,
    canonicalize: impl Fn(&str) -> io::Result<PathBuf>,
) -> PathBuf {
    if names_a_path(binary) {
        canonicalize(binary).unwrap_or_else(|_| PathBuf::from(binary))
    } else {
        PathBuf::from(binary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn paths_are_told_from_bare_names() {
        let abs = std::env::temp_dir().join("rocm");
        for (binary, is_path) in [
            ("rocm", false),
            ("rocm.exe", false),
            ("./rocm", true),
            ("bin/rocm", true),
            ("../x/rocm", true),
            (abs.to_str().expect("UTF-8 temp dir"), true),
        ] {
            assert_eq!(names_a_path(binary), is_path, "{binary}");
        }
    }

    #[test]
    fn a_path_is_canonicalised_and_a_bare_name_is_not_even_when_a_local_file_would_match() {
        // A canonicalise that succeeds for everything stands for a `rocm` file
        // existing in the harness's directory.
        let asked = RefCell::new(Vec::new());
        let pin = |binary: &str| {
            asked.borrow_mut().push(binary.to_owned());
            Ok(PathBuf::from("/pinned").join(binary))
        };

        assert_eq!(
            program_for_moved_child("rocm", pin),
            PathBuf::from("rocm"),
            "a bare name stays for PATH to resolve"
        );
        assert!(
            asked.borrow().is_empty(),
            "a bare name must not be looked up on disk"
        );
        for binary in ["./rocm", "bin/rocm", "../x/rocm"] {
            assert_eq!(
                program_for_moved_child(binary, pin),
                PathBuf::from("/pinned").join(binary),
                "{binary}"
            );
        }
        assert_eq!(asked.borrow().len(), 3);
    }

    #[test]
    fn a_path_that_cannot_be_canonicalised_is_kept_as_written() {
        assert_eq!(
            program_for_moved_child("./gone/rocm", |_| Err(io::Error::from(
                io::ErrorKind::NotFound
            ))),
            PathBuf::from("./gone/rocm")
        );
    }
}
