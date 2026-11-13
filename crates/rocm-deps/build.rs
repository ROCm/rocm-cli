// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Turn the versions locked in `runtimes/mise.lock` into Rust constants,
//! written to `$OUT_DIR/pins.rs`, which `src/lib.rs` includes. The generator
//! itself lives in `src/pins_gen.rs` so the crate's tests can exercise it.
//! Generating at build time (rather than committing a generated source file)
//! leaves no second copy of the pin in non-test source to drift. A lock this
//! generator cannot read — wrong format, a missing or unsupported platform, a
//! non-https URL, a malformed digest, an unsafe tool name — is a build error;
//! where a download comes from is checked by `cargo xtask deps check`, which
//! also runs this generator (`xtask/src/deps/mod.rs` includes it by path).

use std::path::PathBuf;
use std::{env, fs};

#[path = "src/pins_gen.rs"]
mod pins_gen;

/// Lock file, relative to this crate's manifest directory.
const LOCK_FILE: &str = "../../runtimes/mise.lock";

fn main() {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo"),
    );
    let lock_path = manifest_dir.join(LOCK_FILE);
    println!("cargo:rerun-if-changed={}", lock_path.display());
    println!("cargo:rerun-if-changed=src/pins_gen.rs");

    let text = fs::read_to_string(&lock_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", lock_path.display()));
    let generated =
        pins_gen::render(&text).unwrap_or_else(|err| panic!("{}: {err}", lock_path.display()));

    let out_path =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo")).join("pins.rs");
    fs::write(&out_path, generated)
        .unwrap_or_else(|err| panic!("failed to write {}: {err}", out_path.display()));
}
