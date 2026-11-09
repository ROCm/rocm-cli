// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The runners of `rocm fix`: one per auto-applicable fix.
//!
//! A runner takes the caller's [`FixOptions`] (the Linux override runner takes
//! none: it only reports), prints the exact change, honors `--dry-run`, asks
//! for consent through [`confirm`] and returns an [`exit`] code. The `_linux`
//! and `_windows` variants are plain functions that every platform compiles;
//! the catalog in `fix/catalog.rs` names the entry point of each recipe's
//! runner and the running OS decides which variant runs.

use super::consent::{QUERY_TIMEOUT, RUN_TIMEOUT, confirm, fail, relay_output};
use super::options::{FixOptions, exit};
use crate::probe_exec::{run, which};
use crate::quoting::ps_single_quote_escape;
#[cfg(test)]
use crate::rocm_install::{
    RocmLayout, discover_rocm_installs_in_layout, discover_rocm_installs_on_host_in,
};
use crate::runtime_is_windows;
use std::io::Write;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Environment helpers
// ---------------------------------------------------------------------------

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn is_root() -> bool {
    run("id", &["-u"], QUERY_TIMEOUT).1.trim() == "0"
}

/// Pick the shell rc file to append to (.zshrc for zsh, else .bashrc).
fn shell_rc_file() -> Option<PathBuf> {
    let home = home_dir()?;
    let shell = std::env::var("SHELL").unwrap_or_default();
    let primary = if shell.contains("zsh") {
        home.join(".zshrc")
    } else {
        home.join(".bashrc")
    };
    if !primary.exists() && home.join(".bashrc").exists() {
        Some(home.join(".bashrc"))
    } else {
        Some(primary)
    }
}

fn append_line(path: &Path, header: &str, line: &str) -> std::io::Result<()> {
    use std::fs::OpenOptions;
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "\n{header}")?;
    writeln!(file, "{line}")
}

// ---------------------------------------------------------------------------
// Runners (one per auto-applicable fix)
// ---------------------------------------------------------------------------

/// fix-4: add the current user to the render group (and 'video' for safety).
pub(super) fn run_render_group(opts: &FixOptions) -> i32 {
    let user = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_default();
    if user.is_empty() {
        fail!("Could not determine current user from $USER/$LOGNAME.");
        return exit::NOT_APPLICABLE;
    }
    if !which("usermod") {
        fail!("`usermod` not on PATH; cannot add groups.");
        return exit::NOT_APPLICABLE;
    }
    let root = is_root();
    if !which("sudo") && !root {
        fail!("`sudo` is not on PATH and we are not root; cannot add groups.");
        return exit::NOT_APPLICABLE;
    }
    let (program, args): (&str, Vec<String>) = if root {
        (
            "usermod",
            vec![
                "-a".into(),
                "-G".into(),
                "render,video".into(),
                user.clone(),
            ],
        )
    } else {
        (
            "sudo",
            vec![
                "usermod".into(),
                "-a".into(),
                "-G".into(),
                "render,video".into(),
                user.clone(),
            ],
        )
    };
    println!("Will run: {program} {}", args.join(" "));
    if opts.dry_run {
        println!("(dry-run; not executed)");
        return exit::OK;
    }
    if !confirm("Add user to render,video groups?", opts.yes) {
        return exit::DECLINED;
    }
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let (rc, out, err) = run(program, &arg_refs, RUN_TIMEOUT);
    relay_output(&out, &err);
    if rc != 0 {
        fail!("usermod exited {rc}; group membership NOT changed.");
        return exit::FAILED;
    }
    println!("Added {user} to render,video.");
    println!(
        "IMPORTANT: log out and back in (or reboot) for the membership to take effect in new shells and services. `newgrp render` patches the current shell only."
    );
    exit::OK
}

/// fix-2: help the user clear HSA_OVERRIDE_GFX_VERSION for future shells.
pub(super) fn run_unset_override(opts: &FixOptions) -> i32 {
    if runtime_is_windows() {
        run_unset_override_windows(opts)
    } else {
        run_unset_override_linux()
    }
}

fn run_unset_override_linux() -> i32 {
    let current = std::env::var("HSA_OVERRIDE_GFX_VERSION").unwrap_or_default();
    if current.is_empty() {
        println!("HSA_OVERRIDE_GFX_VERSION is already unset in this shell.");
    } else {
        println!("HSA_OVERRIDE_GFX_VERSION={current} is set in this shell.");
        println!("In your current shell, run:");
        println!("  unset HSA_OVERRIDE_GFX_VERSION");
        println!("(This command can't unset it in your parent shell; it only sees a copy.)");
    }
    let Some(home) = home_dir() else {
        return exit::OK;
    };
    report_persistent_override(&[
        home.join(".bashrc"),
        home.join(".bash_profile"),
        home.join(".zshrc"),
        home.join(".profile"),
        home.join(".config").join("fish").join("config.fish"),
    ])
}

/// Report which of `candidates` persist `HSA_OVERRIDE_GFX_VERSION`, and leave
/// every one of them exactly as it was.
///
/// Takes the paths rather than deriving them from `$HOME`, for the same reason
/// [`pin_device_in_rc_file`] takes its rc path: it makes the promise testable
/// without mutating a process-global. And the promise needs testing — fix-2 is
/// flagged `auto_applicable`, so `skills/rocm-doctor/` has to say plainly that
/// this arm still only reports, and something has to hold that true.
fn report_persistent_override(candidates: &[PathBuf]) -> i32 {
    let hits: Vec<&PathBuf> = candidates
        .iter()
        .filter(|f| {
            std::fs::read_to_string(f).is_ok_and(|b| b.contains("HSA_OVERRIDE_GFX_VERSION"))
        })
        .collect();
    if hits.is_empty() {
        println!("\nNo persistent HSA_OVERRIDE_GFX_VERSION found in your shell rc files.");
        return exit::OK;
    }
    println!("\nPersistent HSA_OVERRIDE_GFX_VERSION found in:");
    for f in &hits {
        println!("  - {}", f.display());
    }
    println!(
        "\nRemove or comment those lines manually. This command does NOT edit your shell rc files for you; that's your dotfiles. Suggested:"
    );
    for f in &hits {
        println!(
            "  $ $EDITOR {}   # delete or comment the HSA_OVERRIDE_GFX_VERSION line",
            f.display()
        );
    }
    exit::OK
}

fn run_unset_override_windows(opts: &FixOptions) -> i32 {
    let current = std::env::var("HSA_OVERRIDE_GFX_VERSION").unwrap_or_default();
    if current.is_empty() {
        println!("HSA_OVERRIDE_GFX_VERSION is not set in this shell.");
    } else {
        println!("HSA_OVERRIDE_GFX_VERSION={current} is set in this shell.");
        println!("Note: clearing it in your Windows env scope does NOT affect this");
        println!("already-open shell -- close and reopen your terminal afterwards.");
    }
    // A failed read here only affects this report-and-clear path, which never
    // overwrites the whole variable (it only clears it via a dedicated
    // `setx VAR ""`), so collapsing a failed read to "unset" preserves this
    // function's existing behavior; it is `path_export_windows`'s
    // read-then-overwrite-the-whole-value shape that made that collapse unsafe.
    let user_val = ps_env_scope("HSA_OVERRIDE_GFX_VERSION", "User").unwrap_or_default();
    let machine_val = ps_env_scope("HSA_OVERRIDE_GFX_VERSION", "Machine").unwrap_or_default();
    report_and_clear_override_windows(opts, &user_val, &machine_val, |prompt| {
        confirm(prompt, opts.yes)
    })
}

/// Does the reporting/clearing work for [`run_unset_override_windows`], with
/// the User/Machine scope values and the consent prompt taken as parameters
/// rather than read from `ps_env_scope`/`confirm` directly. That's what makes
/// the decline path testable without a real Windows host -- same reason
/// [`pin_device_in_rc_file`] takes its consent as a closure.
///
/// A decline must still exit `5`: `skills/rocm-doctor/reference.md` documents
/// `5` as "user declined", and a caller (human or agent) relies on that to
/// tell "nothing happened because you said no" apart from "nothing happened
/// because there was nothing to do". Declining just the User scope can't
/// short-circuit straight to `return exit::DECLINED`, though -- if the Machine scope is
/// also set, the guidance for clearing it (which needs an elevated shell)
/// still has to print unconditionally, so the decline is recorded in a flag
/// and checked only once both scopes have had their say.
fn report_and_clear_override_windows(
    opts: &FixOptions,
    user_val: &str,
    machine_val: &str,
    consent: impl FnOnce(&str) -> bool,
) -> i32 {
    if user_val.is_empty() && machine_val.is_empty() {
        println!("\nNo persistent HSA_OVERRIDE_GFX_VERSION found in either the User");
        println!("or Machine env scope. You're done after closing/reopening shells.");
        return exit::OK;
    }
    println!("\nPersistent HSA_OVERRIDE_GFX_VERSION found in:");
    if !user_val.is_empty() {
        println!("  User scope:    {user_val}");
    }
    if !machine_val.is_empty() {
        println!("  Machine scope: {machine_val}");
    }
    let mut declined = false;
    if !user_val.is_empty() {
        println!("\nClear from the User scope (no admin needed):");
        // What the CLI runs, not a line to paste: the process is started
        // directly with these arguments, empty one included. Windows PowerShell
        // 5.1 drops an empty argument, so pasting the line would not clear
        // anything; the plan above carries the line to paste. The text is built
        // from the argument list that is run, so the two cannot differ.
        let setx_args = ["HSA_OVERRIDE_GFX_VERSION", ""];
        let shown: Vec<String> = setx_args.iter().map(|a| format!("\"{a}\"")).collect();
        println!(
            "  The CLI runs (arguments, not a line to paste): setx {}",
            shown.join(" ")
        );
        if opts.dry_run {
            println!("  (dry-run; not executed)");
        } else if consent("Clear HSA_OVERRIDE_GFX_VERSION from User scope?") {
            let (rc, out, err) = run("setx", &setx_args, RUN_TIMEOUT);
            relay_output(&out, &err);
            if rc != 0 {
                fail!("setx exited {rc}; User scope NOT changed.");
                return exit::FAILED;
            }
            println!("Cleared from User scope. Reopen your terminal for it to take effect.");
        } else {
            declined = true;
        }
    }
    if !machine_val.is_empty() {
        println!(
            "\nThe Machine scope value cannot be cleared without an Admin shell. Either run an elevated PowerShell and execute:"
        );
        println!(
            "  [Environment]::SetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION', $null, 'Machine')"
        );
        println!(
            "or remove it through System Properties -> Environment Variables -> System variables. This command does NOT elevate itself."
        );
    }
    if declined { exit::DECLINED } else { exit::OK }
}

/// fix-6: persist the ROCm/HIP bin directory on PATH (with consent).
pub(super) fn run_path_export(opts: &FixOptions) -> i32 {
    if runtime_is_windows() {
        run_path_export_windows(opts)
    } else {
        run_path_export_linux(opts)
    }
}

fn run_path_export_linux(opts: &FixOptions) -> i32 {
    // Same resolver `examine` uses, so the line we append names the install the
    // report pointed at -- including a versioned root like /opt/rocm-6.4.1.
    let Some(install) = crate::discover_rocm_installs().into_iter().next() else {
        fail!("No ROCm install found; nothing to add to PATH.");
        return exit::NOT_APPLICABLE;
    };
    let bin_path = install.path.join("bin");
    if !bin_path.is_dir() {
        fail!(
            "{} does not exist; nothing to add to PATH.",
            bin_path.display()
        );
        return exit::NOT_APPLICABLE;
    }
    let bin_dir_owned = bin_path.to_string_lossy().into_owned();
    let bin_dir = bin_dir_owned.as_str();
    let Some(rc_file) = shell_rc_file() else {
        fail!("Could not determine your home directory.");
        return exit::NOT_APPLICABLE;
    };
    let export_line = format!("export PATH=\"{bin_dir}:$PATH\"");
    if let Ok(existing) = std::fs::read_to_string(&rc_file)
        && existing
            .lines()
            .any(|l| l.contains("PATH=") && l.contains(bin_dir))
    {
        println!(
            "{} already adds {bin_dir} to PATH; no change.",
            rc_file.display()
        );
        return exit::OK;
    }
    println!("Plan: append the following line to {}:", rc_file.display());
    println!("  {export_line}");
    if opts.dry_run {
        println!("(dry-run; not executed)");
        return exit::OK;
    }
    if !confirm(&format!("Append to {}?", rc_file.display()), opts.yes) {
        return exit::DECLINED;
    }
    if let Err(exc) = append_line(
        &rc_file,
        "# Added by rocm examine (fix-6-path)",
        &export_line,
    ) {
        fail!("Failed to write {}: {exc}", rc_file.display());
        return exit::FAILED;
    }
    println!(
        "Appended to {}. Open a new shell or run `source {}` for the change to take effect.",
        rc_file.display(),
        rc_file.display()
    );
    exit::OK
}

fn run_path_export_windows(opts: &FixOptions) -> i32 {
    let mut sdk_path = std::env::var("HIP_PATH").unwrap_or_default();
    if sdk_path.is_empty() {
        sdk_path = newest_rocm_install_dir();
    }
    if sdk_path.is_empty() {
        fail!("No HIP SDK install found. Run fix-13-hip-sdk-missing first.");
        return exit::NOT_APPLICABLE;
    }
    let bin_dir = Path::new(&sdk_path).join("bin");
    if !bin_dir.is_dir() {
        fail!(
            "{} does not exist on disk; HIP SDK install looks incomplete.",
            bin_dir.display()
        );
        return exit::NOT_APPLICABLE;
    }
    let bin_dir = bin_dir.to_string_lossy().into_owned();
    path_export_windows(opts, &bin_dir, || ps_env_scope("PATH", "User"))
}

/// The literal value that will replace User-scope PATH on a non-empty
/// existing value: `bin_dir` first, so whatever else is already on PATH
/// cannot shadow it with a same-named binary. This matches the Linux
/// sibling's intent (`run_path_export_linux`'s `export PATH="{bin_dir}:$PATH"`
/// also puts `bin_dir` first) and the "Plan: prepend..." message
/// `path_export_windows` prints next to it.
///
/// Kept as its own function, called from both the plan message and the
/// value actually written, so the two can never disagree with each other the
/// way the unfixed code did (print said "prepend", the write appended)
/// -- see AGENTS.md section 3 on deriving a printed claim from the same
/// value the branch acts on.
fn windows_path_prepend(bin_dir: &str, user_path: &str) -> String {
    if user_path.is_empty() {
        bin_dir.to_owned()
    } else {
        format!("{bin_dir};{user_path}")
    }
}

/// Does the read/plan/write work for [`run_path_export_windows`], with the
/// User-scope PATH read taken as an injected closure rather than calling
/// `ps_env_scope` directly. That's what makes the "read failed" branch
/// testable on any host -- same reason [`report_and_clear_override_windows`]
/// takes consent as a closure.
///
/// A failed read (`None`) must never be treated as "PATH is empty": the
/// write below replaces the *entire* User PATH value, so collapsing a
/// PowerShell failure into "nothing there" would wipe the user's real PATH
/// down to just `bin_dir`. So a failed read aborts here, before the
/// "Plan: ..." preview and before the write is ever attempted, with a
/// return code (`4`, matching a write failure) distinct from `0` ("nothing
/// to do") and from `5` ("user declined").
fn path_export_windows(
    opts: &FixOptions,
    bin_dir: &str,
    read_user_path: impl FnOnce() -> Option<String>,
) -> i32 {
    let Some(user_path) = read_user_path() else {
        fail!("Could not read the current User PATH; User PATH NOT changed.");
        return exit::FAILED;
    };
    if !user_path.is_empty() && user_path.to_lowercase().contains(&bin_dir.to_lowercase()) {
        println!("User PATH already contains {bin_dir}; no change.");
        return exit::OK;
    }
    let new_path = windows_path_prepend(bin_dir, &user_path);
    // Escaped once here, before either the printed preview or the real
    // write script embeds it, so the two can never disagree about what
    // actually gets executed (same reasoning as `windows_path_prepend`
    // being shared, above) -- see `ps_single_quote_escape`'s doc comment
    // for why this escape is required at all.
    let escaped_new_path = ps_single_quote_escape(&new_path);
    println!("Plan: prepend {bin_dir} to your User PATH:");
    println!("  [Environment]::SetEnvironmentVariable('PATH', '{escaped_new_path}', 'User')");
    if opts.dry_run {
        println!("(dry-run; not executed)");
        return exit::OK;
    }
    if !confirm("Update User PATH?", opts.yes) {
        return exit::DECLINED;
    }
    // PowerShell's SetEnvironmentVariable, not `setx`: `setx` silently
    // truncates values past 1024 characters while still exiting 0, which a
    // normal-length PATH can hit even when the read above succeeded.
    let script =
        format!("[Environment]::SetEnvironmentVariable('PATH', '{escaped_new_path}', 'User')");
    let (rc, out, err) = run(
        "powershell",
        &["-NoProfile", "-Command", &script],
        RUN_TIMEOUT,
    );
    relay_output(&out, &err);
    if rc != 0 {
        fail!("PowerShell exited {rc}; User PATH NOT changed.");
        return exit::FAILED;
    }
    println!(
        "Added {bin_dir} to your User PATH. This only takes effect in NEW shells -- close this terminal and reopen it before re-running hipInfo."
    );
    exit::OK
}

/// fix-9: persist HIP_VISIBLE_DEVICES so the iGPU is hidden.
pub(super) fn run_hip_visible_devices(opts: &FixOptions) -> i32 {
    if let Some(idx) = opts.device_index.filter(|&i| i < 0) {
        fail!("--device-index must be >= 0 (got {idx}).");
        return exit::NOT_APPLICABLE;
    }
    if runtime_is_windows() {
        run_hip_visible_devices_windows(opts)
    } else {
        run_hip_visible_devices_linux(opts)
    }
}

fn run_hip_visible_devices_linux(opts: &FixOptions) -> i32 {
    let Some(idx) = opts.device_index else {
        // Print-only path: without an index the fix only prints the query that
        // helps the user identify N, so it succeeds like any other preview.
        println!(
            "Run `rocminfo | grep -E 'Agent |Marketing|gfx'` and identify the row of your DISCRETE GPU (the iGPU is typically Agent 1). Then re-run with --device-index N."
        );
        return exit::OK;
    };
    let Some(rc_file) = shell_rc_file() else {
        fail!("Could not determine your home directory.");
        return exit::NOT_APPLICABLE;
    };
    pin_device_in_rc_file(&rc_file, idx, opts, || {
        confirm(&format!("Append to {}?", rc_file.display()), opts.yes)
    })
}

/// The part of fix-9 that actually touches the user's machine: decide whether
/// the rc file already pins a device, honour `--dry-run`, ask for consent, and
/// append.
///
/// Taking the rc path and the consent decision as parameters keeps this — the
/// one auto-fix path on Linux that really mutates a file — testable end to end
/// against a temp file, with no `$HOME` mutation and no dependency on whether
/// stdin happens to be a terminal. Exit codes are the contract documented in
/// `skills/rocm-doctor/reference.md`: 0 applied/dry-run/already-set, 4 write
/// failed, 5 declined.
fn pin_device_in_rc_file(
    rc_file: &Path,
    idx: i64,
    opts: &FixOptions,
    consent: impl FnOnce() -> bool,
) -> i32 {
    let export_line = format!("export HIP_VISIBLE_DEVICES={idx}");
    if let Ok(existing) = std::fs::read_to_string(rc_file)
        && existing.contains("HIP_VISIBLE_DEVICES=")
    {
        println!(
            "{} already sets HIP_VISIBLE_DEVICES; edit by hand rather than appending a second copy.",
            rc_file.display()
        );
        return exit::OK;
    }
    println!("Plan: append the following line to {}:", rc_file.display());
    println!("  {export_line}");
    if opts.dry_run {
        println!("(dry-run; not executed)");
        return exit::OK;
    }
    if !consent() {
        return exit::DECLINED;
    }
    if let Err(exc) = append_line(
        rc_file,
        "# Added by rocm examine (fix-9-igpu-dgpu)",
        &export_line,
    ) {
        fail!("Failed to write {}: {exc}", rc_file.display());
        return exit::FAILED;
    }
    println!(
        "Appended to {}. Open a new shell for the change to take effect, then re-run your workload.",
        rc_file.display()
    );
    exit::OK
}

fn run_hip_visible_devices_windows(opts: &FixOptions) -> i32 {
    let Some(idx) = opts.device_index else {
        // Print-only path: without an index the fix only prints the query that
        // helps the user identify N, so it succeeds like any other preview.
        println!("Run the following to identify the discrete GPU's index:");
        println!(
            "  & \"$env:HIP_PATH\\bin\\hipInfo.exe\" | Select-String \"device#|Name|gcnArchName\""
        );
        println!(
            "Then re-run with --device-index N (the iGPU is typically device# 0; the dGPU is usually device# 1)."
        );
        return exit::OK;
    };
    // Out of scope for ROCMAI-449 (that fix is specific to fix-6-path's
    // read-then-overwrite-the-whole-PATH-value shape): a failed read here
    // only risks re-running `setx HIP_VISIBLE_DEVICES <idx>`, which sets a
    // single unrelated variable rather than overwriting PATH.
    let existing = ps_env_scope("HIP_VISIBLE_DEVICES", "User").unwrap_or_default();
    if !existing.is_empty() {
        println!(
            "User scope already sets HIP_VISIBLE_DEVICES={existing:?}; remove or update it manually rather than overwriting from this command."
        );
        return exit::OK;
    }
    println!("Plan: persist HIP_VISIBLE_DEVICES in the User env scope:");
    println!("  setx HIP_VISIBLE_DEVICES {idx}");
    if opts.dry_run {
        println!("(dry-run; not executed)");
        return exit::OK;
    }
    if !confirm("Set HIP_VISIBLE_DEVICES in the User scope?", opts.yes) {
        return exit::DECLINED;
    }
    let (rc, out, err) = run(
        "setx",
        &["HIP_VISIBLE_DEVICES", &idx.to_string()],
        RUN_TIMEOUT,
    );
    relay_output(&out, &err);
    if rc != 0 {
        fail!("setx exited {rc}; HIP_VISIBLE_DEVICES NOT changed.");
        return exit::FAILED;
    }
    println!(
        "setx only takes effect in NEW shells -- close this terminal and reopen it before re-running your workload."
    );
    exit::OK
}

/// Read a Windows environment variable from a given scope via PowerShell.
///
/// Returns `None` if the read itself failed -- PowerShell missing, blocked by
/// execution policy, or the query timing out -- which must stay distinguishable
/// from `Some(String::new())`, meaning the read succeeded and the variable is
/// genuinely unset/empty in that scope. Collapsing both to `""` is what let a
/// failed PATH read look like an empty PATH to a caller that then wrote the
/// "empty" value back out, wiping the real one (ROCMAI-449).
fn ps_env_scope(var: &str, scope: &str) -> Option<String> {
    let script = format!("[Environment]::GetEnvironmentVariable('{var}','{scope}')");
    let (rc, out, _) = run(
        "powershell",
        &["-NoProfile", "-Command", &script],
        QUERY_TIMEOUT,
    );
    if rc == 0 {
        Some(out.trim().to_owned())
    } else {
        None
    }
}

/// Newest ROCm/HIP SDK install root on this host, or empty if there is none.
///
/// This was the third independent Windows install scan in the codebase, and it
/// disagreed with the other two on every axis that matters: it searched
/// `C:\Program Files (x86)\AMD\ROCm` where the resolver searches
/// `C:\Program Files\ROCm`, it sorted with a plain `versions.sort()` so `6.2`
/// outranked `6.10`, and it accepted any directory whose name began with a
/// digit without checking for an install marker. So `fix-6-path` could put an
/// older SDK — or an empty directory — on the user's PATH.
///
/// It now asks the same resolver as `examine`, which also means `$ROCM_PATH` is
/// honoured here for the first time.
fn newest_rocm_install_dir() -> String {
    first_install_path(crate::discover_rocm_installs())
}

/// The best install's path, or empty when there is none.
fn first_install_path(installs: Vec<crate::RocmInstall>) -> String {
    installs
        .into_iter()
        .next()
        .map(|install| install.path.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// [`newest_rocm_install_dir`] against a caller-supplied `$ROCM_PATH` and
/// search roots, so a test can drive it without touching the process
/// environment. Same seam as [`crate::discover_rocm_installs_in`].
#[cfg(test)]
fn newest_rocm_install_dir_in(
    search_dirs: &[std::path::PathBuf],
    env_override: Option<&std::path::Path>,
) -> String {
    first_install_path(discover_rocm_installs_on_host_in(search_dirs, env_override))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Serializes tests that replace the process-global `ROCM_PATH` env var while
    // they run. Because env is shared across all test threads, two such tests
    // running concurrently can otherwise see each other's value mid-test.
    //
    // Deliberately kept alongside the seam rather than instead of it, and the
    // split is not arbitrary: tests about resolution semantics take the seam
    // and never touch the environment, and exactly one test — the one whose
    // subject IS the `$ROCM_PATH` read — takes this lock. Anything provable
    // through the seam should not be reaching for the lock.
    static PROCESS_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Plant a directory the shared resolver will accept as a ROCm install.
    /// `bin/rocminfo` is one of the markers it gates on; a bare directory is
    /// deliberately not enough.
    fn plant_install(root: &std::path::Path) {
        std::fs::create_dir_all(root.join("bin")).expect("create planted install");
        std::fs::write(root.join("bin").join("rocminfo"), "").expect("write marker");
    }

    #[test]
    fn the_path_fix_finds_the_install_the_rest_of_the_cli_found() {
        // Discriminating on purpose: the scanner this replaces looked only in
        // two hardcoded `C:\Program Files` directories and ignored $ROCM_PATH
        // outright, so it returned nothing here no matter what was planted.
        // Going through the shared resolver is what makes this pass -- and is
        // what stops fix-6-path putting 6.2 on PATH when 6.10 is installed.
        let root = std::env::temp_dir().join(format!(
            "rocm-fix-path-resolver-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let install = root.join("rocm-6.10.0");
        plant_install(&install);

        // A second, older install reachable through the hardcoded search roots.
        // Passing it alongside the override is what exercises the ordering the
        // assertion below claims: with `&[]` the search loop never runs, so
        // "the override outranks the search roots" would hold vacuously.
        //
        // Planted in BOTH shapes because the resolver is told the host's
        // layout: `rocm-6.2.0` is a versioned sibling and only matches on
        // Linux, `6.2` is a bare-version child and only matches on Windows.
        // Planting one shape would leave the search loop empty on the other
        // platform and make the ordering half vacuous again -- on Windows
        // first, which is the lane this test exists for.
        let searched = root.join("search");
        plant_install(&searched.join("rocm-6.2.0"));
        plant_install(&searched.join("6.2"));

        // The override goes in as an argument rather than through
        // `std::env::set_var`: the environment is process-global, so a sibling
        // test mutating $ROCM_PATH between this set and its read used to make
        // this assertion fail on whichever test lost the race.
        // `..._reads_rocm_path_from_the_environment` covers the real read.
        let found = newest_rocm_install_dir_in(std::slice::from_ref(&searched), Some(&install));
        // Run the same search WITHOUT the override, so the ordering claim below
        // cannot pass by finding nothing to outrank. A planted decoy the
        // resolver's layout does not recognise is indistinguishable, from the
        // assertion's point of view, from no decoy at all.
        let without_override = newest_rocm_install_dir_in(std::slice::from_ref(&searched), None);

        // The check above only exercises whichever decoy shape THIS host's
        // layout recognises, so deleting the other one leaves Linux green and
        // the regression waits for the Windows lane to surface it. The seam
        // already takes the layout, so driving it with both discriminates on
        // every host for the cost of one loop.
        let by_layout: Vec<(RocmLayout, bool)> = [
            (RocmLayout::Siblings, "rocm-6.2.0"),
            (RocmLayout::Children, "6.2"),
        ]
        .into_iter()
        .map(|(layout, decoy)| {
            let installs =
                discover_rocm_installs_in_layout(std::slice::from_ref(&searched), None, layout);
            (
                layout,
                installs.iter().any(|install| install.path.ends_with(decoy)),
            )
        })
        .collect();

        std::fs::remove_dir_all(&root).ok();

        assert!(
            !without_override.is_empty(),
            "the decoy must be reachable through the search roots on this \
             platform, or 'the override outranks them' holds vacuously"
        );
        for (layout, reached) in by_layout {
            assert!(
                reached,
                "the {layout:?} decoy must be reachable under that layout, or \
                 the ordering claim is vacuous on the platform that uses it"
            );
        }
        assert_eq!(
            found,
            install.to_string_lossy(),
            "fix-6-path must resolve installs the same way examine does, and \
             $ROCM_PATH must outrank the hardcoded search roots"
        );
    }

    /// The seam tests above deliberately bypass `$ROCM_PATH`, so on their own
    /// the production read could be deleted and the suite would stay green.
    /// This one drives the real entry point against a real variable.
    ///
    /// It takes the lock rather than a seam because exercising the env read IS
    /// the point — the escape hatch the contract guard advertises for exactly
    /// this case.
    #[test]
    fn the_path_fix_reads_rocm_path_from_the_environment() {
        let _guard = PROCESS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let root = std::env::temp_dir().join(format!(
            "rocm-fix-path-env-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let install = root.join("rocm-6.10.0");
        plant_install(&install);

        // Restored on drop rather than on the next line, so a panic inside the
        // resolver cannot leave this key pointing at the directory removed
        // below. `RestoredEnvVar` only restores; the lock above is what
        // serializes, and the contract guard requires it here because
        // `RestoredEnvVar::set(` is in its mutation list.
        let restore = crate::test_env::RestoredEnvVar::set("ROCM_PATH", &install);
        let found = newest_rocm_install_dir();
        drop(restore);

        std::fs::remove_dir_all(&root).ok();

        assert_eq!(
            found,
            install.to_string_lossy(),
            "fix-6-path must reach $ROCM_PATH through the shared resolver"
        );
    }

    #[test]
    fn the_path_fix_reports_nothing_rather_than_a_directory_with_no_install_in_it() {
        // The old scan accepted any directory whose name started with a digit,
        // so an empty leftover could be put on PATH. The resolver requires a
        // marker.
        let root = std::env::temp_dir().join(format!(
            "rocm-fix-path-empty-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let empty = root.join("6.10");
        std::fs::create_dir_all(&empty).expect("create empty dir");

        let found = newest_rocm_install_dir_in(&[], Some(&empty));

        std::fs::remove_dir_all(&root).ok();

        assert!(
            found.is_empty(),
            "an empty directory is not an install, but {found:?} was returned"
        );
    }

    // ── Consent and mutation contract ──────────────────────────────
    //
    // `skills/rocm-doctor/reference.md` documents six exit codes, and the skill
    // tells an agent it may run an auto-fix because the CLI "refuses on a
    // non-interactive shell without --yes" and "confirms before mutating".
    // Before these tests, codes 1, 4 and 5 were asserted nowhere in the repo and
    // no test ever reached a runner that writes to disk: the two that look like
    // they cover it don't — `dry_run_never_mutates_...` picks fix-2, whose Linux
    // runner takes no FixOptions and never prompts, and diagnose.feature's
    // preview scenario picks print-only fix-1, which returns before any runner.
    //
    // fix-9 is the auto-fix used here because its Linux runner is the one that
    // genuinely appends to a file, and `pin_device_in_rc_file` lets that run
    // against a temp path with an injected consent verdict.

    /// A unique scratch path under the workspace's test-artifact dir, following
    /// the convention in `lib.rs`'s tests (no `tempfile` dev-dependency here).
    fn scratch_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("core")
            .join(format!(
                "fix-{name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
        std::fs::create_dir_all(&dir).expect("failed to create scratch dir");
        dir
    }

    fn pinning_opts() -> FixOptions {
        FixOptions {
            device_index: Some(1),
            ..FixOptions::default()
        }
    }

    /// `auto_applicable` means "the CLI has a runner", not "the runner mutates".
    ///
    /// fix-2's Linux arm only reports where the override is persisted; it never
    /// edits dotfiles. `auto_applicable_recipes_have_a_runner` cannot catch a
    /// regression here, because fix-2 does have a runner — so without this, a
    /// doc promising a `--dry-run` preview the Linux user never gets stays
    /// green. That is the drift `skills/rocm-doctor/` exists to prevent.
    ///
    /// Asserted on the files themselves rather than on an exit code: a runner
    /// that started stripping the line would still return 0.
    #[test]
    fn reporting_a_persistent_override_never_edits_the_rc_file() {
        let dir = scratch_dir("fix2-report");
        let carries = dir.join(".bashrc");
        let clean = dir.join(".profile");
        let before = "export HSA_OVERRIDE_GFX_VERSION=10.3.0\nexport PATH=$PATH:/x\n";
        std::fs::write(&carries, before).expect("seed rc file");
        std::fs::write(&clean, "# nothing to see\n").expect("seed rc file");

        let code = report_persistent_override(&[carries.clone(), clean.clone()]);

        assert_eq!(code, 0, "reporting is a success, not a refusal");
        assert_eq!(
            std::fs::read_to_string(&carries).expect("rc file still readable"),
            before,
            "fix-2 on Linux must leave the user's dotfiles byte-identical — \
             skills/rocm-doctor/ tells an agent it only reports"
        );
        assert_eq!(
            std::fs::read_to_string(&clean).expect("rc file still readable"),
            "# nothing to see\n",
            "a file that never carried the override must not be touched either"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The Windows arm of fix-2 does mutate (it runs `setx` to clear the User
    /// scope), so unlike the Linux arm above, a decline has to come back as
    /// `5`, not `0` -- `skills/rocm-doctor/reference.md` documents `5` as
    /// "user declined" and an agent tells that apart from "nothing to do"
    /// this way.
    #[test]
    fn windows_decline_on_user_scope_returns_5() {
        let code =
            report_and_clear_override_windows(&FixOptions::default(), "10.3.0", "", |_| false);

        assert_eq!(code, 5, "a declined User-scope clear must exit 5, not 0");
    }

    /// Declining the User scope must not swallow the Machine-scope guidance:
    /// a user with both scopes set still needs the elevated-shell
    /// instructions printed, so the decline can't `return 5` on the spot --
    /// it has to fall through and let the Machine-scope block run first.
    /// This test can't see stdout, but it pins the return code so a future
    /// change that reintroduces an early `return 5` (skipping that block)
    /// would have to change this assertion to keep passing.
    #[test]
    fn windows_decline_with_machine_scope_also_set_still_returns_5() {
        let code =
            report_and_clear_override_windows(&FixOptions::default(), "10.3.0", "11.0.0", |_| {
                false
            });

        assert_eq!(code, 5);
    }

    #[test]
    fn windows_consent_granted_is_not_treated_as_a_decline() {
        // `setx` isn't on this host, so a real run would report failure (`4`)
        // -- the point here is only that the consent gate itself was
        // reached and answered "yes", not routed as a decline.
        let code =
            report_and_clear_override_windows(&FixOptions::default(), "10.3.0", "", |_| true);

        assert_ne!(
            code, 5,
            "granted consent must never be reported as declined"
        );
    }

    #[test]
    fn windows_dry_run_short_circuits_before_the_consent_gate() {
        let opts = FixOptions {
            dry_run: true,
            ..FixOptions::default()
        };

        let code = report_and_clear_override_windows(&opts, "10.3.0", "", |_| {
            panic!("dry-run must not reach the consent gate")
        });

        assert_eq!(code, 0, "a dry-run preview is not a decline");
    }

    #[test]
    fn windows_nothing_persisted_in_either_scope_returns_0_without_prompting() {
        let code = report_and_clear_override_windows(&FixOptions::default(), "", "", |_| {
            panic!("nothing to clear means no consent gate at all")
        });

        assert_eq!(code, 0);
    }

    /// ROCMAI-449: a failed PATH read must never be treated as "PATH is
    /// empty". Against the unfixed code (`ps_env_scope` collapsing any
    /// failure to `""`), this read would look identical to a genuinely empty
    /// PATH, and the old `run_path_export_windows` would go on to run
    /// `setx PATH "<bin_dir>"` -- replacing the user's entire real PATH with
    /// one entry.
    ///
    /// `opts.dry_run = true` is the discriminator, not `opts.yes`: on a host
    /// without a real `powershell`/`setx` (this Mac included), the actual
    /// write fails regardless of whether it was ever reached, so asserting
    /// against a "the write failed" code cannot tell "aborted before the
    /// write" apart from "attempted the write and the write itself failed"
    /// -- both land on the same non-zero code by coincidence of this host's
    /// missing binary. Collapsing the failed read to `""` instead falls
    /// through to the dry-run short-circuit (which returns `0`, "preview
    /// printed, nothing to do"), while the fix aborts at the read itself and
    /// never reaches that branch -- so this is the one assertion that is
    /// both host-independent and still fails against the unfixed
    /// collapse-to-empty behavior.
    #[test]
    fn windows_path_read_failure_aborts_before_plan_or_write() {
        let opts = FixOptions {
            dry_run: true,
            ..FixOptions::default()
        };

        let code = path_export_windows(&opts, r"C:\ROCm\bin", || {
            None // the read itself failed (PowerShell missing/blocked/timed out)
        });

        assert_ne!(
            code, 0,
            "a failed read must not be silently absorbed by the dry-run preview"
        );
        assert_ne!(
            code, 5,
            "a failed read was never offered a prompt to decline"
        );
    }

    /// A genuinely empty User PATH (the read succeeded and returned `""`,
    /// not a failure) must still work exactly as before: the new value is
    /// `bin_dir` alone, with no leading or trailing separator.
    #[test]
    fn windows_path_prepend_on_genuinely_empty_path_is_bin_dir_alone() {
        assert_eq!(windows_path_prepend(r"C:\ROCm\bin", ""), r"C:\ROCm\bin");
    }

    /// Per AGENTS.md section 3: a claim of outcome ("Plan: prepend...") must
    /// be asserted together with the resulting state, not pinned as a string
    /// on its own. `windows_path_prepend` is the single function both the
    /// printed plan and the real write derive their value from (see its doc
    /// comment), so calling it directly here pins both at once: the literal
    /// value a real write would send, which is the same test that would have
    /// failed against the unfixed `format!("{user_path};{bin_dir}")` -- that
    /// built `"C:\Windows;C:\Other;C:\ROCm\bin"`, which does NOT start with
    /// `bin_dir`.
    #[test]
    fn windows_path_prepend_on_existing_path_starts_with_bin_dir_not_ends_with_it() {
        let bin_dir = r"C:\ROCm\bin";
        let existing = r"C:\Windows;C:\Other";

        let new_path = windows_path_prepend(bin_dir, existing);

        assert_eq!(new_path, format!("{bin_dir};{existing}"));
        assert!(
            new_path.starts_with(bin_dir),
            "a prepend must start with bin_dir, not end with it: got {new_path:?}"
        );
    }

    /// Round 2 of ROCMAI-449: an apostrophe anywhere in the real User PATH
    /// (e.g. `C:\Users\O'Brien\bin`, an ordinary Windows path, not a
    /// contrived edge case) must not be able to split the generated
    /// PowerShell `-Command` script mid-statement. Before this fix, the
    /// write script was built as
    /// `format!("...SetEnvironmentVariable('PATH', '{new_path}', 'User')")`
    /// with `new_path` interpolated raw: an unescaped `'` in it would end
    /// the literal early, and the remainder of `new_path` plus the trailing
    /// `, 'User')` would be parsed as separate PowerShell tokens rather than
    /// staying part of the string value.
    ///
    /// This builds the exact script string [`path_export_windows`]'s write
    /// path constructs and asserts it is well-formed: every single quote in
    /// it belongs to one of exactly three delimiter pairs (`'PATH'`,
    /// `'<value>'`, `'User'`), so the embedded value itself must contain
    /// zero *unescaped* quotes -- i.e. the quote count inside the value
    /// region is exactly double the number of literal `'` characters the
    /// unescaped input contained.
    #[test]
    fn write_script_escapes_apostrophe_in_real_path_value() {
        let bin_dir = r"C:\ROCm\bin";
        let user_path = r"C:\Users\O'Brien\bin;C:\Windows";

        let new_path = windows_path_prepend(bin_dir, user_path);
        let escaped_new_path = ps_single_quote_escape(&new_path);
        let script =
            format!("[Environment]::SetEnvironmentVariable('PATH', '{escaped_new_path}', 'User')");

        // The raw (unescaped) new_path must contain exactly one embedded
        // apostrophe here (from "O'Brien"), carried over unchanged from
        // user_path.
        let raw_quote_count = new_path.matches('\'').count();
        assert_eq!(raw_quote_count, 1);

        // After escaping, that one embedded quote must have become two
        // (`''`), so the escaped value contributes 2 quote characters to
        // the script, on top of the 6 delimiter quotes from the 3
        // single-quoted literals ('PATH', the value, 'User').
        let script_quote_count = script.matches('\'').count();
        assert_eq!(
            script_quote_count,
            6 + 2 * raw_quote_count,
            "every embedded ' must appear as '' in the script, on top of the 3 literals' own delimiters: {script:?}"
        );

        // Concretely: the value literal must read as a single balanced
        // '...' token with the embedded quote doubled, not end early at the
        // first unescaped '.
        assert!(
            script.contains(r"'C:\ROCm\bin;C:\Users\O''Brien\bin;C:\Windows'"),
            "escaped value literal not found intact in script: {script:?}"
        );
        // And the trailing ", 'User')" must stay the closing argument list,
        // not get parsed as stray tokens after a prematurely-closed string.
        assert!(script.ends_with(", 'User')"));
    }

    /// Regression coverage for control flow this change does not touch: an
    /// existing User PATH that already contains `bin_dir` must short-circuit
    /// to 0 without ever reaching the confirm gate. `FixOptions::default()`
    /// has `yes: false` and `dry_run: false`, and `confirm()` on a
    /// non-interactive test process always declines (code `5`), so if this
    /// short-circuit broke and control fell through to the confirm gate,
    /// this would observe `5` instead of `0`.
    #[test]
    fn windows_path_already_contains_bin_dir_short_circuits_with_no_change() {
        let bin_dir = r"C:\ROCm\bin";
        let code = path_export_windows(&FixOptions::default(), bin_dir, || {
            Some(format!(r"C:\Windows;{bin_dir}"))
        });

        assert_eq!(
            code, 0,
            "already containing bin_dir must short-circuit before the confirm gate"
        );
    }

    /// Regression coverage: `--dry-run` must still preview and return 0
    /// without ever reaching the confirm gate, by the same non-interactive
    /// discriminator as the test above (a fall-through would decline and
    /// return 5, not 0).
    #[test]
    fn windows_path_dry_run_short_circuits_before_confirm() {
        let opts = FixOptions {
            dry_run: true,
            ..FixOptions::default()
        };

        let code = path_export_windows(&opts, r"C:\ROCm\bin", || Some(String::new()));

        assert_eq!(code, 0, "a dry-run preview is not a decline");
    }

    #[test]
    fn declining_consent_returns_5_and_writes_nothing() {
        let rc = scratch_dir("declined").join(".bashrc");
        std::fs::write(&rc, "# existing\n").expect("seed rc file");

        let code = pin_device_in_rc_file(&rc, 1, &pinning_opts(), || false);

        assert_eq!(code, 5, "a declined fix must exit 5");
        assert_eq!(
            std::fs::read_to_string(&rc).expect("read rc"),
            "# existing\n",
            "a declined fix must not touch the file"
        );
        std::fs::remove_dir_all(rc.parent().expect("parent")).ok();
    }

    #[test]
    fn dry_run_returns_0_and_writes_nothing_even_with_consent() {
        let rc = scratch_dir("dryrun").join(".bashrc");
        std::fs::write(&rc, "# existing\n").expect("seed rc file");
        let opts = FixOptions {
            dry_run: true,
            ..pinning_opts()
        };

        // Consent would be granted; --dry-run must short-circuit before it.
        let code = pin_device_in_rc_file(&rc, 1, &opts, || {
            panic!("dry-run must not reach the consent gate")
        });

        assert_eq!(code, 0);
        assert_eq!(
            std::fs::read_to_string(&rc).expect("read rc"),
            "# existing\n",
            "a dry-run must not touch the file"
        );
        std::fs::remove_dir_all(rc.parent().expect("parent")).ok();
    }

    #[test]
    fn granting_consent_appends_the_export_and_returns_0() {
        let rc = scratch_dir("granted").join(".bashrc");
        std::fs::write(&rc, "# existing\n").expect("seed rc file");

        let code = pin_device_in_rc_file(&rc, 3, &pinning_opts(), || true);

        assert_eq!(code, 0);
        let body = std::fs::read_to_string(&rc).expect("read rc");
        assert!(
            body.starts_with("# existing\n"),
            "existing rc content must be preserved:\n{body}"
        );
        assert!(
            body.contains("export HIP_VISIBLE_DEVICES=3"),
            "expected the export line to be appended:\n{body}"
        );
        std::fs::remove_dir_all(rc.parent().expect("parent")).ok();
    }

    #[test]
    fn an_rc_file_that_already_pins_a_device_is_left_alone() {
        let rc = scratch_dir("already-pinned").join(".bashrc");
        let original = "export HIP_VISIBLE_DEVICES=0\n";
        std::fs::write(&rc, original).expect("seed rc file");

        let code = pin_device_in_rc_file(&rc, 1, &pinning_opts(), || {
            panic!("an already-pinned rc file must not reach the consent gate")
        });

        assert_eq!(code, 0, "already-pinned is success, not failure");
        assert_eq!(
            std::fs::read_to_string(&rc).expect("read rc"),
            original,
            "must not append a second, conflicting pin"
        );
        std::fs::remove_dir_all(rc.parent().expect("parent")).ok();
    }

    #[test]
    fn a_failed_write_returns_4_rather_than_reporting_success() {
        // A directory where the rc file should be: the append fails at open().
        // This is the only way the documented "attempted but the command
        // failed" code is reachable for this fix.
        let dir = scratch_dir("write-fails");
        let rc = dir.join(".bashrc");
        std::fs::create_dir_all(&rc).expect("create dir in place of rc file");

        let code = pin_device_in_rc_file(&rc, 1, &pinning_opts(), || true);

        assert_eq!(code, 4, "a failed write must exit 4, not 0");
        std::fs::remove_dir_all(&dir).ok();
    }
}
