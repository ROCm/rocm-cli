// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm __man-page`: the rocm(1) manual page, as roff.
//!
//! The Linux installer runs the verb against the binary it has just placed, the
//! way it runs `rocm completions fish`, so the page always describes the binary
//! it is installed with. The parts clap knows (title, name, synopsis,
//! description, options and the command list) come from the same command tree
//! `rocm --help` prints, with the hidden internal verbs removed. The rest
//! (examples, files, environment, exit status) is written here, from the
//! constants the code itself uses where one exists.
//!
//! Only clap_mangen's per-section renderers are used. Its whole-page render
//! would add a SUBCOMMANDS section of `rocm-<command>(1)` references to pages
//! that are not shipped, and an EXTRA section repeating the `--help` epilogue.

use std::io::{self, Write};

use clap_mangen::Man;
use rocm_core::{CACHE_DIR_ENV, CONFIG_DIR_ENV, DATA_DIR_ENV};
use roff::{Roff, bold, italic, roman};

/// One example invocation (the arguments after `rocm`) and what it does. A
/// test parses every one through the real command line, so an example cannot
/// name a command or flag that does not exist.
const EXAMPLES: &[(&[&str], &str)] = &[
    (
        &["examine"],
        "Check this computer's GPU, ROCm install, engines, and setup folders. A good first command.",
    ),
    (
        &["install", "sdk", "--dry-run"],
        "Show which ROCm SDK would be installed, and where, without changing anything.",
    ),
    (
        &["install", "sdk"],
        "Install the ROCm SDK into a Python folder ROCm CLI manages.",
    ),
    (
        &["install", "driver", "--dry-run"],
        "Show what installing AMD driver support would do, without changing anything.",
    ),
    (
        &["runtimes", "list"],
        "List the ROCm runtimes ROCm CLI knows about, and which one is the default.",
    ),
    (
        &["update", "--dry-run"],
        "Show what installing a newer ROCm build would do, without changing anything.",
    ),
    (
        &["diagnose"],
        "Match this computer against a catalog of known ROCm, PyTorch and llama.cpp failure modes.",
    ),
    (
        &["fix"],
        "List the known fixes; pass an id that rocm diagnose reported to apply one.",
    ),
    (
        &["serve", "qwen"],
        "Start a local OpenAI-compatible server for the qwen model.",
    ),
    (
        &["services", "list"],
        "Show the local model servers that are running.",
    ),
    (
        &["daemon", "--status"],
        "Print the status of the automation watchers once, without starting the background \
         supervisor.",
    ),
    (
        &["uninstall", "--dry-run"],
        "Show what rocm uninstall would remove, without removing anything.",
    ),
];

/// Where the installer puts this page, below the folder above the install
/// folder: `~/.local/share/man/man1/rocm.1` for the default `~/.local/bin`.
const MAN_PAGE_PATH: &str = "<prefix>/share/man/man1/rocm.1";

/// The variable the Linux installer reads for the folder it installs into.
const INSTALL_DIR_ENV: &str = "ROCM_CLI_INSTALL_DIR";

/// Write the manual page for this binary to `w`.
pub(crate) fn write(w: &mut impl Write) -> io::Result<()> {
    render(crate::cli_command(), w)
}

/// Write the manual page for the command tree `root` to `w`.
fn render(root: clap::Command, w: &mut impl Write) -> io::Result<()> {
    let preview = preview_note(&root);
    let cmd = crate::without_hidden_subcommands(root)
        // The page is `rocm`'s, whatever `--version` calls the program.
        .display_name("rocm")
        .subcommand_value_name("command");
    let commands = commands_section(&cmd, preview.as_deref());
    let version = cmd.get_version().unwrap_or_default().replace('"', "'");
    // Written here rather than by `Man::render_title`, which leaves an empty
    // date unquoted, so the source and manual shift into the wrong fields.
    writeln!(w, ".TH ROCM 1 \"\" \"rocm {version}\" \"ROCm CLI\"")?;
    let man = Man::new(cmd);
    man.render_name_section(w)?;
    man.render_synopsis_section(w)?;
    man.render_description_section(w)?;
    man.render_options_section(w)?;
    commands.to_writer(w)?;
    examples_section().to_writer(w)?;
    files_section().to_writer(w)?;
    environment_section().to_writer(w)?;
    exit_status_section().to_writer(w)?;
    see_also_section().to_writer(w)
}

/// The paragraph of the `--help` epilogue that says what `[preview]` means,
/// so the page explains the marker its command list shows.
fn preview_note(root: &clap::Command) -> Option<String> {
    let epilogue = root.get_after_help()?.to_string();
    epilogue
        .split("\n\n")
        .find(|paragraph| paragraph.contains("[preview]"))
        .map(|paragraph| paragraph.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// One entry per top-level command `cmd` has, in the order `--help` lists
/// them, each with the summary `--help` prints for it.
fn commands_section(cmd: &clap::Command, preview: Option<&str>) -> Roff {
    let mut subcommands: Vec<&clap::Command> = cmd.get_subcommands().collect();
    subcommands.sort_by_key(|sc| sc.get_name());
    let mut roff = Roff::new();
    roff.control("SH", ["COMMANDS"]);
    for sc in subcommands {
        roff.control("TP", []);
        roff.text([bold(sc.get_name())]);
        if let Some(about) = sc.get_about() {
            roff.text([roman(about.to_string())]);
        }
    }
    roff.control("PP", []);
    if let Some(preview) = preview {
        roff.text([roman(preview)]);
        roff.control("PP", []);
    }
    roff.text([
        roman("Run "),
        bold("rocm"),
        roman(" "),
        italic("command"),
        roman(" "),
        bold("--help"),
        roman(" for each command's options."),
    ]);
    roff
}

fn examples_section() -> Roff {
    let mut roff = Roff::new();
    roff.control("SH", ["EXAMPLES"]);
    for (args, what) in EXAMPLES {
        let argv: Vec<&str> = std::iter::once("rocm")
            .chain(args.iter().copied())
            .collect();
        roff.control("TP", []);
        roff.text([bold(argv.join(" "))]);
        roff.text([roman(*what)]);
    }
    roff
}

fn files_section() -> Roff {
    let entries: [(String, String); 8] = [
        (
            "~/.rocm".to_owned(),
            format!(
                "The configuration and data folder: runtimes, engines, model servers and \
                 downloaded models. {CONFIG_DIR_ENV} and {DATA_DIR_ENV} move it."
            ),
        ),
        (
            "~/.rocm/config.json".to_owned(),
            "Saved settings, which rocm config shows and changes.".to_owned(),
        ),
        (
            "~/.rocm/cache".to_owned(),
            format!("Downloads and other caches. {CACHE_DIR_ENV} moves it."),
        ),
        (
            "~/.rocm/logs".to_owned(),
            "ROCm CLI's own log files.".to_owned(),
        ),
        (
            "<install-dir>/rocm, <install-dir>/rocmd".to_owned(),
            format!(
                "The binaries the installer places, in ~/.local/bin unless {INSTALL_DIR_ENV} \
                 names another folder."
            ),
        ),
        (
            "<install-dir>/.rocm-cli-manifest".to_owned(),
            "The binaries the installer placed, which the next install replaces.".to_owned(),
        ),
        (
            format!("<install-dir>/{}", crate::uninstall::PATH_EDIT_RECORD),
            "What else the installer changed: the shell profile it added the install folder \
             to, and the fish completions and manual page it wrote. rocm uninstall reads it \
             to undo them."
                .to_owned(),
        ),
        (
            MAN_PAGE_PATH.to_owned(),
            "This manual page, as the Linux installer writes it; <prefix> is the folder \
             above the install folder."
                .to_owned(),
        ),
    ];
    let mut roff = Roff::new();
    roff.control("SH", ["FILES"]);
    for (path, what) in entries {
        roff.control("TP", []);
        roff.text([italic(path)]);
        roff.text([roman(what)]);
    }
    roff
}

fn environment_section() -> Roff {
    let entries: [(&str, &str); 8] = [
        (
            CONFIG_DIR_ENV,
            "The configuration folder, in place of ~/.rocm.",
        ),
        (
            DATA_DIR_ENV,
            "The data folder, in place of ~/.rocm. Unless ROCM_CLI_CACHE_DIR is set too, \
             the cache moves with it, to a cache folder inside it.",
        ),
        (
            CACHE_DIR_ENV,
            "The cache folder, in place of ~/.rocm/cache.",
        ),
        (
            "ROCM_CLI_PYTHON",
            "A Python interpreter of your own, used instead of searching PATH or preparing \
             one. It must be a Python the ROCm install accepts.",
        ),
        (
            "ROCM_CLI_MANAGED_PYTHON_VERSION",
            "The Python version ROCm CLI prepares when it sets up a Python itself.",
        ),
        (
            "ROCM_CLI_DISABLE_STARTUP_UPDATE_CHECK",
            "When set, commands do not check for a newer build of the default ROCm runtime \
             as they start.",
        ),
        (
            INSTALL_DIR_ENV,
            "Read by the installer: the folder it installs rocm and rocmd into, in place of \
             ~/.local/bin.",
        ),
        (
            "ROCM_CLI_UPDATE_SHELL_PATH",
            "Read by the installer: set to 0 to leave shell profiles alone, which also skips \
             the fish completions. The manual page is installed either way.",
        ),
    ];
    let mut roff = Roff::new();
    roff.control("SH", ["ENVIRONMENT"]);
    for (name, what) in entries {
        roff.control("TP", []);
        roff.text([bold(name)]);
        roff.text([roman(what)]);
    }
    roff
}

/// The code clap exits with when it cannot parse the command line, asked of
/// clap rather than assumed.
fn usage_error_code() -> i32 {
    crate::cli_command()
        .try_get_matches_from(["rocm", "--no-such-option"])
        .map_or_else(|err| err.exit_code(), |_| 2)
}

fn exit_status_section() -> Roff {
    let usage = usage_error_code();
    let mut roff = Roff::new();
    roff.control("SH", ["EXIT STATUS"]);
    roff.control("TP", []);
    roff.text([bold("0")]);
    roff.text([roman("Success.")]);
    roff.control("TP", []);
    roff.text([bold("1")]);
    roff.text([roman(
        "The command failed; the reason is printed on standard error.",
    )]);
    roff.control("TP", []);
    roff.text([bold(usage.to_string())]);
    roff.text([roman(
        "The command line could not be parsed, such as an unknown option or a missing or \
         invalid argument.",
    )]);
    let others: Vec<String> = rocm_core::catalog_manifest()
        .exit_codes
        .into_iter()
        .filter(|(_, code)| ![0, 1, usage].contains(code))
        .map(|(name, code)| (code, name))
        .collect::<std::collections::BTreeMap<_, _>>()
        .into_iter()
        .map(|(code, name)| format!("{code} ({name})"))
        .collect();
    if !others.is_empty() {
        roff.control("PP", []);
        roff.text([
            bold("rocm fix"),
            roman(format!(" also exits {}; ", others.join(", "))),
            bold("rocm fix --json"),
            roman(" lists every code it uses under exit_codes."),
        ]);
    }
    roff
}

fn see_also_section() -> Roff {
    let mut roff = Roff::new();
    roff.control("SH", ["SEE ALSO"]);
    roff.text([
        bold("rocm"),
        roman(" "),
        italic("command"),
        roman(" "),
        bold("--help"),
    ]);
    roff.control("PP", []);
    roff.text([roman("https://github.com/ROCm/rocm-cli")]);
    roff
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory as _;

    fn page() -> String {
        let mut out = Vec::new();
        write(&mut out).expect("rendering into memory cannot fail");
        String::from_utf8(out).expect("the page is UTF-8")
    }

    /// The lines of the section headed `name`, up to the next heading.
    fn section(page: &str, name: &str) -> Vec<String> {
        let heading = format!(".SH {name}");
        let heading_quoted = format!(".SH \"{name}\"");
        page.lines()
            .skip_while(|line| *line != heading && *line != heading_quoted)
            .skip(1)
            .take_while(|line| !line.starts_with(".SH "))
            .map(str::to_owned)
            .collect()
    }

    /// The term of every `.TP` entry in `lines`.
    fn terms(lines: &[String]) -> Vec<String> {
        lines
            .windows(2)
            .filter(|pair| pair[0] == ".TP")
            .map(|pair| {
                pair[1]
                    .replace("\\fB", "")
                    .replace("\\fR", "")
                    .replace("\\-", "-")
            })
            .collect()
    }

    /// COMMANDS lists exactly the top-level commands `rocm --help` lists
    /// (bar clap's own `help`), so no internal verb such as `__man-page`
    /// itself appears and no visible one is missing.
    #[test]
    fn the_command_list_is_the_visible_top_level_commands() {
        let mut visible: Vec<String> = crate::Cli::command()
            .get_subcommands()
            .filter(|sc| !sc.is_hide_set())
            .map(|sc| sc.get_name().to_owned())
            .collect();
        visible.sort();
        let listed = terms(&section(&page(), "COMMANDS"));
        assert_eq!(listed, visible);
        assert!(!listed.iter().any(|name| name == "__man-page"));
    }

    /// Every example is a command line `rocm` accepts.
    #[test]
    fn every_example_parses() {
        for (args, _) in EXAMPLES {
            let argv = std::iter::once("rocm").chain(args.iter().copied());
            if let Err(err) = crate::cli_command().try_get_matches_from(argv) {
                panic!("example `rocm {}` does not parse: {err}", args.join(" "));
            }
        }
        let shown = terms(&section(&page(), "EXAMPLES"));
        assert_eq!(shown.len(), EXAMPLES.len(), "{shown:?}");
    }

    /// The variables that move the config, data and cache folders are the
    /// ones path discovery reads.
    #[test]
    fn the_environment_section_names_each_folder_override() {
        let names = terms(&section(&page(), "ENVIRONMENT"));
        for var in [CONFIG_DIR_ENV, DATA_DIR_ENV, CACHE_DIR_ENV] {
            assert!(
                names.iter().any(|name| name == var),
                "{var} missing: {names:?}"
            );
        }
    }

    /// The usage-error code is the one clap exits with, and `rocm fix`'s own
    /// codes are the ones it publishes.
    #[test]
    fn the_exit_status_section_names_the_codes_rocm_exits_with() {
        let clap_code = crate::cli_command()
            .try_get_matches_from(["rocm", "examine", "--bogus"])
            .expect_err("an unknown flag is refused")
            .exit_code();
        let lines = section(&page(), "EXIT STATUS");
        let codes = terms(&lines);
        assert_eq!(codes, ["0", "1", &clap_code.to_string()], "{lines:?}");
        let text = lines.join("\n");
        for (name, code) in rocm_core::catalog_manifest().exit_codes {
            if ![0, 1, clap_code].contains(&code) {
                assert!(
                    text.contains(&format!("{code} ({name})")),
                    "{name} missing: {text}"
                );
            }
        }
    }

    /// The page refers to no other `rocm-*` page, since only rocm(1) ships, and
    /// carries none of the sections the whole-page render adds.
    #[test]
    fn the_page_refers_to_no_other_manual_page() {
        let page = page();
        assert!(!page.contains("(1)"), "{page}");
        for heading in [".SH SUBCOMMANDS", ".SH EXTRA"] {
            assert!(!page.contains(heading), "{page}");
        }
    }

    /// The page explains the `[preview]` marker its command list shows.
    #[test]
    fn the_page_says_what_preview_means() {
        let commands = section(&page(), "COMMANDS").join(" ");
        assert!(commands.contains("[preview]"), "{commands}");
        assert!(
            commands.contains("Commands marked [preview] work but are outside"),
            "{commands}"
        );
    }

    /// The title is the program's, and the version is the one `--version`
    /// prints.
    #[test]
    fn the_title_carries_the_version_rocm_reports() {
        let page = page();
        let title = page
            .lines()
            .find(|line| line.starts_with(".TH "))
            .expect("a title line");
        // Title, section, an empty date, then source and manual, each in
        // its own field.
        assert!(title.starts_with(".TH ROCM 1 \"\" \"rocm "), "{title}");
        assert!(title.ends_with("\" \"ROCm CLI\""), "{title}");
        let version = crate::cli_version_string();
        assert!(title.contains(&version), "{title} does not carry {version}");
    }
}
