// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Readers for what the CLI's help promises.
//!
//! Kept apart from the steps that run the CLI so a table test can drive each one
//! against help text of either shape clap renders: wrapped at a terminal width,
//! or not wrapped at all.

/// The model each `rocm serve <model>` example in `help` names, in order.
///
/// A worked example is copied verbatim, so the name in it is the one that has to
/// work. The generic placeholders the help uses to describe the FORM of a model
/// reference (`<model>`) are not names anyone is expected to run.
///
/// Reads only a line that begins with `rocm serve `, and takes the first word
/// after it: an example written inside a sentence, or wrapped so that the name
/// falls on the next line, is not read. A help text that stopped showing its
/// examples that way yields none, which the caller must treat as a failure.
#[must_use]
pub fn serve_example_models(help: &str) -> Vec<String> {
    help.lines()
        .filter_map(|line| line.trim().strip_prefix("rocm serve "))
        .filter_map(|rest| rest.split_whitespace().next())
        .filter(|model| !model.starts_with('<'))
        .map(str::to_owned)
        .collect()
}

/// Every name `rocm model --verbose` answers to: each recipe's canonical id and
/// its aliases, read from the `<id> aliases=[a, b] ...` header of each recipe.
#[must_use]
pub fn known_model_names(listing: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in listing.lines() {
        let Some((head, rest)) = line.trim().split_once(" aliases=[") else {
            continue;
        };
        names.push(head.trim().to_owned());
        if let Some((aliases, _)) = rest.split_once(']') {
            names.extend(
                aliases
                    .split(',')
                    .map(str::trim)
                    .filter(|alias| !alias.is_empty())
                    .map(str::to_owned),
            );
        }
    }
    names
}

/// The one-line summary the command list gives `name`, if the help lists it.
///
/// Takes the first line that starts with `name` and a space, so it is the
/// command's own line only while the command list is the first place the help
/// starts a line with a command name. The list's order is pinned by `examine-03`
/// (`features/examine.feature`), and `examine-70` reads it through this function.
#[must_use]
pub fn command_summary(help: &str, name: &str) -> Option<String> {
    help.lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix(name)?.strip_prefix(' '))
        .map(|summary| summary.trim().to_owned())
        .filter(|summary| !summary.is_empty())
}

/// The sentence of `help` that contains `phrase`, with any line wrap undone.
///
/// clap wraps a long description at the terminal width, so the sentence can
/// straddle a line break, and reading a single line would let a wording that
/// matters slip through whenever the wrap fell between two of its words. The
/// sentence is taken from the paragraph (a block bounded by blank lines) that
/// holds the phrase, which is joined into one line and cut at the full stops
/// around it. A full stop followed by something other than a space, as in a file
/// name, does not end a sentence.
#[must_use]
pub fn sentence_containing(help: &str, phrase: &str) -> Option<String> {
    let paragraph = help
        .split("\n\n")
        .find(|block| join_wrapped(block).contains(phrase))?;
    let joined = join_wrapped(paragraph);
    let at = joined.find(phrase)?;
    let start = joined[..at].rfind(". ").map_or(0, |end| end + 2);
    let end = joined[at..]
        .find(". ")
        .map_or(joined.len(), |stop| at + stop + 1);
    Some(joined[start..end].trim().to_owned())
}

fn join_wrapped(block: &str) -> String {
    block.lines().map(str::trim).collect::<Vec<_>>().join(" ")
}

/// A help line that starts a new option: `--long` or `-x, --long`.
fn is_option_line(line: &str) -> bool {
    let line = line.trim_start();
    line.starts_with("--")
        || (line.len() > 1 && line.starts_with('-') && line.as_bytes()[1].is_ascii_alphanumeric())
}

/// The values `help` advertises for the option `--{option}`.
///
/// Read from that option's own block rather than from the first list of possible
/// values in the help: `serve --help` advertises values for more than one option,
/// so taking the first list would return another option's.
///
/// Both of clap's renderings are read, because which one an option gets depends
/// on how it happens to be declared, not on any contract:
///
/// * inline: `--device <DEVICE>  Device policy [possible values: a, b]`
/// * block: a `Possible values:` line followed by one `- name: description`
///   entry per value, which is what clap emits for a documented `ValueEnum`
///
/// An option that advertises nothing, or is not in the help, gives an empty list
/// and never borrows the next option's. Returning nothing does not mean the
/// command offers nothing, so a caller treats an empty result as a read failure.
#[must_use]
pub fn advertised_values(help: &str, option: &str) -> Vec<String> {
    let flag = format!("--{option}");
    let lines: Vec<&str> = help.lines().collect();
    let Some(start) = lines.iter().position(|line| {
        line.trim_start()
            .strip_prefix(flag.as_str())
            .is_some_and(|rest| rest.is_empty() || rest.starts_with([' ', '=', '<']))
    }) else {
        return Vec::new();
    };
    // The option's own block: everything up to the next option.
    let block: Vec<&str> = lines
        .iter()
        .skip(start)
        .enumerate()
        .take_while(|(offset, line)| *offset == 0 || !is_option_line(line))
        .map(|(_, line)| *line)
        .collect();

    let inline = block.iter().find_map(|line| {
        let (_, rest) = line.split_once("[possible values:")?;
        let (values, _) = rest.split_once(']')?;
        Some(
            values
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>(),
        )
    });
    if let Some(values) = inline
        && !values.is_empty()
    {
        return values;
    }

    let Some(heading) = block
        .iter()
        .position(|line| line.trim().eq_ignore_ascii_case("possible values:"))
    else {
        return Vec::new();
    };
    block
        .iter()
        .skip(heading + 1)
        // `filter_map`, not `map_while`: a long description wraps onto a
        // continuation line, and stopping at the first line that is not an entry
        // would drop every value after it.
        .filter_map(|line| {
            let entry = line.trim_start().strip_prefix("- ")?;
            // `- <name>: <description>`; only the first colon separates the two.
            let name = entry.split_once(':').map_or(entry, |(name, _)| name);
            Some(name.trim().to_owned())
        })
        .filter(|value| !value.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_examples_are_read_from_each_block_and_placeholders_are_skipped() {
        let help = "EXAMPLES:\n  rocm serve qwen                   Start a server\n  \
                    rocm serve <model> --engine vllm\n  rocm serve org/repo --port 8000\n  \
                    rocm examine\n";
        assert_eq!(serve_example_models(help), ["qwen", "org/repo"]);
        assert!(serve_example_models("nothing to serve here").is_empty());
    }

    #[test]
    fn a_listing_gives_each_recipe_id_and_each_alias() {
        let listing = "model recipes\n  Org/Model-1B aliases=[one, two] task=chat\n      \
                       catalog: hidden\n  Other aliases=[] task=chat\n";
        assert_eq!(
            known_model_names(listing),
            ["Org/Model-1B", "one", "two", "Other"]
        );
        assert!(known_model_names("no recipes here").is_empty());
    }

    #[test]
    fn a_command_summary_is_the_text_after_its_name() {
        let help = "Commands:\n  dash         Launch the unified telemetry dashboard (TUI)\n  \
                    dashboard-x  Not this one\n";
        assert_eq!(
            command_summary(help, "dash").as_deref(),
            Some("Launch the unified telemetry dashboard (TUI)")
        );
        assert_eq!(command_summary(help, "missing"), None);
    }

    #[test]
    fn a_sentence_is_found_whole_when_the_wrap_falls_inside_it() {
        let wrapped = "Intro.\n\nRun `rocm` with no subcommand to open the\n\
                       interactive launcher (TUI). Use `rocm examine` to check.\n\nOutro.\n";
        assert_eq!(
            sentence_containing(wrapped, "with no subcommand").as_deref(),
            Some("Run `rocm` with no subcommand to open the interactive launcher (TUI).")
        );
    }

    #[test]
    fn a_sentence_does_not_borrow_its_neighbours() {
        let text = "Open the dashboard with `rocm dash`. Run `rocm` with no subcommand to open \
                    the launcher. Then serve.\n";
        let sentence = sentence_containing(text, "with no subcommand").expect("found");
        assert_eq!(
            sentence,
            "Run `rocm` with no subcommand to open the launcher."
        );
        assert!(!sentence.contains("dashboard"));
    }

    #[test]
    fn a_phrase_the_help_does_not_contain_gives_no_sentence() {
        assert_eq!(
            sentence_containing("Some other text.\n", "no subcommand"),
            None
        );
    }

    const ENGINE_THEN_DEVICE_BLOCK: &str = "\
      --engine <ENGINE>
          Engine to use

          [possible values: lemonade, vllm, aim]

      --device <DEVICE>
          Device policy

          Possible values:
          - gpu_required:  Require a ROCm GPU; fail if none is usable
          - gpu_preferred: Accepted for compatibility; behaves like `gpu_required`
            (a description that wraps onto a second line)
          - third: has a colon: in its description

      --gpu <INDEX|auto>
          GPU device
";

    #[test]
    fn an_options_block_form_values_are_read_across_a_wrapped_description() {
        assert_eq!(
            advertised_values(ENGINE_THEN_DEVICE_BLOCK, "device"),
            ["gpu_required", "gpu_preferred", "third"]
        );
    }

    #[test]
    fn another_options_list_is_not_taken_for_this_ones() {
        assert_eq!(
            advertised_values(ENGINE_THEN_DEVICE_BLOCK, "engine"),
            ["lemonade", "vllm", "aim"]
        );
    }

    #[test]
    fn the_inline_form_is_read() {
        let help =
            "      --device <DEVICE>  Device policy [possible values: a, b]\n      --gpu <N>\n";
        assert_eq!(advertised_values(help, "device"), ["a", "b"]);
    }

    #[test]
    fn an_option_that_advertises_nothing_does_not_borrow_the_next_ones() {
        let help = "      --device <DEVICE>\n          Device policy\n\n      --engine <E>\n          \
                    [possible values: lemonade]\n";
        assert!(advertised_values(help, "device").is_empty());
        assert!(advertised_values(help, "missing").is_empty());
    }

    #[test]
    fn a_longer_option_name_is_not_mistaken_for_the_one_asked_for() {
        let help = "      --device-index <N>\n          [possible values: x]\n";
        assert!(advertised_values(help, "device").is_empty());
    }
}
