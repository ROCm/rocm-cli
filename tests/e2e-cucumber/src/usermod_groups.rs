// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The groups a `usermod` line in a diagnosis tells the user to join.
//!
//! Read as a list, never as text, so a scenario can ask whether each group
//! exists on the machine. An empty list is returned for a line it cannot read,
//! never a guess: a caller comparing two empty lists, or checking that every
//! element of one is known, would pass while measuring nothing, so each caller
//! rejects an empty result.

/// The first `usermod` command line in `text` that sets supplementary groups,
/// without any `$ ` prompt in front of it.
#[must_use]
pub fn usermod_line(text: &str) -> Option<&str> {
    text.lines()
        .map(str::trim)
        .find(|line| line.contains("usermod") && groups_argument(line).is_some())
        .map(|line| line.trim_start_matches("$ "))
}

/// The group names a `usermod` command line adds the user to.
///
/// Reads every spelling of the option: `-G a,b`, `--groups a,b`,
/// `--groups=a,b`, and a bundled `-aG a,b` or `-aGa,b`.
#[must_use]
pub fn groups_in_usermod_command(command: &str) -> Vec<String> {
    groups_argument(command)
        .map(|names| {
            names
                .trim_matches('"')
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// The raw comma-separated list the group option carries, when there is one.
fn groups_argument(command: &str) -> Option<&str> {
    let mut tokens = command.split_whitespace();
    while let Some(token) = tokens.next() {
        if token == "--groups" {
            return tokens.next();
        }
        if let Some(rest) = token.strip_prefix("--groups=") {
            return Some(rest);
        }
        // Short options only: `-G`, or a bundle such as `-aG`. The list is the
        // rest of the bundle when it carries one, else the next token.
        if token.starts_with('-')
            && !token.starts_with("--")
            && let Some(at) = token.find('G')
        {
            let rest = &token[at + 1..];
            return if rest.is_empty() {
                tokens.next()
            } else {
                Some(rest)
            };
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_spelling_of_the_option_is_read() {
        for command in [
            "sudo usermod -a -G render,video \"$USER\"",
            "sudo usermod -aG render,video \"$USER\"",
            "sudo usermod -aGrender,video \"$USER\"",
            "sudo usermod --groups render,video \"$USER\"",
            "sudo usermod -a --groups=render,video \"$USER\"",
        ] {
            assert_eq!(
                groups_in_usermod_command(command),
                ["render", "video"],
                "{command}"
            );
        }
    }

    #[test]
    fn a_line_with_no_group_option_gives_no_groups_rather_than_a_guess() {
        assert!(groups_in_usermod_command("sudo usermod -a \"$USER\"").is_empty());
        assert!(groups_in_usermod_command("sudo usermod -G").is_empty());
    }

    #[test]
    fn the_line_is_found_under_a_prompt_and_among_other_text() {
        let text = "  #1 (fix-4-render-group)\n    $ sudo usermod -a -G render,video \"$USER\"\n";
        assert_eq!(
            usermod_line(text),
            Some("sudo usermod -a -G render,video \"$USER\"")
        );
        assert_eq!(usermod_line("nothing to run here\n"), None);
    }

    #[test]
    fn a_usermod_mention_without_a_group_option_is_not_the_remedy() {
        assert_eq!(usermod_line("explains why usermod is needed\n"), None);
    }
}
