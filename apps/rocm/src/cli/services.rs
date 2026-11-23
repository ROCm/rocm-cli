// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm services`' subcommand enum, `ServicesCommand`, moved out of
//! [`super`] unchanged to keep that file under the module-size limit; `cli.rs`
//! re-exports it, so it is still reached as `crate::cli::ServicesCommand`. Its
//! doc comments are `--help` output, kept in step with README.md,
//! docs/testing.md and docs/manual-testing.md like the rest of `cli.rs`.

use crate::service_records::DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS;
use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub(crate) enum ServicesCommand {
    /// Show currently running local model servers, and count the records that
    /// are no longer running.
    ///
    /// Each running server gets a `docs:` line naming the command that serves
    /// its API docs (`rocm services docs <id>`); a vLLM server also gets an
    /// `engine docs:` line, the docs vLLM serves itself.
    List {
        /// Include failed, stopped, and old service records.
        #[arg(short, long)]
        all: bool,
        /// Emit the service records as JSON instead of a table.
        ///
        /// This is the machine-readable form `rocm remote` reads back over its
        /// control channel to discover which service a remote `rocm serve` just
        /// started, rather than scraping the human table.
        #[arg(long)]
        json: bool,
    },
    /// Serve browsable API docs for running local model servers until Ctrl+C.
    ///
    /// Serves a loopback-only page (Swagger UI) per running server listing the
    /// routes it answers, with a working "Try it out" that sends requests
    /// through the page to the server. Prints the page links, then serves them
    /// until Ctrl+C; nothing keeps running afterwards and the model servers
    /// are not touched. Refuses when the named server is not running, or when
    /// no server is.
    Docs {
        /// Service id from `rocm services list`; without it, every running
        /// server's docs are linked.
        service_id: Option<String>,
        /// Port to serve the docs on (127.0.0.1 only). Without it, 11430 is
        /// used, or a free port when 11430 is taken.
        #[arg(long)]
        port: Option<u16>,
    },
    /// Show logs for a local model server.
    Logs {
        /// Service id from `rocm services list`.
        service_id: String,
        /// Keep printing new output until the server stops (Ctrl-C ends the follow and
        /// leaves the server running). Use it to watch a model download.
        #[arg(long, short = 'f')]
        follow: bool,
    },
    /// Stop a local model server.
    ///
    /// Reports the server stopped only once every process it recorded is
    /// confirmed gone. Otherwise it reports on stderr that the stop was not
    /// confirmed and exits with an error, keeping the record, which still names
    /// the processes, and any endpoint key, which the server may still be
    /// enforcing. It names each process that did not exit (on Windows, also one
    /// whose child processes may still be running), and for one it could not
    /// verify it signals nothing and names an operating-system command to check
    /// on it. It does not suggest running the stop again.
    Stop {
        /// Service id from `rocm services list`.
        service_id: String,
        /// Required; stop refuses without it.
        #[arg(long)]
        yes: bool,
    },
    /// Restart a local model server.
    ///
    /// Stops the server first, and starts it again only once every process it
    /// recorded is confirmed gone. If the stop cannot confirm that, nothing is
    /// started and the stop request stays recorded on the server.
    Restart {
        /// Service id from `rocm services list --all`.
        service_id: String,
        /// Required; restart refuses without it.
        #[arg(long)]
        yes: bool,
    },
    /// Delete one local server record and its files.
    ///
    /// Only works on a record that is not running: stop the server first (on
    /// Windows a process `rocm` cannot query counts as running). The record's
    /// details file, log, engine state file, and endpoint key file (with any
    /// copy of the key an interrupted save left beside it) are all deleted, so
    /// its log can no longer be read and it can no longer be restarted.
    Remove {
        /// Service id from `rocm services list --all`.
        service_id: String,
        /// Required; remove refuses without it.
        #[arg(long)]
        yes: bool,
        /// Remove a record whose recorded process `rocm` cannot query.
        ///
        /// Only when a stop was requested, no recorded process is confirmed
        /// running, and at least one could not be queried (on Windows, a live
        /// process the account may not open). Deletes the same files as a
        /// plain `remove`, and says which process; if it is still running it
        /// keeps running. Refused for a process confirmed running.
        #[arg(long)]
        force: bool,
    },
    /// Delete local server records that are no longer running.
    ///
    /// Running servers are always left alone; a record held only by a process
    /// `rocm` cannot query (Windows) is listed with the `remove --force` command.
    /// Leftover files whose record is already gone are cleaned up too.
    ///
    /// Waits for a managed launch already under way to publish its record
    /// before reading the directory, and waits as long as that launch takes.
    /// There is no timeout. On an interactive terminal it says so while it
    /// waits; piped or redirected output stays silent.
    Prune {
        /// Only remove records and files untouched for at least this many hours.
        ///
        /// Age is measured from when the record file was last written - a stop,
        /// a restart, or a status correction all count as touching it - so a
        /// server that has only just stopped keeps its log and stays
        /// restartable. Pass 0 to include everything that is not running.
        #[arg(long, default_value_t = DEFAULT_SERVICE_PRUNE_MIN_AGE_HOURS)]
        older_than_hours: u64,
        /// Remove every record that is not running, however recent.
        ///
        /// The same thing as `--older-than-hours 0`, named so it can be reached
        /// without knowing the age rule exists -- which is how most people will
        /// arrive here, after the summary tells them recent records were kept.
        #[arg(long, conflicts_with = "older_than_hours")]
        any_age: bool,
        /// Show what would be removed, without removing anything.
        #[arg(long)]
        dry_run: bool,
        /// Required unless --dry-run; prune refuses without it.
        #[arg(long)]
        yes: bool,
    },
}

#[cfg(test)]
mod tests {
    use crate::cli::Cli;
    use clap::Parser;

    /// `--any-age` and an explicit `--older-than-hours` would be two answers to
    /// one question; clap has to reject the pair rather than silently pick one.
    #[test]
    fn services_prune_rejects_any_age_with_an_explicit_age() {
        let error = Cli::try_parse_from([
            "rocm",
            "services",
            "prune",
            "--any-age",
            "--older-than-hours",
            "5",
        ])
        .expect_err("the two age arguments must conflict");
        let rendered = error.to_string();
        assert!(
            rendered.contains("--any-age") && rendered.contains("--older-than-hours"),
            "the conflict must name both arguments:\n{rendered}"
        );
    }
}
