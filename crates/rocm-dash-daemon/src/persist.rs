// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Append-only NDJSON session writer.
//!
//! Each daemon session opens exactly one file under `--persist-dir`, named
//! `session-YYYYMMDD-HHMMSS.ndjson`. Every broadcast [`Event`] is written as
//! a [`PersistedEntry`] line with a wallclock timestamp. The TUI's
//! `--replay <path>` mode reads these back.
//!
//! Buffered + line-flushed: under power loss we lose at most the in-flight
//! tick. The writer is intentionally simple — no rotation, no compression.
//! One file per session means each file is self-contained and trivially
//! shareable / portable.

use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use chrono::Utc;
use rocm_dash_core::persist::PersistedEntry;
use rocm_dash_core::protocol::Event;

pub struct SessionWriter {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl SessionWriter {
    /// Create a fresh session file in `dir`. The directory is created if it
    /// does not exist. Returns an error if the file cannot be created.
    pub fn new(dir: &Path) -> std::io::Result<Self> {
        create_dir_all(dir)?;
        let stamp = Utc::now().format("%Y%m%d-%H%M%S").to_string();
        let path = dir.join(format!("session-{stamp}.ndjson"));
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Stamp `event` with the current time and append as a single NDJSON line.
    /// Flushes after each write so the file survives a crash.
    pub fn append(&mut self, event: &Event) -> std::io::Result<()> {
        let entry = PersistedEntry::now(event.clone());
        let line = serde_json::to_string(&entry).map_err(std::io::Error::other)?;
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};

    fn tmp_dir(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "rocm-dash-persist-test-{}-{}-{label}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    #[test]
    fn creates_a_session_file_under_dir() {
        let dir = tmp_dir("new");
        let writer = SessionWriter::new(&dir).expect("create");
        assert!(writer.path().exists());
        assert!(
            writer
                .path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("session-")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn appends_serializable_lines() {
        let dir = tmp_dir("append");
        let mut writer = SessionWriter::new(&dir).expect("create");
        let ev1 = Event::Welcome {
            protocol_version: 1,
            daemon_version: "0.0.0".into(),
            host: "h".into(),
        };
        let ev2 = Event::Bye;
        writer.append(&ev1).expect("append 1");
        writer.append(&ev2).expect("append 2");

        let path = writer.path().to_path_buf();
        drop(writer);
        let file = File::open(&path).unwrap();
        let lines: Vec<String> = BufReader::new(file).lines().map(Result::unwrap).collect();
        assert_eq!(lines.len(), 2);
        let e1: PersistedEntry = serde_json::from_str(&lines[0]).unwrap();
        let e2: PersistedEntry = serde_json::from_str(&lines[1]).unwrap();
        assert!(matches!(e1.event, Event::Welcome { .. }));
        assert!(matches!(e2.event, Event::Bye));
        assert!(e2.ts_us >= e1.ts_us);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
