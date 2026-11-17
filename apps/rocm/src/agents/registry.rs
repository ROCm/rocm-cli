// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! One `record.json` per agent instance (PROTOTYPE fleet view).

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AgentStatus {
    Running,
    Exited,
    /// Recorded as running, but its supervising `rocm` process is gone.
    Lost,
}

impl AgentStatus {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Exited => "exited",
            Self::Lost => "lost",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AgentRecord {
    pub id: String,
    pub agent: String,
    pub workspace: PathBuf,
    pub supervisor_pid: u32,
    pub started_unix: u64,
    #[serde(default)]
    pub ended_unix: Option<u64>,
    pub proxy: String,
    pub upstream: String,
    pub requests: u64,
    pub rejected: u64,
    pub status: AgentStatus,
    pub exit_code: Option<i32>,
}

pub(crate) fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

pub(crate) fn save(path: &Path, record: &AgentRecord) -> Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(record)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

pub(crate) fn load_all(root: &Path) -> Result<Vec<AgentRecord>> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return Ok(out);
    };
    for entry in entries.flatten() {
        let Ok(bytes) = std::fs::read(entry.path().join("record.json")) else {
            continue;
        };
        let Ok(mut rec) = serde_json::from_slice::<AgentRecord>(&bytes) else {
            continue;
        };
        if rec.status == AgentStatus::Running && !pid_alive(rec.supervisor_pid) {
            rec.status = AgentStatus::Lost;
        }
        out.push(rec);
    }
    out.sort_by_key(|r| std::cmp::Reverse(r.started_unix));
    Ok(out)
}

fn pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}
