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

/// Delete the instance directories of `stale` agents under `root`, re-reading
/// each record first so an agent that started running again is never
/// deleted. Returns how many were deleted.
pub(crate) fn delete_stopped(root: &Path, stale: &[&AgentRecord]) -> Result<usize> {
    let mut deleted = 0;
    for record in stale {
        // The id names a directory under `root` only: refuse anything that
        // could step outside it.
        if record.id.is_empty() || record.id.contains(['/', '\\']) || record.id.starts_with('.') {
            continue;
        }
        let dir = root.join(&record.id);
        let current = std::fs::read(dir.join("record.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<AgentRecord>(&bytes).ok());
        let still_stopped = current
            .is_some_and(|r| r.status != AgentStatus::Running || !pid_alive(r.supervisor_pid));
        if still_stopped {
            std::fs::remove_dir_all(&dir)?;
            deleted += 1;
        }
    }
    Ok(deleted)
}

fn pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, status: AgentStatus, pid: u32) -> AgentRecord {
        AgentRecord {
            id: id.to_owned(),
            agent: "shell".to_owned(),
            workspace: PathBuf::from("/w"),
            supervisor_pid: pid,
            started_unix: 1,
            ended_unix: None,
            proxy: String::new(),
            upstream: String::new(),
            requests: 0,
            rejected: 0,
            status,
            exit_code: None,
        }
    }

    #[test]
    fn prune_deletes_only_stopped_agents_inside_the_root() {
        let root = std::env::temp_dir().join(format!("rocm-agents-prune-{}", std::process::id()));
        let alive = std::process::id();
        let gone = u32::MAX - 1;
        for r in [
            record("exited-1", AgentStatus::Exited, gone),
            record("running-1", AgentStatus::Running, alive),
            record("lost-1", AgentStatus::Running, gone),
        ] {
            std::fs::create_dir_all(root.join(&r.id)).unwrap();
            save(&root.join(&r.id).join("record.json"), &r).unwrap();
        }
        // A sibling outside the root must survive an id that tries to escape.
        let outside = root.with_extension("outside");
        std::fs::create_dir_all(&outside).unwrap();
        let escape = record("../rocm-agents-prune-escape", AgentStatus::Exited, gone);

        let all = load_all(&root).unwrap();
        // Hand prune everything, the running agent included: the re-read keeps it.
        let mut candidates: Vec<&AgentRecord> = all.iter().collect();
        candidates.push(&escape);
        assert_eq!(delete_stopped(&root, &candidates).unwrap(), 2);
        assert!(!root.join("exited-1").exists());
        assert!(!root.join("lost-1").exists());
        assert!(root.join("running-1").exists());
        assert!(outside.exists());

        std::fs::remove_dir_all(&root).unwrap();
        std::fs::remove_dir_all(&outside).unwrap();
    }
}
