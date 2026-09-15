// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Local records of work running on other machines.
//!
//! A session is the pairing of two things on the remote host: a managed model
//! server, and a tailnet publish pointing at it. Neither lives here — this is
//! only the note-to-self that lets a later `rocm remote status` or
//! `rocm remote stop`, run from a different shell or after a reboot, find them
//! again.
//!
//! One JSON file per session, mirroring the managed-service registry's shape.
//! The record is a point-in-time snapshot, never a source of truth about
//! liveness: `status` re-probes the remote rather than trusting what was
//! written here, because both halves can disappear without anyone updating a
//! local file.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, unix_time_millis};
use serde::{Deserialize, Serialize};

/// A model served on a remote machine and published to the tailnet.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct RemoteSessionRecord {
    /// Stable identifier, also the record's filename.
    pub(crate) session_id: String,
    /// The target exactly as the user typed it, so messages echo their words
    /// back rather than a resolved name they never used.
    pub(crate) target: String,
    /// The peer's MagicDNS name — what the endpoint URL is built from.
    pub(crate) peer_host: String,
    /// Explicit SSH port for the control channel, when one was given.
    #[serde(default)]
    pub(crate) ssh_port: Option<u16>,
    pub(crate) model: String,
    /// Id of the managed service on the remote's own registry.
    pub(crate) remote_service_id: String,
    /// How to invoke the CLI on the remote (may be a path, not just `rocm`).
    pub(crate) remote_cli: String,
    /// Loopback port the model server is bound to *on the remote*.
    pub(crate) remote_port: u16,
    /// Port the remote publishes to the tailnet.
    pub(crate) tailnet_port: u16,
    /// The address a user calls, including the OpenAI-compatible path.
    pub(crate) base_url: String,
    pub(crate) created_at_unix_ms: u128,
}

impl RemoteSessionRecord {
    /// Identifier for a session, derived from the peer and the port it serves on
    /// rather than randomly.
    ///
    /// Deterministic on purpose: re-running `serve` against the same machine and
    /// port resolves to the same record instead of accumulating a new one per
    /// invocation, which is what turns a repeated command into an idempotent
    /// action rather than a leak.
    pub(crate) fn id_for(peer_host: &str, remote_port: u16) -> String {
        let host = peer_host
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || character == '-' {
                    character.to_ascii_lowercase()
                } else {
                    // Anything else becomes a dash: the id is used as a filename,
                    // and a MagicDNS name carries dots while a raw IPv6 address
                    // carries colons, which Windows rejects outright.
                    '-'
                }
            })
            .collect::<String>();
        let host = host.trim_matches('-');
        format!("remote-{host}-{remote_port}")
    }

    /// Check an id against the rules the rest of the CLI uses for anything that
    /// becomes a path segment.
    ///
    /// [`Self::id_for`] only ever produces safe ids, but a record is also read
    /// back from disk, and deserializing does not re-run that construction. An
    /// id carrying `..` or a separator would make [`Self::path`] resolve outside
    /// the sessions directory — and `stop` deletes that path.
    ///
    /// Deliberately delegating to [`rocm_core::ServiceId`] rather than spelling
    /// the rules again here. It is the type the endpoint-key path builder
    /// documents as its contract, this record's id is passed straight to that
    /// builder, and a second hand-written rule set is exactly how the two drift
    /// into disagreeing about what is safe.
    fn validate_id(session_id: &str) -> Result<()> {
        rocm_core::ServiceId::new(session_id)
            .map(|_| ())
            .with_context(|| format!("unusable remote session id `{session_id}`"))
    }

    pub(crate) fn path_in(paths: &AppPaths, session_id: &str) -> PathBuf {
        paths
            .remote_sessions_dir()
            .join(format!("{session_id}.json"))
    }

    pub(crate) fn path(&self, paths: &AppPaths) -> PathBuf {
        Self::path_in(paths, &self.session_id)
    }

    pub(crate) fn write(&self, paths: &AppPaths) -> Result<()> {
        let directory = paths.remote_sessions_dir();
        fs::create_dir_all(&directory)
            .with_context(|| format!("failed to create {}", directory.display()))?;
        let path = self.path(paths);
        let bytes =
            serde_json::to_vec_pretty(self).context("failed to serialize the remote session")?;

        // Write beside the record and rename over it. A plain write is not
        // atomic, so two serves racing on the same machine and port could leave
        // a half-written file — which `load_all` then skips, hiding a session
        // whose model is running and whose endpoint is published. Rename is
        // atomic on both platforms we target, so a reader sees either the old
        // record or the new one.
        // Unique per writer. A single shared `<id>.json.tmp` just moves the
        // race: two serves for the same machine and port would write the same
        // staging file and rename each other's half-written bytes into place.
        let staging = staging_path_for(&path);
        fs::write(&staging, &bytes)
            .with_context(|| format!("failed to write {}", staging.display()))?;
        fs::rename(&staging, &path).with_context(|| {
            let _ = fs::remove_file(&staging);
            format!("failed to replace {}", path.display())
        })
    }

    /// Forget this session locally. Best-effort and idempotent, so a teardown
    /// that has already removed it can call this without special-casing.
    pub(crate) fn remove(&self, paths: &AppPaths) {
        let _ = fs::remove_file(self.path(paths));
    }

    /// Timestamp helper so callers do not each reach for the clock.
    pub(crate) fn now() -> u128 {
        unix_time_millis()
    }
}

/// Whether anything is already filed under this session id.
///
/// The id is derived from the machine and the port, not minted per attempt, so
/// two `serve` runs against the same machine and port compute the same one. That
/// makes the id a *name*, not a claim.
///
/// This answers a question; it does not take the name. Only [`store_key`] does
/// that, and it has to, because the gap between asking and writing is where two
/// runs both get "free" for an answer — a gap that spans the whole readiness
/// probe on the serve path, which is minutes when it provisions a CLI. So this
/// is for *diagnosis*: it runs first because it is cheap and because it can tell
/// a recorded session from a stray credential, and each of those needs a
/// different remedy. The claim itself comes later and is what actually decides.
///
/// The key is checked as well as the record, because they are written at
/// different moments: a session that failed between minting its key and writing
/// its record leaves the key alone on disk.
pub(crate) fn exists(paths: &AppPaths, session_id: &str) -> bool {
    RemoteSessionRecord::path_in(paths, session_id).exists() || key_path(paths, session_id).exists()
}

/// A staging path no other writer will pick, for the write-then-rename above.
///
/// The pid separates processes and the timestamp separates most writes, but two
/// threads in one process can reach the same millisecond — and a shared staging
/// file means each renames the other's half-written bytes into place, which is
/// the very race the rename exists to prevent. The counter closes that: it is
/// per-process and monotonic, so no two calls here can agree.
fn staging_path_for(path: &std::path::Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    path.with_extension(format!(
        "{}.{}.{}.tmp",
        std::process::id(),
        unix_time_millis(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Where a session's endpoint credential is kept.
///
/// Beside the session record, not in the managed-service registry. The endpoint
/// key helpers used for local serving build their path under `services_dir()`
/// from a *service* id, and a remote session is not a local service — it is a
/// different registry with its own ids. Borrowing that directory made two
/// namespaces share one folder and made a remote session's id masquerade as a
/// service id, which is a collision waiting to happen and reads wrong to anyone
/// inspecting either registry.
pub(crate) fn key_path(paths: &AppPaths, session_id: &str) -> PathBuf {
    paths
        .remote_sessions_dir()
        .join(format!("{session_id}.endpoint-key"))
}

/// Returned by [`store_key`] when the session name is already held.
///
/// A distinct type rather than a message, because the caller has to tell it
/// apart from a full disk or a permission problem: one means somebody else owns
/// this session and we must leave every file under it alone, the other means our
/// own write failed and there is nothing of ours on disk to protect.
#[derive(Debug)]
pub(crate) struct NameAlreadyHeld {
    pub(crate) path: PathBuf,
}

impl std::fmt::Display for NameAlreadyHeld {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "the session name is already held by {}",
            self.path.display()
        )
    }
}

impl std::error::Error for NameAlreadyHeld {}

/// Claim the session name by creating the credential that guards its endpoint,
/// owner-only, failing if the name is already held.
///
/// The creation is the claim, and it is one syscall. `id_for` is the machine and
/// the port with no nonce, so two `serve` runs against one machine compute the
/// same name; asking [`exists`] and then writing lets both get past the question
/// before either answers it, and on the serve path the two are separated by the
/// entire readiness probe. The loser of that race then overwrites the holder's
/// key, fails to start a model on a port the holder still has, and clears the
/// key on its way out — leaving the holder serving a published tailnet endpoint
/// that nobody, including its owner, can call.
///
/// `create_new` is what closes it: `O_EXCL` on unix, `CREATE_NEW` on Windows,
/// and on both the existence check and the creation are one indivisible step. It
/// is the same discipline [`super::publish`] applies to a port — establish
/// ownership before writing, never after — with the difference that here the
/// establishing and the writing can be the same act.
///
/// The id is validated first: it becomes a path segment, and this is the second
/// place that matters after the record itself.
pub(crate) fn store_key(paths: &AppPaths, session_id: &str, key: &str) -> Result<()> {
    RemoteSessionRecord::validate_id(session_id)?;
    let directory = paths.remote_sessions_dir();
    fs::create_dir_all(&directory)
        .with_context(|| format!("failed to create {}", directory.display()))?;
    let path = key_path(paths, session_id);

    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    // 0600 at creation, not tightened afterwards: a credential must never exist,
    // even briefly, at whatever the umask would have given it.
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }

    let mut file = match options.open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(anyhow::Error::new(NameAlreadyHeld { path }));
        }
        Err(error) => {
            return Err(anyhow::Error::new(error))
                .with_context(|| format!("failed to write {}", path.display()));
        }
    };
    std::io::Write::write_all(&mut file, key.as_bytes())
        .with_context(|| format!("failed to write {}", path.display()))
}

/// Forget a session's credential. Idempotent, so teardown can call it without
/// checking.
pub(crate) fn clear_key(paths: &AppPaths, session_id: &str) {
    if RemoteSessionRecord::validate_id(session_id).is_ok() {
        let _ = fs::remove_file(key_path(paths, session_id));
    }
}

/// Every session recorded on this machine, ordered by id so listings are stable.
pub(crate) fn load_all(paths: &AppPaths) -> Result<Vec<RemoteSessionRecord>> {
    let directory = paths.remote_sessions_dir();
    if !directory.is_dir() {
        return Ok(Vec::new());
    }

    let mut records = Vec::new();
    for entry in fs::read_dir(&directory)
        .with_context(|| format!("failed to read {}", directory.display()))?
    {
        let path = entry?.path();
        // A staging file from an interrupted write is not a record; its
        // extension is `tmp`, so this already skips it, but say why.
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        match read_one(&path) {
            Ok(record) => records.push(record),
            // One unreadable record must not hide the rest. A half-written file
            // from an interrupted run would otherwise make every session
            // invisible, including the ones still publishing an endpoint.
            Err(error) => {
                eprintln!(
                    "warning: ignoring unreadable remote session {}: {error}",
                    path.display()
                );
            }
        }
    }
    records.sort_by(|left, right| left.session_id.cmp(&right.session_id));
    Ok(records)
}

fn read_one(path: &Path) -> Result<RemoteSessionRecord> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let record: RemoteSessionRecord = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    RemoteSessionRecord::validate_id(&record.session_id)
        .with_context(|| format!("refusing to act on {}", path.display()))?;
    Ok(record)
}

/// Find the one session a user meant.
///
/// Accepts an exact session id, or any unambiguous fragment of the target or
/// peer name. Ambiguity is refused rather than resolved by picking: `stop` on
/// the wrong session tears down someone's running model.
pub(crate) fn resolve(paths: &AppPaths, needle: &str) -> Result<RemoteSessionRecord> {
    let sessions = load_all(paths)?;
    if sessions.is_empty() {
        bail!("no remote sessions are recorded on this machine");
    }

    if let Some(found) = sessions
        .iter()
        .find(|session| session.session_id.eq_ignore_ascii_case(needle))
    {
        return Ok(found.clone());
    }

    let lowered = needle.to_ascii_lowercase();
    let matches = sessions
        .iter()
        .filter(|session| {
            session.target.to_ascii_lowercase().contains(&lowered)
                || session.peer_host.to_ascii_lowercase().contains(&lowered)
                || session.session_id.to_ascii_lowercase().contains(&lowered)
        })
        .collect::<Vec<_>>();

    match matches.as_slice() {
        [] => bail!(
            "no remote session matches `{needle}`. Recorded sessions: {}",
            sessions
                .iter()
                .map(|session| session.session_id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        [only] => Ok((*only).clone()),
        several => bail!(
            "`{needle}` matches more than one remote session: {}\n\
             Name one of them exactly.",
            several
                .iter()
                .map(|session| session.session_id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_paths(tag: &str) -> (PathBuf, AppPaths) {
        let root = std::env::temp_dir().join(format!(
            "rocm-remote-session-{tag}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        (
            root.clone(),
            AppPaths {
                config_dir: root.join("config"),
                data_dir: root.join("data"),
                cache_dir: root.join("cache"),
            },
        )
    }

    fn sample(peer_host: &str, remote_port: u16) -> RemoteSessionRecord {
        RemoteSessionRecord {
            session_id: RemoteSessionRecord::id_for(peer_host, remote_port),
            target: peer_host.split('.').next().unwrap_or(peer_host).to_owned(),
            peer_host: peer_host.to_owned(),
            ssh_port: None,
            model: "qwen".to_owned(),
            remote_service_id: "svc-1".to_owned(),
            remote_cli: "rocm".to_owned(),
            remote_port,
            tailnet_port: 8000,
            base_url: format!("http://{peer_host}:8000/v1"),
            created_at_unix_ms: 1,
        }
    }

    #[test]
    fn a_session_id_is_stable_and_safe_as_a_filename() {
        // Re-serving the same machine and port must land on the same record
        // rather than accumulating one per invocation.
        let first = RemoteSessionRecord::id_for("gpu-box-1.example-tailnet.ts.net", 11434);
        let second = RemoteSessionRecord::id_for("gpu-box-1.example-tailnet.ts.net", 11434);
        assert_eq!(first, second);
        assert_ne!(first, RemoteSessionRecord::id_for("gpu-box-1", 11435));

        // Dots and colons are not filename-safe everywhere; Windows rejects the
        // colons an IPv6 address carries outright.
        for id in [
            RemoteSessionRecord::id_for("gpu-box-1.example-tailnet.ts.net", 11434),
            RemoteSessionRecord::id_for("fd7a:115c:a1e0::3", 11434),
        ] {
            assert!(
                id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'),
                "not filename-safe: {id}"
            );
        }
    }

    #[test]
    fn a_session_round_trips_through_disk() -> Result<()> {
        let (root, paths) = temp_paths("roundtrip");
        let record = sample("gpu-box-1.example-tailnet.ts.net", 11434);
        record.write(&paths)?;

        let loaded = load_all(&paths)?;
        assert_eq!(loaded, vec![record.clone()]);

        record.remove(&paths);
        assert!(load_all(&paths)?.is_empty());
        // Removing twice must be safe: teardown calls it without checking.
        record.remove(&paths);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn a_record_is_replaced_atomically_so_a_reader_never_sees_half_of_one() {
        // A torn record is skipped by `load_all`, which hides a session whose
        // model is running and whose endpoint is published — the one thing that
        // must never happen quietly.
        let (root, paths) = temp_paths("atomic-write");
        let first = sample("gpu-box.example-tailnet.ts.net", 11434);
        first.write(&paths).unwrap();

        let mut second = first;
        second.model = "a-different-model".to_owned();
        second.write(&paths).unwrap();

        let loaded = load_all(&paths).unwrap();
        assert_eq!(loaded.len(), 1, "a rewrite must replace, not accumulate");
        assert_eq!(loaded[0].model, "a-different-model");

        // No staging file is left behind for `load_all` to trip over.
        let leftovers: Vec<_> = fs::read_dir(paths.remote_sessions_dir())
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains("tmp"))
            .collect();
        assert!(leftovers.is_empty(), "staging files left: {leftovers:?}");

        let _ = fs::remove_dir_all(root);
    }

    /// The assertions above describe the *outcome* of an atomic replace, and an
    /// in-place `fs::write` produces the same outcome — so they pass with the
    /// staging file and the rename deleted outright, certifying a guarantee
    /// nothing exercises. This observes the mechanism instead.
    ///
    /// A rename replaces the directory entry, so the destination gets a new
    /// inode. An in-place write keeps the inode and truncates, which is the
    /// window where a reader sees half a record. Comparing inodes across a
    /// rewrite therefore distinguishes the two without needing to catch the race.
    #[cfg(unix)]
    #[test]
    fn replacing_a_record_swaps_the_file_rather_than_rewriting_it_in_place() {
        use std::os::unix::fs::MetadataExt as _;

        let (root, paths) = temp_paths("atomic-swap");
        let first = sample("gpu-box.example-tailnet.ts.net", 11434);
        first.write(&paths).unwrap();
        let before = fs::metadata(first.path(&paths)).unwrap().ino();

        let mut second = first;
        second.model = "a-different-model".to_owned();
        second.write(&paths).unwrap();
        let after = fs::metadata(second.path(&paths)).unwrap().ino();

        assert_ne!(
            before, after,
            "the record was rewritten in place, so a reader can observe it truncated; \
             it must be staged and renamed over"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn two_writers_in_one_process_do_not_share_a_staging_path() {
        // The staging name carries the pid and a millisecond timestamp. Two
        // threads writing the same session id inside one millisecond would
        // otherwise pick the same staging file and rename each other's
        // half-written bytes into place — the exact race the unique name exists
        // to prevent.
        let (root, paths) = temp_paths("atomic-staging-unique");
        let record = sample("gpu-box.example-tailnet.ts.net", 11434);
        let names: std::collections::HashSet<String> = (0..64)
            .map(|_| {
                staging_path_for(&record.path(&paths))
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(
            names.len(),
            64,
            "staging names collided within a millisecond: {names:?}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn an_unreadable_record_does_not_hide_the_others() -> Result<()> {
        // A half-written file from an interrupted run must not make every
        // session invisible — including ones still publishing an endpoint that
        // the user now has no listed way to find and stop.
        let (root, paths) = temp_paths("corrupt");
        sample("gpu-box-1.example-tailnet.ts.net", 11434).write(&paths)?;
        fs::write(
            paths.remote_sessions_dir().join("truncated.json"),
            b"{\"session_id\": ",
        )?;

        let loaded = load_all(&paths)?;
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].peer_host, "gpu-box-1.example-tailnet.ts.net");

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn a_credential_is_kept_beside_its_session_and_is_recoverable() {
        // Stored owner-only, next to the record rather than in the local
        // service registry — a remote session is not a local service, and
        // sharing that folder made one registry's ids masquerade as another's.
        let (root, paths) = temp_paths("session-key");
        store_key(&paths, "remote-gpu-box-11434", "s3cret").expect("store");

        // Asserting the relationship, and saying so in the message rather than
        // dumping the path: a failure here is about *where* the credential
        // landed, which the value alone does not explain.
        let path = key_path(&paths, "remote-gpu-box-11434");
        assert!(
            path.starts_with(paths.remote_sessions_dir()),
            "a session's credential must live beside its record"
        );
        assert!(
            !path.starts_with(paths.services_dir()),
            "a remote session must not write into the local service registry"
        );
        // The user sees the key once when serving; without a readable file they
        // could never recover it.
        assert_eq!(fs::read_to_string(&path).expect("read"), "s3cret");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "a credential must be owner-only");
        }

        clear_key(&paths, "remote-gpu-box-11434");
        assert!(!path.exists());
        // Teardown calls this unconditionally, so a second clear must be safe.
        clear_key(&paths, "remote-gpu-box-11434");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_session_credential_is_claimed_rather_than_overwritten() {
        // Writing the key is what claims the session name, so it has to be the
        // thing that fails when the name is already taken. Checking first and
        // writing second cannot do that: the id is the machine and the port with
        // no nonce, so two `serve` runs compute the same name, and both can pass
        // a check before either writes.
        //
        // This observes the exclusivity itself rather than the end state. "Only
        // one key file exists afterwards" is true either way — an overwrite
        // leaves exactly one file too.
        let (root, paths) = temp_paths("claim-exclusive");
        store_key(&paths, "remote-gpu-box-11434", "the-first-sessions-key").expect("first claim");

        let second = store_key(&paths, "remote-gpu-box-11434", "a-second-runs-key");
        assert!(
            second.is_err(),
            "a second claim on a name already held must fail rather than take it over"
        );
        assert_eq!(
            fs::read_to_string(key_path(&paths, "remote-gpu-box-11434")).expect("read"),
            "the-first-sessions-key",
            "the holder's credential must survive another run's attempt to claim the name"
        );

        // Released, the name can be claimed again — teardown and a later serve
        // on the same machine and port have to keep working.
        clear_key(&paths, "remote-gpu-box-11434");
        store_key(&paths, "remote-gpu-box-11434", "a-later-sessions-key")
            .expect("a released name is claimable again");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn an_unusable_id_cannot_place_a_credential_outside_the_sessions_directory() {
        let (root, paths) = temp_paths("session-key-escape");
        assert!(store_key(&paths, "../../escaped", "s3cret").is_err());
        // And clearing one is a no-op rather than a delete somewhere else.
        clear_key(&paths, "../../escaped");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_record_with_an_unusable_id_is_refused_rather_than_followed() {
        // Ids are constructed safe, but records are also read back from disk and
        // deserializing does not re-run that construction. An id carrying a
        // traversal would make the record's path resolve outside the sessions
        // directory — and `stop` deletes that path.
        let (root, paths) = temp_paths("unsafe-id");
        let mut record = sample("gpu-box.example-tailnet.ts.net", 11434);
        record.session_id = "../../escaped".to_owned();
        fs::create_dir_all(paths.remote_sessions_dir()).unwrap();
        fs::write(
            paths.remote_sessions_dir().join("planted.json"),
            serde_json::to_vec(&record).unwrap(),
        )
        .unwrap();

        // Skipped like any other unreadable record: the rest stay visible.
        sample("other-box.example-tailnet.ts.net", 11434)
            .write(&paths)
            .unwrap();
        let loaded = load_all(&paths).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].peer_host, "other-box.example-tailnet.ts.net");

        // The rules come from the shared service-id type, so a traversal, a
        // separator or an empty id is refused with a message naming the rule.
        for bad in ["../../escaped", "a/b", "a\\b", ""] {
            assert!(
                RemoteSessionRecord::validate_id(bad).is_err(),
                "should refuse {bad:?}"
            );
        }
        assert!(RemoteSessionRecord::validate_id("remote-gpu-box-11434").is_ok());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_session_resolves_by_id_or_by_an_unambiguous_fragment() -> Result<()> {
        let (root, paths) = temp_paths("resolve");
        let first = sample("gpu-box-1.example-tailnet.ts.net", 11434);
        let second = sample("other-box.example-tailnet.ts.net", 11434);
        first.write(&paths)?;
        second.write(&paths)?;

        assert_eq!(resolve(&paths, &first.session_id)?, first);
        assert_eq!(resolve(&paths, "gpu-box-1")?, first);
        assert_eq!(resolve(&paths, "other")?, second);

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn an_ambiguous_session_is_refused_rather_than_guessed() -> Result<()> {
        // Stopping the wrong session tears down a model someone is using.
        let (root, paths) = temp_paths("ambiguous");
        sample("gpu-box-1.example-tailnet.ts.net", 11434).write(&paths)?;
        sample("gpu-box-2.example-tailnet.ts.net", 11434).write(&paths)?;

        let error = resolve(&paths, "gpu-box").unwrap_err().to_string();
        assert!(error.contains("gpu-box-1"), "{error}");
        assert!(error.contains("gpu-box-2"), "{error}");

        let unknown = resolve(&paths, "nothing-like-this")
            .unwrap_err()
            .to_string();
        assert!(
            unknown.contains("Recorded sessions"),
            "an unmatched name should list what does exist: {unknown}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
