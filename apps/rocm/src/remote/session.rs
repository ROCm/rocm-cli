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

    /// Refuse a `remote_cli` this code would never have written.
    ///
    /// Unlike every other field, `remote_cli` is interpolated **unquoted** into
    /// the remote shell string — it has to be, because the path form is
    /// `$HOME/.local/bin/rocm` and the remote shell is what expands `$HOME`.
    /// That is sound for the two values [`super::bootstrap`] produces and for
    /// nothing else: any other value is parsed by the remote shell as part of
    /// the command line the next time `rocm remote status` polls the session.
    ///
    /// Quoting it is not the fix (it would break `$HOME`), so the field is
    /// pinned to the set this code writes instead. Records are plain JSON under
    /// the user's data dir, so this is a tamper/corruption guard rather than a
    /// privilege boundary — but a session file is exactly the kind of thing a
    /// backup, a sync tool or a misbehaving script rewrites, and the blast
    /// radius is a command on another machine.
    fn validate_remote_cli(remote_cli: &str) -> Result<()> {
        if super::bootstrap::is_known_remote_cli(remote_cli) {
            return Ok(());
        }
        // Debug-formatted: the value is whatever was in the file, and echoing
        // it raw would hand the terminal any control sequences it carries.
        bail!(
            "remote session names an unexpected remote CLI {remote_cli:?}; \
             it is run unquoted on the remote, so only the forms this tool \
             installs are accepted. To use the session again, set \"remote_cli\" \
             in the file back to \"rocm\" or \"{}\", whichever the machine has. \
             Removing the file instead forgets the session but does not stop a \
             model it started on the remote machine, so check the machine first",
            super::bootstrap::REMOTE_CLI_PATH
        )
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
        ensure_sessions_dir(paths)?;
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

/// Create the session directory, private to its owner on unix.
///
/// The records in it name commands this tool runs on other machines with the
/// user's SSH identity, and the endpoint keys beside them are credentials, so
/// no other account should be able to add, replace or list a file here. `0700`
/// is set at creation rather than tightened afterwards, so the directory never
/// exists at whatever the umask would have given it.
///
/// A directory an earlier version created with the umask is tightened in place.
/// One that already exists but belongs to another account — or is a link
/// another account owns — is refused instead: its owner could list, replace or
/// remove anything written into it, and could not be locked out by a `chmod`
/// this user is not allowed to make. Refusing is safe at each call site:
/// `serve` calls this before it prepares the machine, [`store_key`] runs before
/// a model is started there, and a failed [`RemoteSessionRecord::write`] unwinds
/// the session it was recording.
pub(crate) fn ensure_sessions_dir(paths: &AppPaths) -> Result<()> {
    #[cfg(unix)]
    let owner = Some(expected_owner());
    #[cfg(not(unix))]
    let owner = None;
    ensure_sessions_dir_owned_by(paths, owner)
}

/// [`ensure_sessions_dir`], requiring an existing directory to belong to
/// `owner` when one is given. Split out for the same reason as
/// [`load_all_owned_by`]: a test cannot create a directory another user owns.
fn ensure_sessions_dir_owned_by(paths: &AppPaths, owner: Option<u32>) -> Result<()> {
    let directory = paths.remote_sessions_dir();
    if let Some(parent) = directory.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    #[cfg(unix)]
    let builder = {
        use std::os::unix::fs::DirBuilderExt as _;
        let mut builder = fs::DirBuilder::new();
        builder.mode(0o700);
        builder
    };
    #[cfg(not(unix))]
    let builder = fs::DirBuilder::new();
    match builder.create(&directory) {
        Ok(()) => Ok(()),
        // Already there — from an earlier session, or another writer that got
        // here first. Use it only if it is this user's own directory.
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists && directory.is_dir() => {
            // Ownership is a unix check only; on Windows the directory sits in
            // the per-user profile and `owner` is always `None`.
            #[cfg(unix)]
            if let Some(owner) = owner {
                claim_existing_dir(&directory, owner)
                    .with_context(|| format!("refusing to use {}", directory.display()))?;
            }
            #[cfg(not(unix))]
            let _ = owner;
            Ok(())
        }
        Err(error) => Err(anyhow::Error::new(error))
            .with_context(|| format!("failed to create {}", directory.display())),
    }
}

/// Accept an existing sessions directory only if `owner` owns it — and, when
/// it is a link, the link too — then drop group and other access from it.
#[cfg(unix)]
fn claim_existing_dir(directory: &Path, owner: u32) -> Result<()> {
    use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
    let link = fs::symlink_metadata(directory)
        .with_context(|| format!("failed to inspect {}", directory.display()))?;
    let target = fs::metadata(directory)
        .with_context(|| format!("failed to inspect {}", directory.display()))?;
    if let Some(found) = first_foreign_uid(&[link.uid(), target.uid()], owner) {
        bail!(
            "this directory is owned by another user (uid {found}), who could list, \
             replace or remove the session records and credentials kept in it. {}",
            // Emptying a directory needs write access to it, which this user
            // lacks; its parent is this user's, so moving it aside works.
            foreign_owner_remedy(
                found,
                owner,
                "move it aside or have its owner remove it",
                "Moving it aside forgets every session recorded in it, but does not stop \
                 the models those sessions started on remote machines."
            )
        );
    }
    if target.mode() & 0o077 != 0 {
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("failed to restrict {} to its owner", directory.display()))?;
    }
    Ok(())
}

/// The first of `uids` that is not `owner`.
///
/// A path is checked twice — the link itself and what it resolves to — and
/// either belonging to someone else is enough to refuse it, so the order the
/// two are passed in must not matter.
#[cfg(unix)]
fn first_foreign_uid(uids: &[u32], owner: u32) -> Option<u32> {
    uids.iter().copied().find(|uid| *uid != owner)
}

/// What to tell a user whose record or sessions directory belongs to `found`
/// while this process runs as `owner`.
///
/// Derived from the same uid the refusal compared against, so the advice cannot
/// disagree with the decision. Running as root is the case that needs its own
/// words: `sudo -E`, or a `sudo` that keeps `HOME`, reads the invoking user's
/// own files as somebody else's, so that branch says only to run as their owner
/// and never to get rid of them — `otherwise` and its `consequence` are advice
/// for a user who really is looking at somebody else's file.
#[cfg(unix)]
fn foreign_owner_remedy(found: u32, owner: u32, otherwise: &str, consequence: &str) -> String {
    if owner == 0 {
        format!(
            "This command is running as root: if it is yours, run `rocm remote` as the \
             user who owns it (uid {found}), not under sudo."
        )
    } else {
        format!(
            "If it is yours, restore its ownership; otherwise {otherwise}, then retry. \
             {consequence}"
        )
    }
}

/// The effective user id of this process.
#[cfg(unix)]
fn current_uid() -> u32 {
    // SAFETY: `geteuid` takes no arguments, cannot fail, and touches no memory.
    #[allow(unsafe_code)] // libc FFI
    unsafe {
        libc::geteuid()
    }
}

/// The uid records and the sessions directory must belong to: this process's
/// own, except where a test has stood in another, which is how a test reaches
/// the owner checks through [`load_all`] and [`ensure_sessions_dir`] themselves
/// rather than only through their `_owned_by` halves.
#[cfg(unix)]
fn expected_owner() -> u32 {
    #[cfg(test)]
    if let Some(owner) = OWNER_OVERRIDE.with(std::cell::Cell::get) {
        return owner;
    }
    current_uid()
}

#[cfg(all(unix, test))]
thread_local! {
    static OWNER_OVERRIDE: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// Run `body` as if this process were `owner`, for the owner checks only.
///
/// An unprivileged test cannot create a file another user owns, but it can
/// expect a uid other than its own, which takes the same comparison. Scoped to
/// the calling thread and reset even if `body` panics.
#[cfg(all(unix, test))]
pub(crate) fn with_expected_owner<T>(owner: u32, body: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            OWNER_OVERRIDE.with(|cell| cell.set(None));
        }
    }
    OWNER_OVERRIDE.with(|cell| cell.set(Some(owner)));
    let _reset = Reset;
    body()
}

/// This process's own uid, for tests that need a uid that is not it.
#[cfg(all(unix, test))]
pub(crate) fn current_uid_for_tests() -> u32 {
    current_uid()
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
    ensure_sessions_dir(paths)?;
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

/// What [`load_all`] found: the records it will act on, and the paths of the
/// `.json` files it would not act on.
///
/// The refused ones are kept rather than dropped because the session each
/// describes may still be running: a listing that silently omits it tells the
/// user there is nothing on the machine, and `stop` cannot reach it either.
#[derive(Debug, Default)]
pub(crate) struct LoadedSessions {
    pub(crate) records: Vec<RemoteSessionRecord>,
    pub(crate) refused: Vec<PathBuf>,
}

impl LoadedSessions {
    /// One sentence for `status` and `resolve`, which would otherwise report
    /// "no sessions" or "no such session", or `None` when nothing was refused.
    ///
    /// It gives no remedy of its own: each refused record's warning, printed as
    /// it was loaded, carries the remedy for its own reason, and a second one
    /// here could contradict it — under root, for one. Nor does it say to run
    /// `rocm remote stop`, which cannot reach a refused record.
    pub(crate) fn refused_summary(&self) -> Option<String> {
        if self.refused.is_empty() {
            return None;
        }
        Some(format!(
            "{} remote session record(s) were refused and are not listed (see the warnings \
             above): {}. The machine(s) they name may still be serving a model.",
            self.refused.len(),
            self.refused
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

/// Every session recorded on this machine that is safe to act on, ordered by
/// id so listings are stable, together with every record that was refused.
///
/// A record is refused — skipped with a warning on stderr and returned in
/// [`LoadedSessions::refused`] — when it cannot be read or parsed, when it fails
/// the checks in [`read_one`], or, on unix, when another user owns it.
pub(crate) fn load_all(paths: &AppPaths) -> Result<LoadedSessions> {
    #[cfg(unix)]
    let owner = Some(expected_owner());
    #[cfg(not(unix))]
    let owner = None;
    load_all_owned_by(paths, owner)
}

/// [`load_all`], accepting only records owned by `owner` when one is given.
///
/// Split out so a test can stand in for "written by another account": an
/// unprivileged test cannot create a file someone else owns, but it can ask for
/// records owned by a uid other than its own, which takes the same branch.
fn load_all_owned_by(paths: &AppPaths, owner: Option<u32>) -> Result<LoadedSessions> {
    let directory = paths.remote_sessions_dir();
    let mut loaded = LoadedSessions::default();
    if !directory.is_dir() {
        return Ok(loaded);
    }

    for entry in fs::read_dir(&directory)
        .with_context(|| format!("failed to read {}", directory.display()))?
    {
        let path = entry?.path();
        // A staging file from an interrupted write is not a record; its
        // extension is `tmp`, so this already skips it, but say why.
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        match read_one(&path, owner) {
            Ok(record) => loaded.records.push(record),
            // One unreadable record must not hide the rest. A half-written file
            // from an interrupted run would otherwise make every session
            // invisible, including the ones still publishing an endpoint.
            //
            // The whole chain, not just the outermost context: that context is
            // only "refusing to act on <path>", and a refusal that does not say
            // why leaves the user nothing to act on. Every context `read_one`
            // adds already names the file.
            Err(error) => {
                eprintln!("warning: ignoring a remote session record: {error:#}");
                loaded.refused.push(path);
            }
        }
    }
    loaded
        .records
        .sort_by(|left, right| left.session_id.cmp(&right.session_id));
    loaded.refused.sort();
    Ok(loaded)
}

/// Read the record filed under `session_id`, with every check [`load_all`]
/// applies. For a caller that has to tell a usable record from a refused one.
pub(crate) fn read_recorded(paths: &AppPaths, session_id: &str) -> Result<RemoteSessionRecord> {
    #[cfg(unix)]
    let owner = Some(expected_owner());
    #[cfg(not(unix))]
    let owner = None;
    read_one(&RemoteSessionRecord::path_in(paths, session_id), owner)
}

/// Read and vet one record.
///
/// When `owner` is given, the record must belong to that user. A record is a
/// list of commands this tool runs on another machine with the user's SSH
/// identity, so one another account wrote — through a group-writable install
/// prefix, say — is refused rather than trusted. The owner is taken from the
/// opened file, so the file checked is the file read; a symlink's own owner is
/// checked too, so a link another account planted cannot borrow this user's
/// ownership of whatever it points at.
///
/// The record must also be filed under its own id. `stop` and `serve` find a
/// record by building its path from the id, so one filed under another name is
/// a session those commands would act on under the wrong name — and `stop`
/// would delete a file other than the one it read.
fn read_one(path: &Path, owner: Option<u32>) -> Result<RemoteSessionRecord> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to read {}", path.display()))?;
    // Ownership is a unix check only; on Windows the records sit in the
    // per-user profile and `owner` is always `None`.
    #[cfg(unix)]
    if let Some(owner) = owner {
        check_owner(path, &file, owner)
            .with_context(|| format!("refusing to act on {}", path.display()))?;
    }
    #[cfg(not(unix))]
    let _ = owner;
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut bytes)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let record: RemoteSessionRecord = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    RemoteSessionRecord::validate_id(&record.session_id)
        .with_context(|| format!("refusing to act on {}", path.display()))?;
    let filed_as = path.file_stem().and_then(|stem| stem.to_str());
    if filed_as != Some(record.session_id.as_str()) {
        bail!(
            "refusing to act on {}: it records session {:?} but is not named after it; \
             rename it to {:?} or remove it",
            path.display(),
            record.session_id,
            format!("{}.json", record.session_id)
        );
    }
    RemoteSessionRecord::validate_remote_cli(&record.remote_cli)
        .with_context(|| format!("refusing to act on {}", path.display()))?;
    Ok(record)
}

/// Refuse a record, or a link to one, that `owner` does not own.
#[cfg(unix)]
fn check_owner(path: &Path, file: &fs::File, owner: u32) -> Result<()> {
    use std::os::unix::fs::MetadataExt as _;
    let link = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect {}", path.display()))?
        .uid();
    let opened = file
        .metadata()
        .with_context(|| format!("failed to inspect {}", path.display()))?
        .uid();
    if let Some(found) = first_foreign_uid(&[link, opened], owner) {
        bail!(
            "this session record is owned by another user (uid {found}), and its commands \
             would run on a remote machine as you. {}",
            foreign_owner_remedy(
                found,
                owner,
                "delete it",
                "Deleting it forgets the session but does not stop a model it started on \
                 the remote machine."
            )
        );
    }
    Ok(())
}

/// Find the one session a user meant.
///
/// Accepts an exact session id, or any unambiguous fragment of the target or
/// peer name. Ambiguity is refused rather than resolved by picking: `stop` on
/// the wrong session tears down someone's running model.
///
/// A refused record is never a match — it is not safe to act on. When nothing
/// is recorded or nothing matches, the refused records are named, since one of
/// them may be the session the user meant.
pub(crate) fn resolve(paths: &AppPaths, needle: &str) -> Result<RemoteSessionRecord> {
    let loaded = load_all(paths)?;
    let refused = loaded
        .refused_summary()
        .map(|summary| format!("\n{summary}"))
        .unwrap_or_default();
    let sessions = &loaded.records;
    if sessions.is_empty() {
        if refused.is_empty() {
            bail!("no remote sessions are recorded on this machine");
        }
        bail!("no usable remote session is recorded on this machine.{refused}");
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
            "no remote session matches `{needle}`. Recorded sessions: {}{refused}",
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
            "rocm-remote-session-{tag}-{}",
            crate::test_support::unique_suffix()
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

    #[test]
    fn temp_paths_gives_each_call_its_own_root_even_for_one_label() {
        crate::test_support::assert_each_call_gets_its_own_root(|| temp_paths("same-label").0);
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

        let loaded = load_all(&paths)?.records;
        assert_eq!(loaded, vec![record.clone()]);

        record.remove(&paths);
        assert!(load_all(&paths)?.records.is_empty());
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

        let loaded = load_all(&paths).unwrap().records;
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

        let loaded = load_all(&paths)?.records;
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
        let loaded = load_all(&paths).unwrap().records;
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

    /// The refusal names the file and the reason, and shows the offending value
    /// escaped: it came from the file, so printing it raw would pass any
    /// terminal control sequences it carries straight to the user's terminal.
    /// `load_all` prints this chain as its warning (`{error:#}`); the
    /// `remote-status-refuses-altered-record` scenario checks it reaches the
    /// user and that nothing ran on the machine.
    #[test]
    fn a_refused_record_says_which_file_and_why() {
        let (root, paths) = temp_paths("refusal-reason");
        let mut record = sample("gpu-box.example-tailnet.ts.net", 11434);
        record.remote_cli = "rocm\u{1b}[2J; touch /tmp/unexpected #".to_owned();
        // `write` does not vet the record; only reading does.
        record.write(&paths).unwrap();
        let path = record.path(&paths);

        let refusal = format!("{:#}", read_one(&path, None).unwrap_err());
        assert!(
            refusal.contains(&path.display().to_string()),
            "names the file: {refusal}"
        );
        assert!(
            refusal.contains("unexpected remote CLI"),
            "says why: {refusal}"
        );
        assert!(
            !refusal.contains('\u{1b}'),
            "a control character from the file reached the message: {refusal:?}"
        );
        assert!(load_all(&paths).unwrap().records.is_empty());

        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    fn mode_of(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt as _;
        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    /// Both ways into the directory — a record and a credential — create it
    /// owner-only, because either can be the first file a session writes.
    #[cfg(unix)]
    #[test]
    fn the_sessions_directory_is_created_private_to_its_owner() {
        let (root, paths) = temp_paths("private-dir-record");
        sample("gpu-box.example-tailnet.ts.net", 11434)
            .write(&paths)
            .unwrap();
        assert_eq!(mode_of(&paths.remote_sessions_dir()), 0o700);
        let _ = fs::remove_dir_all(root);

        let (root, paths) = temp_paths("private-dir-key");
        store_key(&paths, "remote-gpu-box-11434", "secret").unwrap();
        assert_eq!(mode_of(&paths.remote_sessions_dir()), 0o700);
        let _ = fs::remove_dir_all(root);
    }

    /// A sessions directory that already exists but belongs to someone else is
    /// not written into: its owner could list, replace or remove what lands
    /// there. Asserted together with the message, and with the remedy the
    /// message names — removing the directory — actually clearing it.
    #[cfg(unix)]
    #[test]
    fn a_sessions_directory_owned_by_another_user_is_not_used() {
        use std::os::unix::fs::PermissionsExt as _;

        let (root, paths) = temp_paths("foreign-dir");
        let directory = paths.remote_sessions_dir();
        fs::create_dir_all(&directory).unwrap();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o775)).unwrap();
        let someone_else = current_uid().wrapping_add(1);

        let refusal = format!(
            "{:#}",
            ensure_sessions_dir_owned_by(&paths, Some(someone_else)).unwrap_err()
        );
        assert!(
            refusal.contains(&directory.display().to_string())
                && refusal.contains("owned by another user"),
            "the user must be told which directory and why: {refusal}"
        );
        assert_eq!(
            mode_of(&directory),
            0o775,
            "a directory that is refused is left as it was"
        );

        // The remedy the message gives a non-root user: move it aside — its
        // parent is this user's — and the next write recreates it private.
        assert!(refusal.contains("move it aside"), "{refusal}");
        fs::rename(&directory, directory.with_extension("aside")).unwrap();
        store_key(&paths, "remote-gpu-box-11434", "secret").unwrap();
        assert_eq!(mode_of(&directory), 0o700);

        let _ = fs::remove_dir_all(root);
    }

    /// Earlier versions created the directory with the umask, typically `0755`.
    /// The next write brings such a directory in line rather than leaving every
    /// existing install at the old permissions.
    #[cfg(unix)]
    #[test]
    fn a_sessions_directory_left_open_by_an_earlier_version_is_tightened() {
        use std::os::unix::fs::PermissionsExt as _;

        let (root, paths) = temp_paths("tighten-dir");
        let directory = paths.remote_sessions_dir();
        fs::create_dir_all(&directory).unwrap();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o775)).unwrap();

        sample("gpu-box.example-tailnet.ts.net", 11434)
            .write(&paths)
            .unwrap();
        assert_eq!(mode_of(&directory), 0o700);

        let _ = fs::remove_dir_all(root);
    }

    /// A record somebody else owns is refused, and the refusal is what the user
    /// is shown — asserted together, so the message cannot claim a refusal
    /// that did not happen. An unprivileged test cannot create a file another
    /// user owns, so it asks for a different owner instead, which takes the
    /// same comparison.
    #[cfg(unix)]
    #[test]
    fn a_record_owned_by_another_user_is_refused() {
        let (root, paths) = temp_paths("foreign-owner");
        let record = sample("gpu-box.example-tailnet.ts.net", 11434);
        record.write(&paths).unwrap();
        let someone_else = current_uid().wrapping_add(1);

        assert!(
            load_all_owned_by(&paths, Some(someone_else))
                .unwrap()
                .records
                .is_empty(),
            "a record owned by another account must not be acted on"
        );
        let refusal = format!(
            "{:#}",
            read_one(&record.path(&paths), Some(someone_else)).unwrap_err()
        );
        assert!(
            refusal.contains("owned by another user"),
            "the user must be told why the record was skipped: {refusal}"
        );
        // Expected owner is not root: the ordinary remedy, and what deleting does.
        assert!(
            refusal.contains("restore its ownership")
                && refusal.contains("delete it")
                && refusal.contains("does not stop"),
            "{refusal}"
        );
        assert!(!refusal.contains("not under sudo"), "{refusal}");

        // The same file, checked against its real owner, is a session like any
        // other — which is what `load_all` itself checks against.
        assert_eq!(load_all(&paths).unwrap().records, vec![record]);

        let _ = fs::remove_dir_all(root);
    }

    /// Rewrite one field of the record at `path`, as an editor or a script would.
    fn rewrite_field(path: &Path, field: &str, value: &str) {
        let mut json: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        json[field] = serde_json::Value::String(value.to_owned());
        fs::write(path, serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    }

    /// What `load_all` warns with for the record at `path`.
    fn refusal_of(path: &Path, owner: Option<u32>) -> String {
        format!("{:#}", read_one(path, owner).unwrap_err())
    }

    /// A uid other than root's that the given paths belong to: this process's
    /// own, or — run as root — an unprivileged one they are handed to, so the
    /// owner tests run, rather than skip, in a root CI container.
    #[cfg(unix)]
    fn unprivileged_owner_of(paths: &[&Path]) -> u32 {
        const UNPRIVILEGED: u32 = 65534;
        let me = current_uid();
        if me != 0 {
            return me;
        }
        for path in paths {
            std::os::unix::fs::lchown(path, Some(UNPRIVILEGED), None).unwrap();
        }
        UNPRIVILEGED
    }

    /// A refused record is reported back, not only warned about, and the
    /// remedy its message gives — restoring `remote_cli` — makes it a session
    /// again. Message and outcome together, so the advice cannot go stale.
    #[test]
    fn a_refused_remote_cli_is_returned_with_a_remedy_that_works() {
        let (root, paths) = temp_paths("remote-cli-remedy");
        let record = sample("gpu-box.example-tailnet.ts.net", 11434);
        record.write(&paths).unwrap();
        let path = record.path(&paths);
        rewrite_field(&path, "remote_cli", "rocm; true #");

        let loaded = load_all(&paths).unwrap();
        assert!(loaded.records.is_empty(), "{loaded:?}");
        assert_eq!(loaded.refused, vec![path.clone()]);
        let refusal = refusal_of(&path, None);
        assert!(
            refusal.contains("set \"remote_cli\" in the file back to \"rocm\"")
                && refusal.contains("does not stop"),
            "the refusal must say how to recover and what removing does: {refusal}"
        );

        rewrite_field(&path, "remote_cli", "rocm");
        let loaded = load_all(&paths).unwrap();
        assert_eq!(loaded.records, vec![record]);
        assert!(loaded.refused.is_empty());

        let _ = fs::remove_dir_all(root);
    }

    /// `stop` builds a record's path from its id, so a record filed under some
    /// other name would be acted on under a name that is not its file. It is
    /// refused, and renaming it as the message says makes it usable.
    #[test]
    fn a_record_filed_under_another_name_is_refused_until_renamed() {
        let (root, paths) = temp_paths("misfiled");
        let record = sample("gpu-box.example-tailnet.ts.net", 11434);
        record.write(&paths).unwrap();
        let misfiled = paths.remote_sessions_dir().join("remote-other-box-1.json");
        fs::rename(record.path(&paths), &misfiled).unwrap();

        let loaded = load_all(&paths).unwrap();
        assert!(loaded.records.is_empty(), "{loaded:?}");
        assert_eq!(loaded.refused, vec![misfiled.clone()]);
        let refusal = refusal_of(&misfiled, None);
        assert!(
            refusal.contains("is not named after it")
                && refusal.contains(&format!("{}.json", record.session_id)),
            "{refusal}"
        );

        fs::rename(&misfiled, record.path(&paths)).unwrap();
        assert_eq!(load_all(&paths).unwrap().records, vec![record]);

        let _ = fs::remove_dir_all(root);
    }

    /// Either uid being foreign is enough, whichever position it is in.
    #[cfg(unix)]
    #[test]
    fn first_foreign_uid_finds_a_stranger_in_either_position() {
        assert_eq!(first_foreign_uid(&[1000, 1001], 1000), Some(1001));
        assert_eq!(first_foreign_uid(&[1001, 1000], 1000), Some(1001));
        assert_eq!(first_foreign_uid(&[1000, 1000], 1000), None);
    }

    /// The owner check runs inside `load_all` itself, not only in the
    /// `_owned_by` halves the other tests call: a process expecting another uid
    /// refuses the user's records, and reports them as refused.
    #[cfg(unix)]
    #[test]
    fn load_all_and_the_writers_check_ownership_themselves() {
        let (root, paths) = temp_paths("owner-wiring");
        let record = sample("gpu-box.example-tailnet.ts.net", 11434);
        record.write(&paths).unwrap();
        let someone_else = current_uid().wrapping_add(1);

        let loaded = with_expected_owner(someone_else, || load_all(&paths).unwrap());
        assert!(loaded.records.is_empty(), "{loaded:?}");
        assert_eq!(loaded.refused, vec![record.path(&paths)]);
        assert!(
            refusal_of(&record.path(&paths), Some(someone_else)).contains("owned by another user")
        );

        let refusal = with_expected_owner(someone_else, || {
            store_key(&paths, &record.session_id, "secret").unwrap_err()
        });
        assert!(
            format!("{refusal:#}").contains("owned by another user"),
            "{refusal:#}"
        );
        assert!(!key_path(&paths, &record.session_id).exists());

        let _ = fs::remove_dir_all(root);
    }

    /// Under `sudo -E`, or a `sudo` that keeps `HOME`, root reads the user's
    /// own records. The refusal tells root to run as that user and says nothing
    /// about getting rid of them — and running as that user does work.
    #[cfg(unix)]
    #[test]
    fn under_root_a_refusal_says_to_run_as_the_owner_rather_than_delete() {
        let (root, paths) = temp_paths("under-sudo");
        let record = sample("gpu-box.example-tailnet.ts.net", 11434);
        record.write(&paths).unwrap();
        let directory = paths.remote_sessions_dir();
        let user = unprivileged_owner_of(&[&directory, &record.path(&paths)]);

        let loaded = load_all_owned_by(&paths, Some(0)).unwrap();
        assert!(loaded.records.is_empty(), "root must not act on it");
        let refusal = refusal_of(&record.path(&paths), Some(0));
        assert!(
            refusal.contains("running as root")
                && refusal.contains(&format!("(uid {user}), not under sudo")),
            "{refusal}"
        );
        for advice in ["delete it", "restore its ownership", "Deleting it"] {
            assert!(
                !refusal.contains(advice),
                "{advice:?} under root: {refusal}"
            );
        }
        assert_eq!(
            load_all_owned_by(&paths, Some(user)).unwrap().records,
            vec![record],
            "run as its owner, the record is a session again"
        );

        let before = fs::metadata(&directory).unwrap().permissions();
        let refusal = format!(
            "{:#}",
            ensure_sessions_dir_owned_by(&paths, Some(0)).unwrap_err()
        );
        assert!(
            refusal.contains("running as root")
                && refusal.contains(&format!("(uid {user}), not under sudo")),
            "{refusal}"
        );
        for advice in ["move it aside", "restore its ownership", "Moving it aside"] {
            assert!(
                !refusal.contains(advice),
                "{advice:?} under root: {refusal}"
            );
        }
        assert_eq!(fs::metadata(&directory).unwrap().permissions(), before);
        ensure_sessions_dir_owned_by(&paths, Some(user)).unwrap();

        let _ = fs::remove_dir_all(root);
    }

    /// A root-owned path for the linked tests: as root, one this test creates
    /// already owner-only, so even a broken check has nothing to tighten; run
    /// unprivileged, a system path this process cannot change.
    #[cfg(unix)]
    fn root_owned(root: &Path, system_path: &str, directory: bool) -> PathBuf {
        use std::os::unix::fs::{MetadataExt as _, PermissionsExt as _};
        let path = if current_uid() == 0 {
            let path = root.join("root-owned");
            if directory {
                fs::create_dir(&path).unwrap();
            } else {
                fs::write(&path, b"{}").unwrap();
            }
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
            path
        } else {
            PathBuf::from(system_path)
        };
        assert_eq!(
            fs::metadata(&path).unwrap().uid(),
            0,
            "{} must be root-owned for this test to mean anything",
            path.display()
        );
        path
    }

    /// A record reached through a link is refused when either end belongs to
    /// someone else. The link is the user's and its target is root's, so each
    /// owner passed sees exactly one foreign end.
    #[cfg(unix)]
    #[test]
    fn a_linked_record_is_refused_if_either_end_is_foreign() {
        let (root, paths) = temp_paths("linked-record");
        let target = root_owned(&root, "/etc/passwd", false);
        fs::create_dir_all(paths.remote_sessions_dir()).unwrap();
        let link = paths
            .remote_sessions_dir()
            .join("remote-gpu-box-11434.json");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let user = unprivileged_owner_of(&[&link]);

        // The opened file is root's: refused as uid 0, though the link is the
        // user's.
        let refusal = refusal_of(&link, Some(user));
        assert!(
            refusal.contains("owned by another user (uid 0)"),
            "{refusal}"
        );
        // The link is the user's, not root's: refused as theirs, though root
        // owns what it points at.
        let refusal = refusal_of(&link, Some(0));
        assert!(
            refusal.contains(&format!("owned by another user (uid {user})")),
            "{refusal}"
        );

        let _ = fs::remove_dir_all(root);
    }

    /// The same for the sessions directory.
    #[cfg(unix)]
    #[test]
    fn a_linked_sessions_directory_is_refused_if_either_end_is_foreign() {
        let (root, paths) = temp_paths("linked-dir");
        let target = root_owned(&root, "/", true);
        fs::create_dir_all(&paths.data_dir).unwrap();
        std::os::unix::fs::symlink(&target, paths.remote_sessions_dir()).unwrap();
        let user = unprivileged_owner_of(&[&paths.remote_sessions_dir()]);

        let refusal = format!(
            "{:#}",
            ensure_sessions_dir_owned_by(&paths, Some(user)).unwrap_err()
        );
        assert!(
            refusal.contains("owned by another user (uid 0)"),
            "{refusal}"
        );
        let refusal = format!(
            "{:#}",
            ensure_sessions_dir_owned_by(&paths, Some(0)).unwrap_err()
        );
        assert!(
            refusal.contains(&format!("owned by another user (uid {user})")),
            "{refusal}"
        );

        let _ = fs::remove_dir_all(root);
    }

    /// Something other than a directory where the sessions directory belongs is
    /// not claimed — not tightened, not written into — but reported.
    #[cfg(unix)]
    #[test]
    fn a_file_where_the_sessions_directory_belongs_is_not_claimed() {
        use std::os::unix::fs::PermissionsExt as _;
        let (root, paths) = temp_paths("file-not-dir");
        fs::create_dir_all(&paths.data_dir).unwrap();
        let directory = paths.remote_sessions_dir();
        fs::write(&directory, b"not a directory").unwrap();
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o644)).unwrap();

        let error = format!(
            "{:#}",
            ensure_sessions_dir_owned_by(&paths, Some(current_uid())).unwrap_err()
        );
        assert!(error.contains("failed to create"), "{error}");
        assert_eq!(mode_of(&directory), 0o644, "the file is left as it was");

        let _ = fs::remove_dir_all(root);
    }
}
