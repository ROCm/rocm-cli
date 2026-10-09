// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Property-test support for the archive extractors (test-only).
//!
//! Every place `rocm` unpacks a downloaded archive (the Lemonade embeddable,
//! the TheRock SDK tarball, the `uv` bootstrap, the ComfyUI source) shares one
//! safety contract: whatever the entry names and kinds, nothing is written
//! outside the extraction root, and the tree left behind carries no link that
//! resolves out of it and no setuid/setgid or world-writable entry.
//!
//! This module is shared by `#[path]` include from each crate that owns an
//! extractor, so the generator and the oracle are one definition:
//!
//! - the **generator** builds archives byte-by-byte from a small alphabet that
//!   is mostly `..`, `.`, empty components, mixed separators and links, so the
//!   dangerous shapes actually occur instead of being drowned in random text;
//! - the **oracle** never re-derives the extractor's path logic: it snapshots
//!   the whole per-case temporary directory before and after, and
//!   walks/canonicalizes what is actually on disk.
//!
//! Every case runs in a fresh temporary directory laid out so that nothing
//! the generator can name reaches past it (see [`Layout`]): absolute names and
//! link targets are re-rooted inside it, and the extraction root sits deeper
//! than any `..` chain the generator can build. An extractor that escapes its
//! root therefore writes somewhere the oracle watches, never onto the host.
//! The one escape that leaves the temporary directory regardless is an
//! extractor ignoring its destination altogether (`tar` without `-C` writes
//! into the working directory); [`check_round_trip`] over [`benign_entries`]
//! is what catches that.
//!
//! Unix-only: the oracle reads mode bits, inode numbers and symlinks. Mode
//! checks read the process umask and effective uid from `/proc/self/status`,
//! because what an extractor may leave behind depends on both (see [`Modes`]).

#![cfg(all(test, unix))]
#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "test-only support shared by #[path] include; not every includer uses every helper"
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use proptest::prelude::*;

/// Placeholder for "an absolute path into the sibling `outside/` directory";
/// substituted per case once the temporary layout exists.
pub const OUTSIDE_TOKEN: &str = "@OUT";

/// Most members in one generated archive.
const MAX_ENTRIES: usize = 6;

/// Most components in one generated name.
const MAX_COMPONENTS: usize = 4;

/// Most `..` steps one generated name or link target can take: one per
/// component, even for an extractor that reads `\` as a separator (no
/// component holds more than one `..`, which is what keeps [`NESTING`], and
/// so the cost of every case, small).
const MAX_UP_PER_STRING: usize = MAX_COMPONENTS;

/// How many directories a [`Fence`] nests each case's fake root under the
/// temporary directory. A path the generator builds starts at the extraction root, at
/// the fake root (absolute names and targets are re-rooted there), or where an
/// earlier member's link lands, and its own name or target then climbs at most
/// [`MAX_UP_PER_STRING`]. Each member contributes a name and at most one
/// target, so no chain over one archive climbs more than
/// `MAX_ENTRIES * 2 * MAX_UP_PER_STRING` above the fake root; one more level
/// keeps even that inside the temporary directory.
const NESTING: usize = MAX_ENTRIES * 2 * MAX_UP_PER_STRING + 1;

/// What one archive member is.
#[derive(Clone, Debug)]
pub enum Kind {
    File(Vec<u8>),
    Dir,
    Symlink(Vec<u8>),
    Hardlink(Vec<u8>),
}

/// One archive member: raw name bytes (not validated, not normalised), kind,
/// and mode.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: Vec<u8>,
    pub kind: Kind,
    pub mode: u32,
}

/// Component alphabet. Weighted toward traversal and separator shapes; `l` is
/// the name links are usually given, so "link then traverse the link" occurs.
fn component() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        6 => Just(b"..".to_vec()),
        2 => Just(b".".to_vec()),
        2 => Just(Vec::new()),
        4 => Just(b"l".to_vec()),
        3 => Just(b"a".to_vec()),
        1 => Just(b"b".to_vec()),
        1 => Just(b"..\\x".to_vec()),
        1 => Just(b"a\\b".to_vec()),
        1 => Just(b"C:".to_vec()),
        1 => Just(b"C:\\x".to_vec()),
        1 => Just(b"\\\\srv\\share".to_vec()),
        1 => Just(vec![0xff, 0xfe]),
        1 => Just(b"x\0y".to_vec()),
        1 => Just(vec![b'n'; 120]),
        1 => Just(b"victim".to_vec()),
    ]
}

fn separator() -> impl Strategy<Value = &'static [u8]> {
    prop_oneof![
        6 => Just(&b"/"[..]),
        2 => Just(&b"\\"[..]),
        1 => Just(&b"//"[..]),
    ]
}

fn prefix() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        6 => Just(Vec::new()),
        2 => Just(b"./".to_vec()),
        2 => Just(b"/".to_vec()),
        2 => Just(format!("{OUTSIDE_TOKEN}/").into_bytes()),
        1 => Just(b"l/".to_vec()),
    ]
}

/// A raw entry name: optional prefix, then 1-4 components joined by mixed
/// separators.
pub fn name() -> impl Strategy<Value = Vec<u8>> {
    (
        prefix(),
        prop::collection::vec((component(), separator()), 1..=MAX_COMPONENTS),
    )
        .prop_map(|(prefix, parts)| {
            let mut out = prefix;
            for (index, (part, sep)) in parts.iter().enumerate() {
                if index > 0 {
                    out.extend_from_slice(sep);
                }
                out.extend_from_slice(part);
            }
            out
        })
}

fn link_target() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        3 => Just(OUTSIDE_TOKEN.as_bytes().to_vec()),
        2 => Just(format!("{OUTSIDE_TOKEN}/sentinel").into_bytes()),
        2 => Just(b"../outside".to_vec()),
        2 => Just(b"../../outside".to_vec()),
        1 => Just(b"../victim".to_vec()),
        1 => Just(b"/".to_vec()),
        1 => Just(b".".to_vec()),
        3 => name(),
    ]
}

fn mode() -> impl Strategy<Value = u32> {
    prop_oneof![
        4 => Just(0o644),
        2 => Just(0o755),
        1 => Just(0o4755),
        1 => Just(0o2755),
        1 => Just(0o777),
        1 => Just(0o1777),
        1 => Just(0o666),
    ]
}

fn kind() -> impl Strategy<Value = Kind> {
    prop_oneof![
        4 => prop::collection::vec(any::<u8>(), 0..8).prop_map(Kind::File),
        2 => Just(Kind::Dir),
        3 => link_target().prop_map(Kind::Symlink),
        1 => link_target().prop_map(Kind::Hardlink),
    ]
}

fn entry() -> impl Strategy<Value = Entry> {
    // A link is usually named `l` (or `a/l`) so a later entry under `l/` walks
    // through it.
    let link_name = prop_oneof![
        3 => Just(b"l".to_vec()),
        1 => Just(b"a/l".to_vec()),
        1 => name(),
    ];
    prop_oneof![
        3 => (name(), kind(), mode()).prop_map(|(name, kind, mode)| Entry { name, kind, mode }),
        2 => (link_name, link_target()).prop_map(|(name, target)| Entry {
            name,
            kind: Kind::Symlink(target),
            mode: 0o777,
        }),
        2 => (prop_oneof![Just(b"l/".to_vec()), Just(b"a/l/".to_vec())], name(), mode())
            .prop_map(|(mut prefix, rest, mode)| {
                prefix.extend_from_slice(&rest);
                Entry { name: prefix, kind: Kind::File(b"pwned".to_vec()), mode }
            }),
    ]
}

/// A whole archive: 1-6 members.
pub fn entries() -> impl Strategy<Value = Vec<Entry>> {
    prop::collection::vec(entry(), 1..=MAX_ENTRIES)
}

/// Archives that every extractor must accept: plain relative names built only
/// from safe components, regular files and directories, ordinary modes. Each
/// extractor's tests run these through [`check_round_trip`], the "does not
/// over-reject / extracts what it was given, where it was told to" direction.
pub fn benign_entries() -> impl Strategy<Value = Vec<Entry>> {
    let part = prop_oneof![Just("a"), Just("b"), Just("c"), Just("d.txt")];
    prop::collection::vec(
        (
            prop::collection::vec(part, 1..=3),
            prop::collection::vec(any::<u8>(), 0..16),
        ),
        1..=6,
    )
    .prop_map(|items| {
        // `top/` keeps every benign archive single-rooted, like the real
        // artifacts, and file names get a unique suffix so no file collides
        // with a directory of the same name.
        let mut out = vec![Entry {
            name: b"top/".to_vec(),
            kind: Kind::Dir,
            mode: 0o755,
        }];
        for (index, (parts, data)) in items.into_iter().enumerate() {
            let name = format!("top/{}.f{index}", parts.join("/"));
            out.push(Entry {
                name: name.into_bytes(),
                kind: Kind::File(data),
                mode: 0o644,
            });
        }
        out
    })
}

/// The benign direction, for an archive from [`benign_entries`]: `top` is
/// where the extractor left the archive's single `top/` directory (the
/// extraction root's `top/` for a plain unpack). Every file must be there,
/// byte for byte. A rejecting extractor fails this, and so does one that
/// ignored its destination and wrote somewhere the escape oracle cannot see.
pub fn check_round_trip(top: &Path, entries: &[Entry]) -> Result<(), String> {
    for entry in entries {
        let Kind::File(data) = &entry.kind else {
            continue;
        };
        let relative = entry
            .name
            .strip_prefix(b"top/")
            .ok_or_else(|| format!("not a benign entry: {entry:?}"))?;
        let path = top.join(std::ffi::OsStr::from_bytes(relative));
        match fs::read(&path) {
            Ok(found) if &found == data => {}
            Ok(found) => {
                return Err(format!(
                    "{} holds {found:?}, the archive had {data:?}",
                    path.display()
                ));
            }
            Err(error) => return Err(format!("{} was not extracted: {error}", path.display())),
        }
    }
    Ok(())
}

fn substitute(bytes: &[u8], layout: &Layout) -> Vec<u8> {
    let token = OUTSIDE_TOKEN.as_bytes();
    let replacement = layout.outside.as_os_str().as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    // An absolute name or target is re-rooted at the case's fake root, the way
    // a chroot would see it: an extractor that honours it then writes inside
    // the watched temporary directory instead of onto the host.
    if bytes.first() == Some(&b'/') {
        out.extend_from_slice(layout.base.as_os_str().as_bytes());
    }
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index..].starts_with(token) {
            out.extend_from_slice(replacement);
            index += token.len();
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    out
}

/// Replace the `@OUT` placeholder in names and link targets with the real
/// absolute path of the case's `outside/` directory, and re-root every other
/// absolute name or target at the case's fake root ([`Layout::base`]).
pub fn concretize(entries: &[Entry], layout: &Layout) -> Vec<Entry> {
    entries
        .iter()
        .map(|entry| Entry {
            name: substitute(&entry.name, layout),
            kind: match &entry.kind {
                Kind::Symlink(target) => Kind::Symlink(substitute(target, layout)),
                Kind::Hardlink(target) => Kind::Hardlink(substitute(target, layout)),
                other => other.clone(),
            },
            mode: entry.mode,
        })
        .collect()
}

fn tar_block(header: &mut tar::Header) -> [u8; 512] {
    header.set_cksum();
    let mut block = [0u8; 512];
    block.copy_from_slice(header.as_bytes());
    block
}

fn push_padded(out: &mut Vec<u8>, data: &[u8]) {
    out.extend_from_slice(data);
    let pad = (512 - data.len() % 512) % 512;
    out.extend(std::iter::repeat_n(0u8, pad));
}

fn push_long(out: &mut Vec<u8>, kind: tar::EntryType, value: &[u8]) {
    let mut header = tar::Header::new_gnu();
    header.as_gnu_mut().unwrap().name[..13].copy_from_slice(b"././@LongLink");
    header.set_mode(0o644);
    header.set_entry_type(kind);
    let mut data = value.to_vec();
    data.push(0);
    header.set_size(data.len() as u64);
    out.extend_from_slice(&tar_block(&mut header));
    push_padded(out, &data);
}

/// Serialise entries as a GNU tar stream, writing names and link targets into
/// the header bytes directly — `tar::Builder` refuses `..` and absolute names,
/// which is exactly what this needs to produce. Names longer than the 100-byte
/// field get a GNU long-name record, as a real `tar` would emit.
pub fn tar_bytes(entries: &[Entry]) -> Vec<u8> {
    let mut out = Vec::new();
    for entry in entries {
        if entry.name.len() > 100 {
            push_long(&mut out, tar::EntryType::GNULongName, &entry.name);
        }
        let target = match &entry.kind {
            Kind::Symlink(target) | Kind::Hardlink(target) => Some(target.clone()),
            _ => None,
        };
        if let Some(target) = &target
            && target.len() > 100
        {
            push_long(&mut out, tar::EntryType::GNULongLink, target);
        }
        let mut header = tar::Header::new_gnu();
        {
            let gnu = header.as_gnu_mut().unwrap();
            let len = entry.name.len().min(100);
            gnu.name[..len].copy_from_slice(&entry.name[..len]);
            if let Some(target) = &target {
                let len = target.len().min(100);
                gnu.linkname[..len].copy_from_slice(&target[..len]);
            }
        }
        header.set_mode(entry.mode);
        header.set_mtime(1_700_000_000);
        let data: &[u8] = match &entry.kind {
            Kind::File(data) => {
                header.set_entry_type(tar::EntryType::Regular);
                data
            }
            Kind::Dir => {
                header.set_entry_type(tar::EntryType::Directory);
                &[]
            }
            Kind::Symlink(_) => {
                header.set_entry_type(tar::EntryType::Symlink);
                &[]
            }
            Kind::Hardlink(_) => {
                header.set_entry_type(tar::EntryType::Link);
                &[]
            }
        };
        header.set_size(data.len() as u64);
        out.extend_from_slice(&tar_block(&mut header));
        push_padded(&mut out, data);
    }
    out.extend(std::iter::repeat_n(0u8, 1024));
    out
}

/// [`tar_bytes`], gzip-compressed.
pub fn tar_gz_bytes(entries: &[Entry]) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(&tar_bytes(entries)).unwrap();
    encoder.finish().unwrap()
}

/// The guard rail every case runs inside: a temporary directory with a
/// chain of [`NESTING`] directories in it, each case's own directory at the
/// bottom. Nothing the generator can name climbs out of the chain, and the
/// oracle watches all of it.
///
/// Creating the chain costs a directory per level, so [`run_property`] keeps
/// one fence for all of a property's cases and builds a fresh one only after
/// a case changed something in it: a clean fence is the same fence a new one
/// would be.
pub struct Fence {
    _temp: tempfile::TempDir,
    top: PathBuf,
    bottom: PathBuf,
    dirty: std::rc::Rc<std::cell::Cell<bool>>,
}

impl Fence {
    pub fn new() -> Self {
        use std::os::unix::fs::DirBuilderExt as _;
        let temp = tempfile::Builder::new()
            .prefix("archive-props-")
            .tempdir()
            .unwrap();
        let top = fs::canonicalize(temp.path()).unwrap();
        let mut bottom = top.clone();
        for _ in 0..NESTING {
            bottom.push("d");
        }
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o755)
            .create(&bottom)
            .unwrap();
        Self {
            _temp: temp,
            top,
            bottom,
            dirty: std::rc::Rc::default(),
        }
    }
}

/// Per-case filesystem layout, at the bottom of a [`Fence`]. `top` is the
/// fence's temporary directory and the oracle watches all of it. `base`, the
/// case's own directory, stands in for `/`; `base/dest` is the extraction
/// root, and `base/outside/sentinel` and `base/victim` are what an escape
/// would touch.
///
/// Every directory and file the case is judged on gets an explicit mode, so
/// the oracle judges what the extractor did, not what the runner's umask made
/// of the fixture.
pub struct Layout {
    // Declared first so the case directory goes before a fence it owns.
    _case: tempfile::TempDir,
    _own_fence: Option<Fence>,
    fence_dirty: std::rc::Rc<std::cell::Cell<bool>>,
    pub top: PathBuf,
    pub base: PathBuf,
    pub dest: PathBuf,
    pub outside: PathBuf,
}

impl Layout {
    /// A one-off case in a fence of its own.
    pub fn new() -> Self {
        let fence = Fence::new();
        let mut layout = Self::in_fence(&fence);
        layout._own_fence = Some(fence);
        layout
    }

    /// A fresh case directory at the bottom of `fence`.
    pub fn in_fence(fence: &Fence) -> Self {
        use std::os::unix::fs::PermissionsExt as _;
        let case = tempfile::Builder::new()
            .prefix("case-")
            .tempdir_in(&fence.bottom)
            .unwrap();
        let base = fs::canonicalize(case.path()).unwrap();
        let mkdir = |dir: &Path| {
            fs::create_dir(dir).unwrap();
            fs::set_permissions(dir, fs::Permissions::from_mode(0o755)).unwrap();
        };
        let write = |file: &Path| {
            fs::write(file, b"original").unwrap();
            fs::set_permissions(file, fs::Permissions::from_mode(0o644)).unwrap();
        };
        let dest = base.join("dest");
        let outside = base.join("outside");
        mkdir(&dest);
        mkdir(&outside);
        write(&outside.join("sentinel"));
        write(&base.join("victim"));
        Self {
            _case: case,
            _own_fence: None,
            fence_dirty: fence.dirty.clone(),
            top: fence.top.clone(),
            base,
            dest,
            outside,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Stamp {
    kind: &'static str,
    mode: u32,
    nlink: u64,
    content: Vec<u8>,
    link: Option<PathBuf>,
}

fn stamp(path: &Path) -> Stamp {
    let Ok(meta) = fs::symlink_metadata(path) else {
        return Stamp {
            kind: "missing",
            mode: 0,
            nlink: 0,
            content: Vec::new(),
            link: None,
        };
    };
    let file_type = meta.file_type();
    let (kind, content, link) = if file_type.is_symlink() {
        ("symlink", Vec::new(), fs::read_link(path).ok())
    } else if file_type.is_dir() {
        ("dir", Vec::new(), None)
    } else {
        ("file", fs::read(path).unwrap_or_default(), None)
    };
    Stamp {
        kind,
        mode: meta.mode(),
        nlink: if file_type.is_dir() { 0 } else { meta.nlink() },
        content,
        link,
    }
}

/// Make a real (non-symlink) directory listable again after an archive set
/// its mode to something like `0644`, so the oracle — and the temporary
/// directory's cleanup — can still walk it. Called only after the mode was
/// recorded.
fn ensure_listable(dir: &Path) {
    use std::os::unix::fs::PermissionsExt as _;
    if let Ok(meta) = fs::symlink_metadata(dir)
        && meta.file_type().is_dir()
        && meta.mode() & 0o700 != 0o700
    {
        let _ = fs::set_permissions(dir, fs::Permissions::from_mode(meta.mode() | 0o700));
    }
}

fn read_dir_paths(dir: &Path) -> Vec<PathBuf> {
    ensure_listable(dir);
    fs::read_dir(dir)
        .map(|read| {
            read.filter_map(|entry| entry.ok().map(|e| e.path()))
                .collect()
        })
        .unwrap_or_default()
}

/// Snapshot of every path under `base` except those under the given roots.
/// [`run_case`] passes the whole temporary directory ([`Layout::top`]).
pub fn snapshot_outside(base: &Path, inside: &[&Path]) -> BTreeMap<PathBuf, String> {
    let mut out = BTreeMap::new();
    let mut stack = vec![base.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for path in read_dir_paths(&dir) {
            if inside.iter().any(|root| path.starts_with(root)) {
                continue;
            }
            let stamp = stamp(&path);
            if stamp.kind == "dir" {
                stack.push(path.clone());
            }
            out.insert(path, format!("{stamp:?}"));
        }
    }
    out
}

/// What the oracle found wrong in the tree an extraction left on disk, split
/// by how serious it is.
#[derive(Debug, Default)]
pub struct Violations {
    /// A file inside the root shares an inode with one outside it: writing
    /// the "installed" file writes the outside one. An escape.
    pub shared_inodes: Vec<String>,
    /// A symlink inside the root resolves outside it. Not itself a write
    /// outside, but every later step that follows links (a directory walk, a
    /// copy, a chmod, writing into the installed tree) inherits it.
    pub outward_links: Vec<String>,
    /// setuid/setgid entries in the installed tree, including the root.
    pub setid_modes: Vec<String>,
    /// World-writable entries in the installed tree, including the root.
    pub world_writable: Vec<String>,
}

fn check_mode(path: &Path, mode: u32, violations: &mut Violations) {
    if mode & 0o6000 != 0 {
        violations.setid_modes.push(format!(
            "{} has setuid/setgid bits ({mode:o})",
            path.display()
        ));
    }
    if mode & 0o002 != 0 {
        violations
            .world_writable
            .push(format!("{} is world-writable ({mode:o})", path.display()));
    }
}

/// Walk `root` on the real filesystem (never following links) and report any
/// symlink whose target resolves outside `root`, any hard link shared with a
/// file outside, and any setuid/setgid or world-writable entry.
pub fn inspect_tree(root: &Path, outside_inodes: &[u64]) -> Violations {
    let mut violations = Violations::default();
    let Ok(root_meta) = fs::symlink_metadata(root) else {
        return violations;
    };
    if root_meta.file_type().is_symlink() {
        violations.outward_links.push(format!(
            "the root {} was replaced by a symlink",
            root.display()
        ));
        return violations;
    }
    check_mode(root, root_meta.mode(), &mut violations);
    ensure_listable(root);
    let Ok(canonical_root) = fs::canonicalize(root) else {
        return violations;
    };
    let mut stack = vec![canonical_root.clone()];
    while let Some(dir) = stack.pop() {
        for path in read_dir_paths(&dir) {
            let Ok(meta) = fs::symlink_metadata(&path) else {
                continue;
            };
            let file_type = meta.file_type();
            if file_type.is_symlink() {
                // `canonicalize` resolves the whole chain on disk; a dangling
                // link is reported by its lexical target instead.
                let target = fs::read_link(&path).unwrap_or_default();
                match fs::canonicalize(&path) {
                    Ok(resolved) if !resolved.starts_with(&canonical_root) => {
                        violations.outward_links.push(format!(
                            "symlink {} -> {} resolves outside the root to {}",
                            path.display(),
                            target.display(),
                            resolved.display()
                        ));
                    }
                    Err(_) if target.is_absolute() && !target.starts_with(&canonical_root) => {
                        violations.outward_links.push(format!(
                            "dangling symlink {} -> {} points outside the root",
                            path.display(),
                            target.display()
                        ));
                    }
                    _ => {}
                }
                continue;
            }
            check_mode(&path, meta.mode(), &mut violations);
            if file_type.is_file() && outside_inodes.contains(&meta.ino()) {
                violations.shared_inodes.push(format!(
                    "{} is a hard link to a file outside the root",
                    path.display()
                ));
            }
            if file_type.is_dir() {
                stack.push(path);
            }
        }
    }
    violations
}

/// Inode numbers of the regular files in the outside area, for the hard-link
/// check.
pub fn outside_inodes(layout: &Layout) -> Vec<u64> {
    [layout.outside.join("sentinel"), layout.base.join("victim")]
        .iter()
        .filter_map(|path| fs::symlink_metadata(path).ok().map(|m| m.ino()))
        .collect()
}

/// Generator reach: how often each dangerous shape actually occurred. A
/// property that passes over a generator that never produced the shape proves
/// nothing, so each test prints these and asserts a floor.
#[derive(Default)]
pub struct Reach {
    pub cases: AtomicUsize,
    pub dotdot: AtomicUsize,
    pub absolute: AtomicUsize,
    pub backslash: AtomicUsize,
    pub symlink_out: AtomicUsize,
    pub through_link: AtomicUsize,
    pub hardlink: AtomicUsize,
    pub special_mode: AtomicUsize,
    pub long_name: AtomicUsize,
    pub non_utf8: AtomicUsize,
}

impl Reach {
    pub const fn new() -> Self {
        Self {
            cases: AtomicUsize::new(0),
            dotdot: AtomicUsize::new(0),
            absolute: AtomicUsize::new(0),
            backslash: AtomicUsize::new(0),
            symlink_out: AtomicUsize::new(0),
            through_link: AtomicUsize::new(0),
            hardlink: AtomicUsize::new(0),
            special_mode: AtomicUsize::new(0),
            long_name: AtomicUsize::new(0),
            non_utf8: AtomicUsize::new(0),
        }
    }

    pub fn record(&self, entries: &[Entry]) {
        let bump = |counter: &AtomicUsize, hit: bool| {
            if hit {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        };
        let has = |needle: &[u8], hay: &[u8]| hay.windows(needle.len()).any(|w| w == needle);
        self.cases.fetch_add(1, Ordering::Relaxed);
        bump(
            &self.dotdot,
            entries
                .iter()
                .any(|e| e.name.split(|b| *b == b'/').any(|c| c == b"..")),
        );
        let absolute = |bytes: &[u8]| {
            bytes.first() == Some(&b'/') || bytes.starts_with(OUTSIDE_TOKEN.as_bytes())
        };
        bump(&self.absolute, entries.iter().any(|e| absolute(&e.name)));
        bump(&self.backslash, entries.iter().any(|e| has(b"\\", &e.name)));
        bump(
            &self.symlink_out,
            entries.iter().any(|e| match &e.kind {
                Kind::Symlink(t) => absolute(t) || has(b"..", t),
                _ => false,
            }),
        );
        // A later entry whose name walks through an earlier symlink's name.
        let mut through = false;
        for (index, entry) in entries.iter().enumerate() {
            if let Kind::Symlink(_) = entry.kind {
                let mut prefix = entry.name.clone();
                prefix.push(b'/');
                if entries[index + 1..]
                    .iter()
                    .any(|later| later.name.starts_with(&prefix))
                {
                    through = true;
                }
            }
        }
        bump(&self.through_link, through);
        bump(
            &self.hardlink,
            entries.iter().any(|e| matches!(e.kind, Kind::Hardlink(_))),
        );
        bump(
            &self.special_mode,
            entries.iter().any(|e| e.mode & 0o6002 != 0),
        );
        bump(&self.long_name, entries.iter().any(|e| e.name.len() > 100));
        bump(
            &self.non_utf8,
            entries
                .iter()
                .any(|e| std::str::from_utf8(&e.name).is_err()),
        );
    }

    /// Print the counts and fail if any dangerous shape was generated in fewer
    /// than `floor_percent` of cases.
    pub fn report_and_check(&self, label: &str, floor_percent: usize) {
        let cases = self.cases.load(Ordering::Relaxed).max(1);
        let rows = [
            ("dotdot", &self.dotdot),
            ("absolute", &self.absolute),
            ("backslash", &self.backslash),
            ("symlink_out", &self.symlink_out),
            ("through_link", &self.through_link),
            ("hardlink", &self.hardlink),
            ("special_mode", &self.special_mode),
            ("long_name", &self.long_name),
            ("non_utf8", &self.non_utf8),
        ];
        let mut line = format!("[{label}] generator reach over {cases} cases:");
        for (name, counter) in rows {
            let hits = counter.load(Ordering::Relaxed);
            let _ = write!(line, " {name}={}%", hits * 100 / cases);
        }
        eprintln!("{line}");
        for (name, counter) in rows {
            let hits = counter.load(Ordering::Relaxed);
            assert!(
                hits * 100 / cases >= floor_percent,
                "[{label}] generator reach too low for {name}: {hits}/{cases}"
            );
        }
    }
}

/// Outcome of one extraction case, judged on the real filesystem.
#[derive(Debug)]
pub struct CaseOutcome {
    /// The extractor's error, if it reported one. Failing is allowed; what a
    /// failure leaves behind is still judged.
    pub error: Option<String>,
    /// Writes that landed outside the root: paths created, modified or
    /// removed outside it, and hard links shared with outside files.
    pub escapes: Vec<String>,
    /// The state of the tree left inside the root.
    pub tree: Violations,
}

/// Write `archive_bytes` to `base/<archive_name>`, run `extract`, and judge
/// the result: nothing in the case's temporary directory outside
/// `inside_roots` (which must include the extraction root) may have changed,
/// and the tree under `inspect_root` is
/// walked by [`inspect_tree`]. `extract` receives the layout and the archive
/// path.
pub fn run_case(
    layout: &Layout,
    archive_name: &str,
    archive_bytes: &[u8],
    inside_roots: &[&Path],
    inspect_root: &Path,
    extract: impl FnOnce(&Layout, &Path) -> anyhow::Result<()>,
) -> CaseOutcome {
    let archive = layout.base.join(archive_name);
    fs::write(&archive, archive_bytes).unwrap();
    let mut excluded: Vec<&Path> = inside_roots.to_vec();
    excluded.push(&archive);
    let inodes = outside_inodes(layout);
    let before = snapshot_outside(&layout.top, &excluded);
    let result = extract(layout, &archive);
    let after = snapshot_outside(&layout.top, &excluded);
    let mut escapes = Vec::new();
    for (path, stamp) in &after {
        match before.get(path) {
            None => escapes.push(format!(
                "created outside the root: {} {stamp}",
                path.display()
            )),
            Some(old) if old != stamp => {
                escapes.push(format!(
                    "modified outside the root: {} {old} -> {stamp}",
                    path.display()
                ));
            }
            Some(_) => {}
        }
    }
    for path in before.keys() {
        if !after.contains_key(path) {
            escapes.push(format!("removed outside the root: {}", path.display()));
        }
    }
    // A change above the case's own directory is a change to the fence, which
    // the next case must not inherit.
    if after
        .iter()
        .filter(|(path, stamp)| before.get(*path) != Some(*stamp))
        .map(|(path, _)| path)
        .chain(before.keys().filter(|path| !after.contains_key(*path)))
        .any(|path| !path.starts_with(&layout.base))
    {
        layout.fence_dirty.set(true);
    }
    let mut tree = inspect_tree(inspect_root, &inodes);
    escapes.append(&mut tree.shared_inodes);
    CaseOutcome {
        error: result.err().map(|error| format!("{error:#}")),
        escapes,
        tree,
    }
}

/// What an extractor promises about the modes it leaves behind.
///
/// Both promises are relative to the process: an extractor that creates
/// directories (or runs `tar`) gets modes through the umask, so under a umask
/// that leaves world-write open (`umask 0`) a world-writable entry is what the
/// user asked for, and only setuid/setgid is judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modes {
    /// Not judged.
    Unchecked,
    /// No setuid/setgid entry, and no world-writable one while the umask
    /// clears world-write. Holds whoever runs it.
    Safe,
    /// [`Modes::Safe`] for the system `tar`, which promises it only to an
    /// unprivileged user: run as root, GNU tar restores archived modes,
    /// setuid included. So under effective uid 0 the modes are not judged,
    /// and [`note_mode_scope`] says so.
    SafeUnlessRoot,
}

/// The process umask and effective uid, from `/proc/self/status` (Linux).
/// `None` for either when it cannot be read.
fn process_umask_and_euid() -> (Option<u32>, Option<u32>) {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |key: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .map(str::trim)
    };
    let umask = field("Umask:").and_then(|value| u32::from_str_radix(value, 8).ok());
    let euid = field("Uid:")
        .and_then(|value| value.split_whitespace().nth(1))
        .and_then(|value| value.parse().ok());
    (umask, euid)
}

impl Modes {
    /// Why this run judges less than the whole promise, if it does.
    fn relaxed_because(self) -> Option<String> {
        let (umask, euid) = process_umask_and_euid();
        match self {
            Self::Unchecked => None,
            Self::SafeUnlessRoot if euid == Some(0) => Some(
                "running as root, where the system tar restores archived modes: modes not judged"
                    .to_owned(),
            ),
            Self::Safe | Self::SafeUnlessRoot => {
                umask.filter(|umask| umask & 0o002 == 0).map(|umask| {
                    format!("umask {umask:03o} leaves world-write open: only setuid/setgid judged")
                })
            }
        }
    }

    fn broken_by(self, tree: &Violations) -> bool {
        let (umask, euid) = process_umask_and_euid();
        match self {
            Self::Unchecked => false,
            Self::SafeUnlessRoot if euid == Some(0) => false,
            Self::Safe | Self::SafeUnlessRoot => {
                let umask_clears_world_write = umask.is_none_or(|umask| umask & 0o002 != 0);
                !tree.setid_modes.is_empty()
                    || (umask_clears_world_write && !tree.world_writable.is_empty())
            }
        }
    }
}

/// Print, once per property, which part of the mode promise this run cannot
/// judge (running as root, or under a umask that leaves world-write open), so
/// a pass there is not read as the whole promise holding.
pub fn note_mode_scope(label: &str, modes: Modes) {
    if let Some(reason) = modes.relaxed_because() {
        eprintln!("[{label}] {reason}");
    }
}

/// Which parts of the contract a given extractor promises. Escapes (writes
/// outside the root) are always checked; the tree-hygiene parts are opt-in,
/// because the system-`tar` extractors deliberately recreate symlinks as
/// archived.
#[derive(Clone, Copy, Debug)]
pub struct Expect {
    /// What the modes in the extracted tree may be.
    pub modes: Modes,
    /// No symlink in the extracted tree resolves outside it.
    pub no_outward_links: bool,
}

impl CaseOutcome {
    /// `Ok` when the outcome meets `expect`, else the outcome as text.
    pub fn judge(&self, expect: Expect) -> Result<(), String> {
        let broken = !self.escapes.is_empty()
            || expect.modes.broken_by(&self.tree)
            || (expect.no_outward_links && !self.tree.outward_links.is_empty());
        if broken {
            Err(format!("{self:#?}"))
        } else {
            Ok(())
        }
    }
}

/// Run `check` over `cases` archives from `strategy`, each in a fresh
/// [`Layout`] with the archive already [`concretize`]d into it. Reach is
/// counted on the archive as generated, before absolute names are re-rooted. Returns the
/// shrunk counterexample as text on failure. On success, when `reach_floor`
/// is set, also fails if any dangerous shape occurred in fewer than that
/// percentage of cases — a pass over a generator that never produced the
/// shape would prove nothing.
pub fn run_property(
    label: &str,
    cases: u32,
    strategy: impl Strategy<Value = Vec<Entry>>,
    reach_floor: Option<usize>,
    check: impl Fn(&Layout, &[Entry]) -> Result<(), String>,
) -> Result<(), String> {
    let reach = Reach::new();
    // `PROPTEST_CASES` (proptest's own knob, which an explicit case count
    // would otherwise override) scales every property up for a longer hunt.
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(cases);
    let mut config = proptest::test_runner::Config {
        cases,
        failure_persistence: None,
        ..proptest::test_runner::Config::default()
    };
    // A fixed seed unless `PROPTEST_RNG_SEED` picks another: the reach floors
    // below are statements about these exact cases, so a run cannot fail on
    // seed luck, and any failure reproduces as-is.
    if config.rng_seed == proptest::test_runner::RngSeed::Random {
        config.rng_seed = proptest::test_runner::RngSeed::Fixed(0x00a2_c41e);
    }
    let fence = std::cell::RefCell::new(Fence::new());
    let result = proptest::test_runner::TestRunner::new(config).run(&strategy, |entries| {
        if fence.borrow().dirty.get() {
            *fence.borrow_mut() = Fence::new();
        }
        let layout = Layout::in_fence(&fence.borrow());
        reach.record(&entries);
        let entries = concretize(&entries, &layout);
        check(&layout, &entries).map_err(proptest::test_runner::TestCaseError::fail)
    });
    if let Err(error) = result {
        return Err(format!("[{label}] {error}"));
    }
    if let Some(floor) = reach_floor {
        reach.report_and_check(label, floor);
    }
    Ok(())
}

/// A second calibration extractor that does sanitise names — it keeps only
/// normal components, so no `..` or absolute name gets through — but still
/// follows a symlink planted by an earlier entry. Only the "link, then write
/// through the link" shape can catch it, so this calibrates that shape
/// specifically.
pub fn naive_sanitizing_unpack(archive: &Path, destination: &Path) -> anyhow::Result<()> {
    let file = fs::File::open(archive)?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    for entry in tar.entries()? {
        let mut entry = entry?;
        let raw = entry.path_bytes().into_owned();
        let mut target = destination.to_path_buf();
        for component in Path::new(std::ffi::OsStr::from_bytes(&raw)).components() {
            if let std::path::Component::Normal(part) = component {
                target.push(part);
            }
        }
        if target == destination {
            continue;
        }
        if let Some(parent) = target.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = entry.unpack(&target);
    }
    Ok(())
}

/// The deliberately unsafe extractor used to calibrate the generator: it
/// joins each raw entry name onto the root and unpacks there, following any
/// link an earlier entry planted. If the property does not fail against this,
/// the generator is not reaching the shapes it claims to.
///
/// Absolute names reach it already re-rooted by [`concretize`], so even this
/// extractor writes only inside the case's temporary directory.
pub fn naive_unpack(archive: &Path, destination: &Path) -> anyhow::Result<()> {
    let file = fs::File::open(archive)?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    for entry in tar.entries()? {
        let mut entry = entry?;
        let raw = entry.path_bytes().into_owned();
        let target = destination.join(std::ffi::OsStr::from_bytes(&raw));
        if let Some(parent) = target.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = entry.unpack(&target);
    }
    Ok(())
}
