// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pick which runtime a scenario should activate out of a shared pre-warm tree.
//!
//! Scenarios that only need *a* runtime present point their `data/runtimes` at a
//! shared, install-once tree (see `E2eWorld::use_shared_runtimes`). Serving out of
//! that tree used to need no further wiring: the CLI auto-selects a runtime when
//! exactly one is installed, and the tree held exactly one.
//!
//! That stopped being true once the pre-warm started adopting a newer runtime
//! side by side with the old one. The CLI's auto-selection is deliberately
//! all-or-nothing — with two installed it refuses to guess and serve fails with
//! "no active ROCm runtime is configured" — so a tree holding more than one
//! silently broke every GPU serve scenario. The count is not something the suite
//! controls: it follows whatever the upstream channel index has published.
//!
//! So the scenario names its runtime explicitly instead of relying on the count.
//! The pre-warm activates the runtime it installed and the CLI records that in
//! `<runtimes>/active.json`, which lives *inside* the shared tree and is therefore
//! visible through the symlink even though each scenario keeps its own config dir.
//! Reading it back is what makes selection deterministic for any tree contents.
//!
//! Naming a runtime is not the same as naming a *usable* one, which is the second
//! half of this. A registry entry can outlive the folder it points at, and a name
//! the CLI rejects fails the scenario just as surely as no name at all — so what
//! gets named is filtered down to the entries `activate` can accept.

use std::path::Path;

/// The runtime key a scenario should activate for a shared tree at `runtimes_dir`.
///
/// `Some(key)` means "run `rocm runtimes activate <key>`"; `None` means there is
/// nothing to name and the caller should leave selection to the CLI — either the
/// tree is empty (the scenario installs its own) or it holds exactly one runtime,
/// which the CLI already auto-selects.
///
/// Chooses only among [`activatable_runtime_keys`] — entries the CLI would
/// actually accept — so neither branch below can name a runtime that fails to
/// activate.
///
/// Prefers `active.json` because that records the runtime the pre-warm actually
/// installed and verified — but only when the registry still holds it, so a
/// marker left behind by a pruned runtime doesn't strand the scenario. Falls back
/// to the sole registry manifest so a tree written before the pre-warm learned to
/// activate still resolves. When neither names one runtime, returns `None` rather
/// than picking arbitrarily: choosing the wrong one of several would serve against
/// an unintended ROCm version, and a clear "no active runtime" failure beats a
/// silently mismatched pass.
#[must_use]
pub fn runtime_key_to_activate(runtimes_dir: &Path) -> Option<String> {
    let installed = activatable_runtime_keys(runtimes_dir);
    active_runtime_key(runtimes_dir)
        .filter(|key| installed.iter().any(|installed| installed == key))
        .or_else(|| match installed.as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        })
}

/// The `runtime_key` recorded in `<runtimes_dir>/active.json`, if it names one.
fn active_runtime_key(runtimes_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(runtimes_dir.join("active.json")).ok()?;
    let key = serde_json::from_str::<serde_json::Value>(&text)
        .ok()?
        .get("runtime_key")?
        .as_str()?
        .trim()
        .to_owned();
    (!key.is_empty()).then_some(key)
}

/// The registry keys `rocm runtimes activate` would actually accept.
///
/// A registry entry is not enough: the CLI refuses to activate a runtime whose
/// recorded `install_root` is gone ("install root is missing"). That is not
/// hypothetical here — a scenario that installs through its own `data/runtimes`
/// symlink records its per-scenario temp dir as the install root, and the folder
/// dies with the scenario, leaving a registry entry pointing at nothing (see
/// rocm-cli#315/#316). The pre-warm normally evicts those, but its repair is
/// deliberately non-fatal and skips the whole tree when `runtimes list` itself
/// fails — which a poisoned entry makes it do. So the suite must expect to meet
/// one and step over it rather than name it and fail.
///
/// Judged by the same rule the pre-warm's own repair uses: an install root
/// inside this tree is sound, one outside it is a corpse. Checking existence
/// alone would be wrong — on a runner where a foreign path happens to exist the
/// scenario would serve against a runtime outside the shared tree.
fn activatable_runtime_keys(runtimes_dir: &Path) -> Vec<String> {
    let roots = comparable_roots(runtimes_dir);
    registry_runtime_keys(runtimes_dir)
        .into_iter()
        .filter(|key| {
            let Ok(text) =
                std::fs::read_to_string(runtimes_dir.join("registry").join(format!("{key}.json")))
            else {
                return false;
            };
            let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
                return false;
            };
            // No recorded root: nothing to disqualify it on, so keep it and let
            // the CLI have the final say.
            let Some(root) = json.get("install_root").and_then(|v| v.as_str()) else {
                return true;
            };
            let root = Path::new(root);
            roots.iter().any(|base| root.starts_with(base))
        })
        .collect()
}

/// Every spelling of `runtimes_dir` a recorded `install_root` might match.
///
/// The two sides are resolved-vs-as-given by construction, so one comparison is
/// not enough. The CLI canonicalizes an install root before recording it, while
/// `E2E_SHARED_RUNTIMES_DIR` reaches us verbatim — `validated_shared_dir` checks
/// that it is absolute and free of `..`, and deliberately does not resolve it.
/// Reach the tree through a symlinked component of the workspace and the two
/// spellings differ, so *every healthy* entry looks out-of-tree: selection comes
/// back empty and the serve fails with the very error this module exists to
/// prevent — silently, since declining to activate is best-effort by design.
///
/// This mirrors `xtask::e2e_prewarm::assess`, which compares against both
/// spellings for the same reason. Windows needs the third: `canonicalize` yields
/// a `\\?\` verbatim path there and the CLI records a plain one, so the prefix
/// has to come off or the comparison fails on it rather than on the folder. That
/// is the same trap `runtime_steps::linked_runtimes_target` documents, and 8.3
/// short paths (which `canonicalize` expands) are why resolving matters even
/// when no symlink is involved.
fn comparable_roots(runtimes_dir: &Path) -> Vec<std::path::PathBuf> {
    let mut roots = vec![runtimes_dir.to_path_buf()];
    let Ok(resolved) = runtimes_dir.canonicalize() else {
        return roots;
    };
    for candidate in [strip_verbatim_prefix(&resolved), resolved] {
        if !roots.contains(&candidate) {
            roots.push(candidate);
        }
    }
    roots
}

/// Drop Windows' `\\?\` verbatim prefix, which never `starts_with`-matches the
/// plain path the CLI records. A no-op on every other platform.
fn strip_verbatim_prefix(path: &Path) -> std::path::PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        return std::path::PathBuf::from(format!(r"\\{rest}"));
    }
    match text.strip_prefix(r"\\?\") {
        Some(rest) => std::path::PathBuf::from(rest),
        None => path.to_path_buf(),
    }
}

/// Every runtime key present in the registry, empty when it is absent.
///
/// Public so a caller that declines to activate can name what it found: "no
/// runtime to activate" is only actionable alongside the list it chose from.
/// Deliberately unfiltered — a diagnostic should report the tree as it is, so a
/// key skipped by [`activatable_runtime_keys`] still shows up in the message.
#[must_use]
pub fn registry_runtime_keys(runtimes_dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(runtimes_dir.join("registry")) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                // Runtime key = the manifest file stem, the same convention
                // `capability::active_runtime_install_root` relies on.
                path.file_stem()?.to_str().map(str::to_owned)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().expect("path has a parent")).expect("create dir");
        std::fs::write(path, body).expect("write file");
    }

    /// A runtime installed in place: its recorded root lives inside the tree.
    ///
    /// This is the shape the CLI really writes, so it is what the tests use.
    /// A manifest with no `install_root` at all takes the "nothing to judge on"
    /// branch and would pass whether or not the root is checked; since
    /// `InstalledRuntimeManifest::install_root` is non-optional and
    /// `load_runtime_manifests` drops anything that fails to deserialize, such a
    /// manifest cannot arise in CI either. Testing against it proved nothing.
    fn installed(dir: &Path, key: &str) {
        let root = dir.join("wheel").join(key);
        std::fs::create_dir_all(&root).expect("create install root");
        write(
            &dir.join("registry").join(format!("{key}.json")),
            &serde_json::json!({ "runtime_key": key, "install_root": root }).to_string(),
        );
    }

    /// A runtime whose recorded root is a dead per-scenario temp dir — the
    /// rocm-cli#315 poisoning the pre-warm failed to evict.
    fn poisoned(dir: &Path, key: &str) {
        write(
            &dir.join("registry").join(format!("{key}.json")),
            &serde_json::json!({
                "runtime_key": key,
                "install_root": format!("/tmp/rocm-e2e-gone/data/runtimes/wheel/{key}"),
            })
            .to_string(),
        );
    }

    fn active(dir: &Path, key: &str) {
        write(
            &dir.join("active.json"),
            &format!(r#"{{"runtime_key": "{key}"}}"#),
        );
    }

    /// The regression this module exists for: a tree the pre-warm grew a second
    /// runtime in must still name one, or every GPU serve scenario fails with
    /// "no active ROCm runtime is configured".
    #[test]
    fn names_the_active_runtime_when_several_are_installed() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let dir = tmp.path();
        installed(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
        installed(dir, "release-wheel-multi-arch-7-14-0");
        active(dir, "release-wheel-multi-arch-7-14-0");

        assert_eq!(
            runtime_key_to_activate(dir).as_deref(),
            Some("release-wheel-multi-arch-7-14-0")
        );
    }

    /// A tree from before the pre-warm activated: one runtime, no marker. The
    /// CLI auto-selects here, so naming it is optional — but resolving it keeps
    /// the step's behaviour identical whether or not a marker was written.
    #[test]
    fn falls_back_to_the_sole_manifest_without_a_marker() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let dir = tmp.path();
        installed(dir, "release-wheel-gfx94x-dcgpu-7-13-0");

        assert_eq!(
            runtime_key_to_activate(dir).as_deref(),
            Some("release-wheel-gfx94x-dcgpu-7-13-0")
        );
    }

    /// Several runtimes and no marker: refuse to guess. Activating an arbitrary
    /// one would serve against an unintended ROCm version and pass, which is
    /// worse than the CLI's own explicit failure.
    #[test]
    fn refuses_to_guess_between_several_without_a_marker() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let dir = tmp.path();
        installed(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
        installed(dir, "release-wheel-multi-arch-7-14-0");

        assert_eq!(runtime_key_to_activate(dir), None);
    }

    /// An empty tree is the first scenario on a cold runner; it installs its own
    /// runtime, so there is nothing to activate beforehand.
    #[test]
    fn names_nothing_for_an_empty_tree() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        assert_eq!(runtime_key_to_activate(tmp.path()), None);
    }

    /// A marker that survives the runtime it names (a prune, a hand-cleaned tree)
    /// must not strand the scenario: fall through to the registry.
    #[test]
    fn ignores_a_marker_naming_no_installed_runtime() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let dir = tmp.path();
        installed(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
        active(dir, "release-wheel-multi-arch-7-14-0");

        assert_eq!(
            runtime_key_to_activate(dir).as_deref(),
            Some("release-wheel-gfx94x-dcgpu-7-13-0")
        );
    }

    /// Corrupt or half-written markers are treated as absent, not fatal: the
    /// registry still answers the question.
    #[test]
    fn treats_an_unreadable_marker_as_absent() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let dir = tmp.path();
        installed(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
        write(&dir.join("active.json"), "{ not json");

        assert_eq!(
            runtime_key_to_activate(dir).as_deref(),
            Some("release-wheel-gfx94x-dcgpu-7-13-0")
        );
    }

    /// The regression from the first attempt at this fix: the sole *registry*
    /// entry was a corpse pointing at a dead per-scenario temp dir, so naming it
    /// made every activate fail with "install root is missing" — the suite
    /// traded one silent breakage for a louder one. Skip it and name the runtime
    /// that is really there.
    #[test]
    fn skips_a_runtime_whose_install_root_left_the_tree() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let dir = tmp.path();
        poisoned(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
        installed(dir, "release-wheel-multi-arch-7-14-0");

        assert_eq!(
            runtime_key_to_activate(dir).as_deref(),
            Some("release-wheel-multi-arch-7-14-0")
        );
    }

    /// Reaching the tree through a symlinked parent must not disqualify every
    /// healthy runtime in it.
    ///
    /// The CLI records a canonicalized `install_root`; `E2E_SHARED_RUNTIMES_DIR`
    /// arrives unresolved. Compare only the spelling we were handed and the two
    /// never match, so selection comes back empty and the serve fails with the
    /// error this module exists to prevent — the original bug through a different
    /// door, and silent. Approximates a symlinked component of the runner's
    /// workspace, which is the shape that would trigger it in CI.
    #[test]
    fn resolves_a_tree_reached_through_a_symlinked_parent() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let real = tmp.path().join("real");
        std::fs::create_dir_all(&real).expect("create real tree");
        // The manifest records the resolved root, as the CLI writes it.
        installed(&real, "release-wheel-multi-arch-7-14-0");

        let link = tmp.path().join("link");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        #[cfg(windows)]
        std::os::windows::fs::symlink_dir(&real, &link).expect("symlink");

        // Selection is handed the unresolved spelling, exactly as CI would.
        assert_eq!(
            runtime_key_to_activate(&link).as_deref(),
            Some("release-wheel-multi-arch-7-14-0"),
            "a tree reached through a symlink must still resolve its runtimes"
        );
    }

    /// The symlink tolerance must not become "any path anywhere": a corpse
    /// pointing outside the tree stays disqualified when the tree is reached
    /// through a link, or the fix above would have re-admitted every poisoned
    /// entry it was meant to skip.
    #[test]
    fn still_skips_a_corpse_when_the_tree_is_reached_through_a_symlink() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let real = tmp.path().join("real");
        std::fs::create_dir_all(&real).expect("create real tree");
        poisoned(&real, "release-wheel-gfx94x-dcgpu-7-13-0");

        let link = tmp.path().join("link");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        #[cfg(windows)]
        std::os::windows::fs::symlink_dir(&real, &link).expect("symlink");

        assert_eq!(runtime_key_to_activate(&link), None);
    }

    /// A marker naming a poisoned runtime must not override a sound one either:
    /// the pre-warm activates before a scenario poisons the tree, so the stale
    /// marker outlives the runtime it names.
    #[test]
    fn ignores_a_marker_naming_a_runtime_that_left_the_tree() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let dir = tmp.path();
        poisoned(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
        installed(dir, "release-wheel-multi-arch-7-14-0");
        active(dir, "release-wheel-gfx94x-dcgpu-7-13-0");

        assert_eq!(
            runtime_key_to_activate(dir).as_deref(),
            Some("release-wheel-multi-arch-7-14-0")
        );
    }

    /// Every runtime is a corpse: name none. Letting the CLI report "no active
    /// ROCm runtime is configured" beats an activate that cannot succeed.
    #[test]
    fn names_nothing_when_every_runtime_left_the_tree() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let dir = tmp.path();
        poisoned(dir, "release-wheel-gfx94x-dcgpu-7-13-0");

        assert_eq!(runtime_key_to_activate(dir), None);
    }

    /// The diagnostic reports the tree as it is: a skipped runtime still has to
    /// appear, or "no runtime to activate" names an empty tree that isn't empty.
    #[test]
    fn reports_a_skipped_runtime_in_the_registry_listing() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let dir = tmp.path();
        poisoned(dir, "release-wheel-gfx94x-dcgpu-7-13-0");

        assert_eq!(
            registry_runtime_keys(dir),
            vec!["release-wheel-gfx94x-dcgpu-7-13-0"]
        );
    }

    /// A marker with a blank key names nothing.
    #[test]
    fn treats_a_blank_marker_key_as_absent() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let dir = tmp.path();
        installed(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
        installed(dir, "release-wheel-multi-arch-7-14-0");
        active(dir, "   ");

        assert_eq!(runtime_key_to_activate(dir), None);
    }
}
