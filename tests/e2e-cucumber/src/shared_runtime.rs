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

use std::path::Path;

/// The runtime key a scenario should activate for a shared tree at `runtimes_dir`.
///
/// `Some(key)` means "run `rocm runtimes activate <key>`"; `None` means there is
/// nothing to name and the caller should leave selection to the CLI — either the
/// tree is empty (the scenario installs its own) or it holds exactly one runtime,
/// which the CLI already auto-selects.
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
    let installed = registry_runtime_keys(runtimes_dir);
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

/// Every runtime key present in the registry, empty when it is absent.
///
/// Public so a caller that declines to activate can name what it found: "no
/// runtime to activate" is only actionable alongside the list it chose from.
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

    fn manifest(dir: &Path, key: &str) {
        write(&dir.join("registry").join(format!("{key}.json")), "{}");
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
        manifest(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
        manifest(dir, "release-wheel-multi-arch-7-14-0");
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
        manifest(dir, "release-wheel-gfx94x-dcgpu-7-13-0");

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
        manifest(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
        manifest(dir, "release-wheel-multi-arch-7-14-0");

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
        manifest(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
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
        manifest(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
        write(&dir.join("active.json"), "{ not json");

        assert_eq!(
            runtime_key_to_activate(dir).as_deref(),
            Some("release-wheel-gfx94x-dcgpu-7-13-0")
        );
    }

    /// A marker with a blank key names nothing.
    #[test]
    fn treats_a_blank_marker_key_as_absent() {
        let tmp = tempfile::TempDir::with_prefix("shared-runtime-").expect("temp dir");
        let dir = tmp.path();
        manifest(dir, "release-wheel-gfx94x-dcgpu-7-13-0");
        manifest(dir, "release-wheel-multi-arch-7-14-0");
        active(dir, "   ");

        assert_eq!(runtime_key_to_activate(dir), None);
    }
}
