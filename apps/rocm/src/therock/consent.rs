// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The consent gate for an install that would displace the active default
//! runtime. `active_default_runtime_relation` reads the registry and the
//! config and says what the install would replace, or `None` when nothing is
//! the active default, and it fails closed (a relation naming what could not
//! be resolved) rather than call an unresolvable active default a fresh
//! install. `SdkInstallConsent` and `SdkInstallApprovalSource` say how consent
//! is obtained and from where, and `sdk_install_approval` turns those and
//! whether a terminal is attached into one of proceed, prompt or refuse
//! (`SdkInstallApproval`). The words each outcome prints come from
//! `preapproved_install_line`, `fresh_install_line`,
//! `refuse_non_interactive_message` and `confirm_overwrite_existing_sdk`, which
//! reads the answer through `prompt_yes_no`.
//!
//! Builds on `source` and `version`, and reaches the registry through
//! `runtime_registry` rather than `runtimes_cmd`.

use super::source::TheRockChannel;
use super::version::{compare_version_strings, runtime_version_display, version_relation};
use crate::runtime_registry::{
    InstalledRuntimeManifest, load_runtime_manifests_reporting_unparsed,
};
use anyhow::{Context, Result};
use rocm_core::{AppPaths, RocmCliConfig};
use std::cmp::Ordering;
use std::path::PathBuf;

/// Describe the managed runtime this install would displace as the active
/// default: the one the runtime config's `active_runtime_key` (or an
/// unambiguous `default_runtime_id`) currently resolves to. Returns `None` only
/// when neither config pointer points at an active default — a genuinely fresh
/// install, where the new runtime takes a slot nothing occupies and there is
/// nothing to consent to.
///
/// Deliberately NOT scoped to the target family and channel. Activation is
/// global: `finalize_successful_sdk_install` activates whatever was just
/// installed regardless of family or channel, so `rocm install sdk --family
/// gfx120X-all` displaces an active `gfx110X-all` runtime just as surely as a
/// same-family upgrade does. A family/channel-scoped gate would wave exactly
/// that case through unconfirmed while every message promised otherwise.
///
/// Errors reading the manifest directory or the config are propagated rather
/// than treated as "no active default": silently falling back to a
/// fresh-install verdict on a read error would skip the confirmation gate
/// precisely when we are least sure what is currently active.
///
/// The same reasoning covers the ways an active default can fail to resolve
/// without any I/O error at all, and all of them used to reach the
/// fresh-install verdict. `current_runtime_manifest` resolves through two
/// pointers — `active_runtime_key` first, then `default_runtime_id` — and each
/// has its own failure shapes:
///
/// * a registry manifest that reads fine but does not deserialize —
///   `load_runtime_manifests` drops those silently, so a manifest written by an
///   older binary (`family_source`, `selected_artifact_url` and
///   `installed_at_unix_ms` carry no `#[serde(default)]`) vanishes from the
///   list and `current_runtime_manifest` misses;
/// * `active_runtime_key` naming a runtime whose manifest is not in the
///   registry at all;
/// * `default_runtime_id` matching no installed manifest — `runtime_id` is not
///   version-scoped, and `rocm config set-default-runtime` stores whatever it
///   is handed without validating it against the registry;
/// * `default_runtime_id` matching more than one installed manifest, which is
///   the ordinary state once two versions of the same family are installed,
///   because they share the one `therock-<channel>:<family>` id.
///   `current_runtime_manifest` resolves only an exactly-one match, so both the
///   zero-match and the multi-match shapes arrive here.
///
/// Every one of these is something `rocm runtimes list` already calls out, as
/// `active_status: missing manifest for active_runtime_key=...`,
/// `active_status: missing manifest for active_runtime_id=...` or
/// `active_status: ambiguous runtime_id=...`, so proceeding as a fresh install
/// would have one CLI assert both that a runtime is active and that none is.
/// These fail closed into the consent gate rather than into a hard error, so
/// `--approve-replacing-active-default` (or `--yes`) still gets an operator
/// through a stale or ambiguous config.
pub(super) fn active_default_runtime_relation(
    paths: &AppPaths,
    channel: TheRockChannel,
    family: &str,
    resolved_version: &str,
) -> Result<Option<String>> {
    let (manifests, unparsed) = load_runtime_manifests_reporting_unparsed(paths)?;
    let config = RocmCliConfig::load(paths)?;
    let Some(active) = crate::runtime_registry::current_runtime_manifest(&config, &manifests)
    else {
        return Ok(unresolved_active_default_relation_text(
            config.active_runtime_key.as_deref(),
            config.default_runtime_id.as_deref(),
            crate::runtime_registry::default_runtime_id_matches(&config, &manifests).len(),
            &unparsed,
        ));
    };
    Ok(Some(active_default_relation_text(
        active,
        channel,
        family,
        resolved_version,
    )))
}

/// Wording for the fail-closed half of [`active_default_runtime_relation`]:
/// nothing resolved, but the on-disk state says something should have.
///
/// `None` here is the only genuinely fresh verdict, and it needs *neither*
/// config pointer to claim an active default: with no `active_runtime_key` and
/// no `default_runtime_id`, neither pointer asserts that a runtime is active,
/// so an unparsable manifest is a registry wart rather than a displacement
/// risk and demanding a consent flag would be a false positive on a genuinely
/// fresh install. Once a pointer does claim something, anything unresolved
/// names what could not be resolved, because the relation string is what the
/// prompt, the preapproved progress line and the non-interactive refusal all
/// print, and "unknown" is the honest answer the operator needs to see.
///
/// `default_runtime_id_match_count` is the number of installed manifests whose
/// `runtime_id` equals `default_runtime_id`. The caller only reaches this
/// function when resolution failed, so that count is 0 (dangling) or greater
/// than 1 (ambiguous) — an exactly-one match is what
/// `current_runtime_manifest` resolves successfully.
///
/// `unparsed` is reported two different ways on purpose. An entry is only the
/// *cause* of an unresolved `active_runtime_key` when it is that key's own
/// manifest; every other unreadable entry is a separate registry wart that
/// happens to be visible at the same time, so it is appended as a suffix rather
/// than named as the reason. Blaming an unrelated file — the ordinary
/// older-binary manifest is exactly that — would point the operator at the
/// wrong path while a deleted runtime went unmentioned.
///
/// Both config pointers are named when both are set. `current_runtime_manifest`
/// tries `active_runtime_key` first and falls through to `default_runtime_id`,
/// so arriving here means *both* failed, and the message is what the prompt,
/// the preapproved progress line and the non-interactive refusal print: it has
/// to name everything that could not be resolved, not just the first pointer.
fn unresolved_active_default_relation_text(
    active_runtime_key: Option<&str>,
    default_runtime_id: Option<&str>,
    default_runtime_id_match_count: usize,
    unparsed: &[PathBuf],
) -> Option<String> {
    let unparsed_text = || {
        unparsed
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let unparsed_suffix = || {
        if unparsed.is_empty() {
            String::new()
        } else {
            format!("; unreadable runtime manifests: {}", unparsed_text())
        }
    };
    let non_empty = |value: Option<&str>| {
        value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    };
    // The registry stores each manifest at `<runtime_key>.json`
    // (`runtime_manifest_path`), so the file stem is the key. Compared
    // case-insensitively because that is how `current_runtime_manifest` matches
    // `active_runtime_key` against `runtime_key`.
    let key_manifest_is_unparsable = |key: &str| {
        unparsed.iter().any(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| stem.eq_ignore_ascii_case(key))
        })
    };
    // Reached only when `active_runtime_key` is also set and also unresolved, so
    // this is always a continuation of the key clause, never a sentence by
    // itself. The count is 0 or >1 for the reason given on the parameter.
    //
    // Pinned rather than left in prose because the else branch below states
    // "no installed runtime manifest matches it" as fact: at exactly 1 that
    // sentence is false, and the operator would be told nothing matched the
    // recorded default while a manifest did — sending them to re-register a
    // runtime that is already there.
    debug_assert_ne!(
        default_runtime_id_match_count, 1,
        "an exactly-one match is what `current_runtime_manifest` resolves, so this function is unreachable with it"
    );
    let also_unresolved_id = |id: &str| {
        if default_runtime_id_match_count > 1 {
            format!(
                "; the recorded default runtime_id `{id}` does not settle it either, because {default_runtime_id_match_count} installed runtime manifests match it"
            )
        } else {
            format!(
                "; the recorded default runtime_id `{id}` does not settle it either, because no installed runtime manifest matches it"
            )
        }
    };

    match (non_empty(active_runtime_key), non_empty(default_runtime_id)) {
        (Some(key), id) => {
            let cause = if key_manifest_is_unparsable(&key) {
                format!("recorded as `{key}`, but its manifest could not be read")
            } else {
                format!("recorded as `{key}`, but no installed runtime manifest matches it")
            };
            Some(format!(
                "{cause}, so what is currently active cannot be determined{}{}",
                id.as_deref().map(also_unresolved_id).unwrap_or_default(),
                unparsed_suffix()
            ))
        }
        (None, Some(id)) if default_runtime_id_match_count > 1 => Some(format!(
            "recorded as runtime_id `{id}`, which {default_runtime_id_match_count} installed runtime manifests match, so which one is currently active cannot be determined{}",
            unparsed_suffix()
        )),
        (None, Some(id)) => Some(format!(
            "recorded as runtime_id `{id}`, but no installed runtime manifest matches it, so what is currently active cannot be determined{}",
            unparsed_suffix()
        )),
        (None, None) => None,
    }
}

/// Pure wording for [`active_default_runtime_relation`].
///
/// Two shapes, because the two cases are not the same event. When the active
/// default is the same family and channel the install targets, the version
/// comparison is meaningful and the user wants to read "upgrade"/"downgrade"/
/// "reinstall". When it is a different family or channel, comparing versions
/// would invent a relation between two unrelated runtimes, so the text names
/// what is actually being displaced instead.
pub(super) fn active_default_relation_text(
    active: &InstalledRuntimeManifest,
    channel: TheRockChannel,
    family: &str,
    resolved_version: &str,
) -> String {
    if active.family == family && active.channel == channel.as_str() {
        // "reinstall" is a claim about versions, not strings: `7.0.0-rc1` and
        // `7.0.0rc1` are one version and must not read as an upgrade. When
        // either side cannot be identified (`version_relation` is `None`) there
        // is no better word available here, so the total order still decides
        // rather than inventing a fourth one; an unreadable version sorts below
        // every readable one, so it reads as the older side.
        let relation = match version_relation(resolved_version, &active.version)
            .unwrap_or_else(|| compare_version_strings(resolved_version, &active.version))
        {
            Ordering::Greater => "upgrade",
            Ordering::Less => "downgrade",
            Ordering::Equal => "reinstall",
        };
        format!(
            "{relation} from installed {installed} ({key})",
            installed = runtime_version_display(&active.version),
            key = active.runtime_key
        )
    } else {
        format!(
            "replaces active default {installed} for family {active_family} on the {active_channel} channel ({key})",
            installed = runtime_version_display(&active.version),
            active_family = active.family,
            active_channel = active.channel,
            key = active.runtime_key
        )
    }
}

/// Where an already-granted approval for displacing the active default came
/// from. Carried rather than collapsed to a bare bool so the line the CLI
/// prints can name the real source: `rocm update --apply` takes its approval
/// from the runtime the user selected, not from a flag, so a message crediting
/// `--yes` would name an approval that was never given. (`rocm update` does
/// accept a `--yes` flag, but it is inert by its own documentation — applying
/// never prompts — so it grants nothing to credit.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SdkInstallApprovalSource {
    /// The user passed `--yes` to `rocm install sdk`, which also approves
    /// installing required system packages with `sudo`.
    AssumeYes,
    /// `--approve-replacing-active-default` was passed: the narrow consent, and
    /// the one ROCm CLI's own terminal-less surfaces (chat, MCP, `rocmd`, the
    /// dashboard, onboarding) inject. Kept distinct from `AssumeYes` so the
    /// install log names the flag that was actually given — those surfaces never
    /// pass `--yes`, and a line crediting it would tell a reader that consent to
    /// run `sudo` had been granted when it was not.
    ApproveReplacingActiveDefault,
    /// `rocm update --apply`, which owns its own consent: the user named the
    /// runtime to update, so replacing it is the operation asked for. Non-
    /// interactive by construction, so it must never reach a prompt.
    ///
    /// `activates` records whether `--activate` was given. Only then does the
    /// new install become the active default; without it `apply_runtime_update`
    /// leaves the current default alone and prints a `runtimes activate` hint.
    UpdateApply { activates: bool },
}

/// How consent for displacing the active default runtime is obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SdkInstallConsent {
    /// Ask the user: prompt when a terminal is attached, refuse when not.
    Ask,
    /// Already granted before the install started, by the named source.
    Preapproved(SdkInstallApprovalSource),
}

/// Whether a real SDK install needs the user's approval before it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SdkInstallApproval {
    /// No active default runtime — nothing is displaced, so install without asking.
    ProceedFresh,
    /// An active default runtime exists and consent was already granted —
    /// displace it without asking, crediting the source.
    ProceedApproved(SdkInstallApprovalSource),
    /// An active default runtime exists and there is a terminal — prompt before
    /// displacing it.
    PromptOverwrite,
    /// An active default runtime exists but there is no terminal and no
    /// preapproval — refuse.
    RefuseNonInteractive,
}

/// Decide whether an SDK install proceeds, prompts, or is refused. Installs with
/// no active default runtime to displace (`existing == false`) always proceed;
/// displacing the active default needs consent, and outside an interactive
/// terminal that consent must have been granted up front.
pub(super) const fn sdk_install_approval(
    existing: bool,
    consent: SdkInstallConsent,
    interactive: bool,
) -> SdkInstallApproval {
    match (existing, consent) {
        (false, _) => SdkInstallApproval::ProceedFresh,
        (true, SdkInstallConsent::Preapproved(source)) => {
            SdkInstallApproval::ProceedApproved(source)
        }
        (true, SdkInstallConsent::Ask) if interactive => SdkInstallApproval::PromptOverwrite,
        (true, SdkInstallConsent::Ask) => SdkInstallApproval::RefuseNonInteractive,
    }
}

/// The progress line printed when an install displaces the active default
/// without asking. Pure so the exact wording is pinned by unit tests.
///
/// Each arm states only what that caller will actually do. `update --apply`
/// without `--activate` installs beside the active default and leaves it alone,
/// so claiming the new install "becomes the active default runtime" there would
/// be false — and crediting `--yes` would claim an approval the update path
/// never received, since `rocm update`'s `--yes` is inert and its approval
/// comes from the runtime selection instead.
///
/// `pub(crate)` so the `Update` subcommand's own test can pin that invariant at
/// the site of the flag it must not credit.
pub(crate) fn preapproved_install_line(
    source: SdkInstallApprovalSource,
    relation: &str,
    resolved_display: &str,
) -> String {
    match source {
        SdkInstallApprovalSource::AssumeYes => format!(
            "Approved by --yes: an existing ROCm SDK is the active default runtime ({relation}); installing ROCm {resolved_display}, which becomes the active default runtime."
        ),
        SdkInstallApprovalSource::ApproveReplacingActiveDefault => format!(
            "Approved by --approve-replacing-active-default: an existing ROCm SDK is the active default runtime ({relation}); installing ROCm {resolved_display}, which becomes the active default runtime."
        ),
        SdkInstallApprovalSource::UpdateApply { activates: true } => format!(
            "Requested by `rocm update --apply --activate`: an existing ROCm SDK is the active default runtime ({relation}); installing ROCm {resolved_display}, which becomes the active default runtime."
        ),
        SdkInstallApprovalSource::UpdateApply { activates: false } => format!(
            "Requested by `rocm update --apply`: an existing ROCm SDK is the active default runtime ({relation}); installing ROCm {resolved_display} alongside it. The active default runtime is unchanged; re-run with --activate, or use `rocm runtimes activate`, to switch to it."
        ),
    }
}

/// The progress line printed when nothing is displaced. States only that no
/// active default exists, because whether this install *becomes* the active
/// default depends on the caller: `rocm install sdk` activates what it
/// installed, `rocm update --apply` without `--activate` does not.
pub(super) fn fresh_install_line(resolved_display: &str, family: &str) -> String {
    format!(
        "No active ROCm SDK runtime is configured; installing ROCm SDK {resolved_display} for family {family}."
    )
}

/// The error raised when the active default would be displaced but there is no
/// terminal to confirm it and no preapproval.
///
/// This is the only message a script or CI job sees when it hits this gate, so
/// it names the narrow flag first: that is the whole consent the caller needs
/// here, and recommending `--yes` instead would hand an unattended caller the
/// second consent it carries — approval to install system packages with `sudo`,
/// whose password prompt a job with no terminal cannot answer. `--yes` is still
/// named, because a user at a terminal who wants both should not have to
/// discover it elsewhere.
pub(super) fn refuse_non_interactive_message(relation: &str) -> String {
    format!(
        "an existing ROCm SDK is the active default runtime ({relation}); continuing would make the newly installed ROCm the active default runtime instead. Re-run with --approve-replacing-active-default to approve this non-interactively, for example `rocm install sdk --approve-replacing-active-default`. Use --yes instead only if you also want to approve installing required system packages with sudo, which needs a terminal to answer a password prompt"
    )
}

/// Interactive confirmation gate for displacing the active default managed
/// runtime. Prints what would be replaced, then reads a yes/no answer from
/// stdin. Only reached when an active default runtime exists, consent was not
/// preapproved, and a terminal is attached (see `sdk_install_approval`).
///
/// "Displacing" rather than "overwriting" is deliberate: in the default managed
/// install root `runtime_key` embeds the resolved version, so an upgrade or
/// downgrade lands in its own install root with its own manifest and the previous
/// install stays on disk — what changes is which runtime is the active default.
/// Only a same-version reinstall reuses the same root. The prompt says so instead
/// of claiming a deletion that does not happen.
///
/// `--prefix` is the exception and the prompt does not claim otherwise:
/// `resolved_install_root` uses the given folder verbatim for every version, so
/// a second install into one prefix does replace the first in place (and
/// `ensure_uv_venv` will `remove_dir_all` it outright if the existing venv's
/// python no longer answers `--version`). The gate is unchanged either way —
/// what is being consented to is the change of active default, not a deletion.
pub(super) fn confirm_overwrite_existing_sdk(
    channel: TheRockChannel,
    family: &str,
    resolved_version: &str,
    relation: &str,
) -> Result<bool> {
    println!("sdk install: an existing ROCm SDK is the active default runtime");
    println!("  active default runtime: {relation}");
    println!(
        "  replacing with: ROCm {} for family {family} ({} channel)",
        runtime_version_display(resolved_version),
        channel.as_str()
    );
    println!(
        "  effect: this install becomes the active default runtime, overriding the current default"
    );
    // The host-newer ROCm explanation is already emitted as a visible progress
    // line before this prompt on the real install path, so it is not repeated
    // here.
    prompt_yes_no("Replace the existing ROCm SDK as the active default? [y/N]: ")
}

fn prompt_yes_no(prompt: &str) -> Result<bool> {
    rocm_core::confirm::print_prompt(&mut std::io::stdout(), prompt)
        .context("failed to write confirmation prompt")?;
    rocm_core::confirm::read_answer(&mut std::io::stdin().lock())
        .context("failed to read confirmation response")
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        test_paths, test_runtime_manifest, write_test_runtime_manifest,
    };
    use super::*;
    use crate::runtime_registry::runtime_manifest_path;
    use std::fs;

    /// Register `manifest` and make it the active default runtime, the way a
    /// completed `install sdk` does.
    fn write_active_test_runtime(
        paths: &AppPaths,
        manifest: &InstalledRuntimeManifest,
    ) -> Result<()> {
        write_test_runtime_manifest(paths, manifest)?;
        let mut config = RocmCliConfig::load(paths)?;
        config.default_runtime_id = Some(manifest.runtime_id.clone());
        config.active_runtime_key = Some(manifest.runtime_key.clone());
        config.save(paths)?;
        Ok(())
    }

    #[test]
    fn active_default_relation_classifies_upgrade_downgrade_reinstall() -> Result<()> {
        let (root, paths) = test_paths("active-default-relation");
        let mut manifest = test_runtime_manifest(
            "release-wheel-gfx120X-all",
            "therock-release:gfx120X-all",
            10,
        );
        manifest.version = "7.13.0".to_owned();
        write_active_test_runtime(&paths, &manifest)?;

        let upgrade = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("relation should be reported while a runtime is the active default");
        assert!(upgrade.starts_with("upgrade from"), "got: {upgrade}");
        assert!(upgrade.contains("7.13.0"));

        let downgrade = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.12.0",
        )?
        .expect("relation should be reported while a runtime is the active default");
        assert!(downgrade.starts_with("downgrade from"), "got: {downgrade}");

        let reinstall = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.13.0",
        )?
        .expect("relation should be reported while a runtime is the active default");
        assert!(reinstall.starts_with("reinstall from"), "got: {reinstall}");

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_gates_a_different_family_or_channel() -> Result<()> {
        // The regression this gate exists for. `finalize_successful_sdk_install`
        // activates whatever was installed last regardless of family or channel,
        // so installing gfx120X-all while a gfx110X-all runtime is the active
        // default displaces it. A family/channel-scoped gate reported "no
        // existing SDK" here and let the displacement through unconfirmed.
        let (root, paths) = test_paths("active-default-relation-cross-family");
        let mut manifest = test_runtime_manifest(
            "release-wheel-gfx110X-all",
            "therock-release:gfx110X-all",
            10,
        );
        manifest.version = "7.13.0".to_owned();
        write_active_test_runtime(&paths, &manifest)?;

        let other_family = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("installing another family must still report the active default it displaces");
        assert!(
            other_family.contains("replaces active default")
                && other_family.contains("gfx110X-all"),
            "got: {other_family}"
        );

        let other_channel = active_default_runtime_relation(
            &paths,
            TheRockChannel::Nightly,
            "gfx110X-all",
            "7.14.0",
        )?
        .expect("installing another channel must still report the active default it displaces");
        assert!(
            other_channel.contains("replaces active default")
                && other_channel.contains("release channel"),
            "got: {other_channel}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_none_without_an_active_default() -> Result<()> {
        // Nothing installed at all, and — the second case — a registered runtime
        // that no config points at. Neither displaces anything, so neither may
        // prompt: an install with no active default runtime is the fresh path.
        let (root, paths) = test_paths("active-default-relation-empty");
        assert!(
            active_default_runtime_relation(
                &paths,
                TheRockChannel::Release,
                "gfx120X-all",
                "7.14.0"
            )?
            .is_none()
        );

        let manifest = test_runtime_manifest(
            "release-wheel-gfx120X-all",
            "therock-release:gfx120X-all",
            10,
        );
        write_test_runtime_manifest(&paths, &manifest)?;
        assert!(
            active_default_runtime_relation(
                &paths,
                TheRockChannel::Release,
                "gfx120X-all",
                "7.14.0"
            )?
            .is_none()
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_propagates_manifest_read_errors() -> Result<()> {
        // A manifest entry that cannot be read (here, a directory sitting where a
        // `*.json` manifest file is expected) must surface as an error, not be
        // silently treated as "no active default" — that would skip the
        // confirmation gate exactly when we're least sure what is active.
        let (root, paths) = test_paths("active-default-relation-error");
        let registry_dir = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(registry_dir.join("broken.json"))?;

        let error = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )
        .expect_err("a manifest read failure should be propagated, not swallowed");
        assert!(!error.to_string().is_empty());

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_fails_closed_on_an_unparsable_active_manifest() -> Result<()> {
        // The other half of the same policy, and the half that used to fail open:
        // the manifest file *reads* fine, so nothing errors, but it does not
        // deserialize. `load_runtime_manifests` dropped it silently,
        // `current_runtime_manifest` then missed, and the gate was skipped with a
        // fresh-install verdict while `rocm runtimes list` still reported the
        // runtime as active. This is the older-manifest/newer-binary shape:
        // `family_source` carries no `#[serde(default)]`.
        let (root, paths) = test_paths("active-default-relation-unparsable");
        let manifest = test_runtime_manifest(
            "release-wheel-gfx120X-all",
            "therock-release:gfx120X-all",
            10,
        );
        write_active_test_runtime(&paths, &manifest)?;

        // The helper carries the preconditions that make this the parse path and
        // not the I/O path: the file still reads, and it no longer deserializes.
        let _ = make_test_runtime_manifest_unparsable(&paths, &manifest.runtime_key)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("an unparsable active manifest must not yield a fresh-install verdict");
        assert!(
            relation.contains(&manifest.runtime_key),
            "the relation must name the runtime the config still calls active: {relation}"
        );
        assert!(
            relation.contains("could not be read"),
            "the relation must say why the active default is unknown: {relation}"
        );

        // Fail closed means the consent gate engages, not that the install is
        // blocked outright: a consent flag still gets an operator through.
        assert_eq!(
            sdk_install_approval(true, SdkInstallConsent::Ask, false),
            SdkInstallApproval::RefuseNonInteractive
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_fails_closed_on_a_dangling_active_runtime_key() -> Result<()> {
        // Same policy, third shape: nothing is unreadable or unparsable, the
        // config simply names an active runtime the registry has no manifest for.
        // `rocm runtimes list` reports this as
        // `active_status: missing manifest for active_runtime_key=...`, so a
        // fresh-install verdict here would have one CLI assert both that a
        // runtime is active and that none is.
        let (root, paths) = test_paths("active-default-relation-dangling");
        let mut config = RocmCliConfig::load(&paths)?;
        config.active_runtime_key = Some("release-wheel-gfx120X-all".to_owned());
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("a dangling active_runtime_key must not yield a fresh-install verdict");
        assert!(
            relation.contains("release-wheel-gfx120X-all"),
            "the relation must name the unresolved key: {relation}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_blames_an_unparsable_manifest_only_when_it_is_the_active_one()
    -> Result<()> {
        // A dangling `active_runtime_key` and an unparsable manifest belonging to
        // some *other* runtime are two independent faults that show up together
        // routinely: an older binary's manifest fails `from_slice` against this
        // one while the runtime the config calls active was removed outright.
        // Naming the stranger's file as "its manifest" would send the operator to
        // repair a path that has nothing to do with the problem and never mention
        // the runtime that actually went missing.
        let (root, paths) = test_paths("active-default-relation-unrelated-unparsable");

        let stranger = test_runtime_manifest(
            "release-wheel-gfx110X-all",
            "therock-release:gfx110X-all",
            10,
        );
        write_test_runtime_manifest(&paths, &stranger)?;
        let stranger_path = make_test_runtime_manifest_unparsable(&paths, &stranger.runtime_key)?;

        let mut config = RocmCliConfig::load(&paths)?;
        // Nothing on disk is stored under this key, so the only unreadable entry
        // in the registry is the stranger's.
        config.active_runtime_key = Some("release-wheel-gfx120X-all".to_owned());
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("a dangling active_runtime_key must not yield a fresh-install verdict");
        assert!(
            relation.contains(
                "recorded as `release-wheel-gfx120X-all`, but no installed runtime manifest matches it"
            ),
            "the missing runtime is the cause, not the stranger's manifest: {relation}"
        );
        assert!(
            !relation.contains("its manifest could not be read"),
            "an unrelated unparsable manifest must not be blamed as the active one's: {relation}"
        );
        assert!(
            relation.contains(&format!(
                "; unreadable runtime manifests: {}",
                stranger_path.display()
            )),
            "the unrelated unparsable manifest is still reported, as a suffix: {relation}"
        );

        // The other direction, in the same registry: once the active key's *own*
        // manifest is unparsable, "could not be read" is the right cause even
        // though the stranger's file is unreadable too.
        let active = test_runtime_manifest(
            "release-wheel-gfx120X-all",
            "therock-release:gfx120X-all",
            20,
        );
        write_test_runtime_manifest(&paths, &active)?;
        let active_path = make_test_runtime_manifest_unparsable(&paths, &active.runtime_key)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("an unparsable active manifest must not yield a fresh-install verdict");
        assert!(
            relation.contains(
                "recorded as `release-wheel-gfx120X-all`, but its manifest could not be read"
            ),
            "the active key's own unparsable manifest is the cause here: {relation}"
        );
        assert!(
            relation.contains(&active_path.display().to_string())
                && relation.contains(&stranger_path.display().to_string()),
            "both unreadable entries are still listed: {relation}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_names_both_unresolved_config_pointers() -> Result<()> {
        // `current_runtime_manifest` tries `active_runtime_key` and falls through
        // to `default_runtime_id`, so arriving at the fail-closed path with both
        // set means both failed. `rocm runtimes activate` writes the pair
        // together, so removing that runtime while another version of the family
        // remains strands them together too. Reporting only the key would have
        // the operator repair half the config and hit the gate again.
        let (root, paths) = test_paths("active-default-relation-both-pointers");
        let runtime_id = "therock-release:gfx120X-all";
        let mut older = test_runtime_manifest("release-wheel-gfx120X-all-7130", runtime_id, 10);
        older.version = "7.13.0".to_owned();
        let mut newer = test_runtime_manifest("release-wheel-gfx120X-all-7140", runtime_id, 20);
        newer.version = "7.14.0".to_owned();
        write_test_runtime_manifest(&paths, &older)?;
        write_test_runtime_manifest(&paths, &newer)?;

        let mut config = RocmCliConfig::load(&paths)?;
        config.active_runtime_key = Some("release-wheel-gfx120X-all-7120".to_owned());
        config.default_runtime_id = Some(runtime_id.to_owned());
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.15.0",
        )?
        .expect("two unresolved pointers must not yield a fresh-install verdict");
        assert!(
            relation.contains(
                "recorded as `release-wheel-gfx120X-all-7120`, but no installed runtime manifest matches it"
            ),
            "the relation must name the unresolved key: {relation}"
        );
        assert!(
            relation.contains(&format!(
                "; the recorded default runtime_id `{runtime_id}` does not settle it either, because 2 installed runtime manifests match it"
            )),
            "the relation must also name the ambiguous fallback id: {relation}"
        );

        // The zero-match half of the same pairing: the fallback is dangling
        // rather than ambiguous, and must still be named.
        let mut config = RocmCliConfig::load(&paths)?;
        config.default_runtime_id = Some("therock-release:gfx110X-all".to_owned());
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.15.0",
        )?
        .expect("two unresolved pointers must not yield a fresh-install verdict");
        assert!(
            relation.contains(
                "; the recorded default runtime_id `therock-release:gfx110X-all` does not settle it either, because no installed runtime manifest matches it"
            ),
            "the relation must also name the dangling fallback id: {relation}"
        );

        // Fail closed means the consent gate engages, not that the install is
        // blocked outright: a consent flag still gets an operator through.
        assert_eq!(
            sdk_install_approval(true, SdkInstallConsent::Ask, false),
            SdkInstallApproval::RefuseNonInteractive
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_fails_closed_on_an_ambiguous_default_runtime_id() -> Result<()> {
        // Same policy, but the *other* resolution path. `current_runtime_manifest`
        // falls back to `default_runtime_id` when `active_runtime_key` is unset,
        // and resolves only an exactly-one match. `runtime_id` is
        // `therock-<channel>:<family>` with no version in it, so two installed
        // versions of one family share it and the fallback returns `None` — while
        // `rocm runtimes list` reports `active_status: ambiguous runtime_id=...`.
        // `rocm config set-default-runtime` reaches this state directly: it stores
        // the id unvalidated and clears `active_runtime_key`.
        let (root, paths) = test_paths("active-default-relation-ambiguous-id");
        let runtime_id = "therock-release:gfx120X-all";
        let mut older = test_runtime_manifest("release-wheel-gfx120X-all-7130", runtime_id, 10);
        older.version = "7.13.0".to_owned();
        let mut newer = test_runtime_manifest("release-wheel-gfx120X-all-7140", runtime_id, 20);
        newer.version = "7.14.0".to_owned();
        write_test_runtime_manifest(&paths, &older)?;
        write_test_runtime_manifest(&paths, &newer)?;

        let mut config = RocmCliConfig::load(&paths)?;
        config.default_runtime_id = Some(runtime_id.to_owned());
        // Precondition: this is the `default_runtime_id` shape, not the
        // `active_runtime_key` shape the sibling tests already cover.
        config.active_runtime_key = None;
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.15.0",
        )?
        .expect("an ambiguous default_runtime_id must not yield a fresh-install verdict");
        assert!(
            relation.contains(runtime_id),
            "the relation must name the id that could not be resolved: {relation}"
        );
        assert!(
            relation.contains("2 installed runtime manifests match"),
            "the relation must say the id is ambiguous and how badly: {relation}"
        );

        // Fail closed means the consent gate engages, not that the install is
        // blocked outright: a consent flag still gets an operator through.
        assert_eq!(
            sdk_install_approval(true, SdkInstallConsent::Ask, false),
            SdkInstallApproval::RefuseNonInteractive
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_fails_closed_on_a_dangling_default_runtime_id() -> Result<()> {
        // The zero-match half of the same fallback path: `_ => None` in
        // `current_runtime_manifest` swallows "no match" exactly as it swallows
        // "many matches". `rocm config set-default-runtime` does not check the id
        // against the registry, so a typo — or uninstalling the last runtime of a
        // family — leaves the config asserting an active default that is not
        // there, which `rocm runtimes list` reports as
        // `active_status: missing manifest for active_runtime_id=...`.
        let (root, paths) = test_paths("active-default-relation-dangling-id");
        let other = test_runtime_manifest(
            "release-wheel-gfx110X-all",
            "therock-release:gfx110X-all",
            10,
        );
        write_test_runtime_manifest(&paths, &other)?;

        let mut config = RocmCliConfig::load(&paths)?;
        config.default_runtime_id = Some("therock-release:gfx120X-all".to_owned());
        config.active_runtime_key = None;
        config.save(&paths)?;

        let relation = active_default_runtime_relation(
            &paths,
            TheRockChannel::Release,
            "gfx120X-all",
            "7.14.0",
        )?
        .expect("a dangling default_runtime_id must not yield a fresh-install verdict");
        assert!(
            relation.contains("therock-release:gfx120X-all"),
            "the relation must name the id that could not be resolved: {relation}"
        );
        assert!(
            relation.contains("no installed runtime manifest matches it"),
            "the relation must say the id resolved to nothing: {relation}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn active_default_relation_is_fresh_when_no_config_pointer_claims_an_active_default()
    -> Result<()> {
        // The limit of the fail-closed policy. An unparsable registry manifest is
        // only evidence of a *displacement* risk if something claims an active
        // default; with neither `active_runtime_key` nor `default_runtime_id` set,
        // nothing does, and demanding a consent flag would block a genuinely fresh
        // install over an unrelated registry wart.
        let (root, paths) = test_paths("active-default-relation-fresh-unparsable");
        let manifest = test_runtime_manifest(
            "release-wheel-gfx120X-all",
            "therock-release:gfx120X-all",
            10,
        );
        write_test_runtime_manifest(&paths, &manifest)?;
        // The helper carries the preconditions that make this the parse path and
        // not the I/O path: the file still reads, and it no longer deserializes.
        let _ = make_test_runtime_manifest_unparsable(&paths, &manifest.runtime_key)?;

        let config = RocmCliConfig::load(&paths)?;
        assert!(config.active_runtime_key.is_none());
        assert!(config.default_runtime_id.is_none());

        assert_eq!(
            active_default_runtime_relation(
                &paths,
                TheRockChannel::Release,
                "gfx120X-all",
                "7.14.0",
            )?,
            None,
            "no config pointer claims an active default, so this is a fresh install"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn sdk_install_approval_only_prompts_when_an_active_default_is_displaced() {
        let assume_yes = SdkInstallConsent::Preapproved(SdkInstallApprovalSource::AssumeYes);
        let update_apply = SdkInstallConsent::Preapproved(SdkInstallApprovalSource::UpdateApply {
            activates: false,
        });

        // No active default runtime -> never prompt, regardless of terminal/consent.
        assert_eq!(
            sdk_install_approval(false, SdkInstallConsent::Ask, false),
            SdkInstallApproval::ProceedFresh
        );
        assert_eq!(
            sdk_install_approval(false, SdkInstallConsent::Ask, true),
            SdkInstallApproval::ProceedFresh
        );
        assert_eq!(
            sdk_install_approval(false, assume_yes, false),
            SdkInstallApproval::ProceedFresh
        );

        // Active default present: preapproved consent proceeds and is credited to
        // its real source; a terminal prompts; neither refuses.
        assert_eq!(
            sdk_install_approval(true, assume_yes, false),
            SdkInstallApproval::ProceedApproved(SdkInstallApprovalSource::AssumeYes)
        );
        assert_eq!(
            sdk_install_approval(
                true,
                SdkInstallConsent::Preapproved(
                    SdkInstallApprovalSource::ApproveReplacingActiveDefault
                ),
                false
            ),
            SdkInstallApproval::ProceedApproved(
                SdkInstallApprovalSource::ApproveReplacingActiveDefault
            )
        );
        assert_eq!(
            sdk_install_approval(true, update_apply, false),
            SdkInstallApproval::ProceedApproved(SdkInstallApprovalSource::UpdateApply {
                activates: false
            })
        );
        assert_eq!(
            sdk_install_approval(true, SdkInstallConsent::Ask, true),
            SdkInstallApproval::PromptOverwrite
        );
        assert_eq!(
            sdk_install_approval(true, SdkInstallConsent::Ask, false),
            SdkInstallApproval::RefuseNonInteractive
        );
    }

    #[test]
    fn preapproved_install_line_credits_the_real_consent_source() {
        // `rocm update --apply` draws its approval from the runtime the user
        // selected, not from a flag — its `--yes` is inert — so a line
        // crediting `--yes` names an approval that was never given. And without
        // `--activate`, `apply_runtime_update` leaves the active default alone,
        // so claiming the install "becomes the active default runtime" is false.
        let by_yes = preapproved_install_line(
            SdkInstallApprovalSource::AssumeYes,
            "upgrade from installed 7.13.0 (release-wheel-gfx120X-all)",
            "7.14.0",
        );
        assert!(by_yes.starts_with("Approved by --yes:"), "got: {by_yes}");
        assert!(by_yes.contains("becomes the active default runtime"));

        // The narrow flag is not `--yes`: ROCm CLI's own terminal-less surfaces
        // pass only this one, and a line crediting `--yes` would tell whoever
        // reads the chat or dashboard transcript that consent to run `sudo` was
        // given when it never was.
        let by_narrow = preapproved_install_line(
            SdkInstallApprovalSource::ApproveReplacingActiveDefault,
            "upgrade from installed 7.13.0 (release-wheel-gfx120X-all)",
            "7.14.0",
        );
        assert!(
            by_narrow.starts_with("Approved by --approve-replacing-active-default:"),
            "got: {by_narrow}"
        );
        assert!(
            !by_narrow.contains("--yes"),
            "the narrow flag must not be credited to --yes: {by_narrow}"
        );
        assert!(by_narrow.contains("becomes the active default runtime"));

        let update_activates = preapproved_install_line(
            SdkInstallApprovalSource::UpdateApply { activates: true },
            "upgrade from installed 7.13.0 (release-wheel-gfx120X-all)",
            "7.14.0",
        );
        assert!(
            !update_activates.contains("--yes"),
            "the update path must not credit an approval `--yes` did not grant: {update_activates}"
        );
        assert!(update_activates.contains("becomes the active default runtime"));

        let update_only = preapproved_install_line(
            SdkInstallApprovalSource::UpdateApply { activates: false },
            "upgrade from installed 7.13.0 (release-wheel-gfx120X-all)",
            "7.14.0",
        );
        assert!(
            !update_only.contains("--yes"),
            "the update path must not credit an approval `--yes` did not grant: {update_only}"
        );
        assert!(
            !update_only.contains("becomes the active default runtime"),
            "`update --apply` without --activate does not change the active default: {update_only}"
        );
        assert!(update_only.contains("The active default runtime is unchanged"));
    }

    #[test]
    fn refusal_recommends_the_narrow_flag_before_yes() {
        // This is the only message a script or CI job reads when it hits the
        // gate, and it is the audience the narrow flag exists for. Recommending
        // `--yes` first would hand an unattended caller the second consent that
        // flag carries — approval to install system packages with `sudo` — and
        // park it on a password prompt it has no terminal to answer.
        let message = refuse_non_interactive_message(
            "upgrade from installed 7.13.0 (release-wheel-gfx120X-all)",
        );

        let narrow = message
            .find("--approve-replacing-active-default")
            .unwrap_or_else(|| panic!("the refusal must name the narrow flag: {message}"));
        // `--yes` still has to appear: a user at a terminal who wants both
        // consents should not have to go looking for it.
        let yes = message
            .find("--yes")
            .unwrap_or_else(|| panic!("the refusal must still explain --yes: {message}"));
        assert!(
            narrow < yes,
            "the narrow flag must be recommended before --yes: {message}"
        );
        assert!(
            message.contains("system packages"),
            "the refusal must say what --yes additionally approves: {message}"
        );
        assert!(
            message.contains("is the active default runtime"),
            "the refusal must name what would be replaced: {message}"
        );
        assert!(
            !message.to_lowercase().contains("overwrit"),
            "an upgrade leaves the previous install on disk; it replaces the \
             active default rather than overwriting it: {message}"
        );
    }

    #[test]
    fn fresh_install_line_claims_no_absent_sdk() {
        // The fresh path is reached whenever no runtime is the active default,
        // which does not mean no SDK is installed anywhere. Saying "no existing
        // ROCm SDK found" there would be false on a host holding a registered
        // but unactivated runtime.
        let line = fresh_install_line("7.14.0", "gfx120X-all");
        assert!(
            !line.to_lowercase().contains("no existing rocm sdk"),
            "got: {line}"
        );
        assert!(line.contains("No active ROCm SDK runtime is configured"));
        assert!(line.contains("gfx120X-all"));
    }

    /// Rewrite an already-written registry manifest so it still *reads* but no
    /// longer deserializes, and return its path. Drops `family_source`, which
    /// carries no `#[serde(default)]` — the real older-binary/newer-binary shape,
    /// not an invented corruption.
    fn make_test_runtime_manifest_unparsable(
        paths: &AppPaths,
        runtime_key: &str,
    ) -> Result<PathBuf> {
        let path = runtime_manifest_path(paths, runtime_key);
        let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        value
            .as_object_mut()
            .expect("manifest is a JSON object")
            .remove("family_source")
            .expect("manifest carries family_source");
        fs::write(&path, serde_json::to_vec_pretty(&value)?)?;
        assert!(
            fs::read(&path).is_ok(),
            "the fixture must still read, or this is the I/O path, not the parse path"
        );
        assert!(
            serde_json::from_slice::<InstalledRuntimeManifest>(&fs::read(&path)?).is_err(),
            "the test fixture must be unparsable, or this asserts nothing"
        );
        Ok(path)
    }
}
