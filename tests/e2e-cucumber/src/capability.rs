// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Host capability probe for the E2E suite.
//!
//! The suite is black-box (it never imports rocm-cli crates), but its
//! per-scenario expectations must still follow the product's real behaviour —
//! chiefly *which serve engine the CLI would pick on this host*. We learn that
//! by spawning the real binary (`rocm examine` + `rocm engines list`) once at
//! startup and caching the result.
//!
//! IMPORTANT — the effective serve engine is RE-IMPLEMENTED here (see
//! [`effective_serve_engine`]), duplicating the product's `select_serve_engine` /
//! `preferred_serve_engine_for_therock_family` logic. It will drift if the product
//! changes engine support, so the unit tests below guard it.
//!
//! `examine`'s `default_engine` now reports the host's real engine rather than the
//! old hardcoded `"lemonade"` constant, so this could in principle be replaced by
//! reading the product's own answer. Deliberately KEEP the re-implementation: the
//! `examine-reports-host-default-engine` scenario asserts the product's value
//! against this one, and that check is only meaningful while the two are derived
//! independently. Reading the product value here would turn it into a tautology.

use std::sync::OnceLock;

/// Name of the `rocm` binary under test (mirrors the test binary's
/// `rocm_binary()`): `ROCM_CLI_BINARY` or plain `rocm`.
fn rocm_binary() -> String {
    std::env::var("ROCM_CLI_BINARY").unwrap_or_else(|_| "rocm".to_string())
}

/// What `rocm serve <model>` would pick with no `--engine`, from GPU family + OS.
///
/// Mirrors the product precedence in `preferred_serve_engine_for_host_gpu_summary`
/// (rocm-core): vLLM on data-center families (`*-dcgpu`) and gfx906/908/90a,
/// never on native Windows; otherwise the platform default, lemonade.
///
/// This is the single re-implemented rule (decision #1). When the product grows
/// an `effective_serve_engine` probe field, replace the callers with the parsed
/// field and keep this only as the drift-check reference.
pub fn effective_serve_engine(gfx_target: Option<&str>, os_family: &str) -> String {
    // The vLLM adapter bails on native Windows (WSL builds as Linux, so it
    // reports os_family "linux" and stays eligible).
    if os_family.eq_ignore_ascii_case("windows") {
        return "lemonade".to_owned();
    }
    if family_prefers_vllm(gfx_target) {
        "vllm".to_owned()
    } else {
        "lemonade".to_owned()
    }
}

/// True when a gfx target's TheRock family is vLLM-preferred: any `*-dcgpu`
/// data-center family, or the explicit gfx906/908/90a set. Mirrors
/// `preferred_serve_engine_for_therock_family` + `normalize_therock_family`.
fn family_prefers_vllm(gfx_target: Option<&str>) -> bool {
    let Some(raw) = gfx_target else {
        return false;
    };
    let family = normalize_family(raw);
    family.ends_with("-dcgpu") || matches!(family.as_str(), "gfx906" | "gfx908" | "gfx90a")
}

/// Coarse gfx-target → TheRock family normalization, matching the subset of
/// `rocm_core::normalize_therock_family` that affects engine preference. We only
/// need enough fidelity to decide vLLM-preference: the data-center families
/// (gfx94x/gfx950 → `*-dcgpu`) and the gfx906/908/90a set. Everything else
/// (e.g. gfx1151 Strix) falls through as itself → not vLLM-preferred.
fn normalize_family(raw: &str) -> String {
    let v = raw.trim().to_ascii_lowercase();
    if v.ends_with("-dcgpu") {
        return v;
    }
    if v.starts_with("gfx90a") {
        return "gfx90a".to_owned();
    }
    if v.starts_with("gfx906") {
        return "gfx906".to_owned();
    }
    if v.starts_with("gfx908") {
        return "gfx908".to_owned();
    }
    // MI300-class (gfx942/gfx94x) and gfx950 are data-center parts.
    if v.starts_with("gfx94") {
        return "gfx94X-dcgpu".to_owned();
    }
    if v.starts_with("gfx950") {
        return "gfx950-dcgpu".to_owned();
    }
    v
}

/// The probed host capability, learned once from the real `rocm` binary.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HostCapability {
    /// `examine`'s `os:` line, lowercased (e.g. "linux", "windows", "macos").
    pub os_family: String,
    /// `examine`'s `wsl:` line.
    pub is_wsl: bool,
    /// First AMD GPU's gfx target from `examine`'s `detected_gfx_target:` line
    /// (e.g. "gfx942", "gfx1151"), if a real one was reported.
    pub gfx_target: Option<String>,
    /// Whether an AMD GPU is present AND usable here, so `@requires-gpu`
    /// scenarios can run. A real `detected_gfx_target`, plus a ready ROCm
    /// driver path on WSL — see [`host_has_usable_gpu`].
    pub has_amd_gpu: bool,
    /// Engine adapters the binary reports as present. Both builtins are always
    /// "built-in", so this is NOT the same as "can start here" — use
    /// [`HostCapability::engine_available`] for that.
    pub available_engines: Vec<String>,
    /// What `serve` picks with no `--engine` on this host (re-implemented rule).
    pub effective_serve_engine: String,
    /// Stable platform identity derived from hardware, not from an artifact name:
    /// "mock" (no AMD GPU), else the family/target (e.g. "mi300x", "strix-halo").
    pub platform_slug: String,
}

impl HostCapability {
    /// Whether a given engine can actually START on this host. Distinct from
    /// "adapter present": vLLM's adapter is built-in everywhere but cannot run on
    /// native Windows or on non-vLLM-preferred families; lemonade runs anywhere.
    pub fn engine_available(&self, engine: &str) -> bool {
        match engine {
            "lemonade" => true,
            "vllm" => {
                !self.os_family.eq_ignore_ascii_case("windows")
                    && family_prefers_vllm(self.gfx_target.as_deref())
            }
            _ => self.available_engines.iter().any(|e| e == engine),
        }
    }
}

/// Component versions on this platform (for the report heading).
///
/// For the consolidated report's per-column heading. All fields are best-effort:
/// `None` when the source isn't present (e.g. an engine that was never installed
/// on this platform). Collected from the harness only — no product command
/// exposes all of these, so we read the OS from `examine`, ROCm from the active
/// managed runtime, and the engine versions from the installed runtime tree (see
/// [`collect_versions`]).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PlatformVersions {
    /// OS distro string, e.g. "Ubuntu 24.04.3 LTS" (`examine`'s `distro:` line).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    /// Active managed ROCm/TheRock runtime version, e.g. "7.13.0"
    /// (`runtimes list`'s `version=` on the active runtime).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rocm: Option<String>,
    /// Installed vLLM version, e.g. "0.23.0+rocm723" (parsed from the
    /// `vllm-<ver>.dist-info` dir in the active runtime venv).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vllm: Option<String>,
    /// Installed lemonade server version, e.g. "10.6.0" (`lemond --version`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lemonade: Option<String>,
}

/// Collect component versions for the report from the installed managed runtime.
///
/// Reads the runtime whose registry lives at `runtimes_dir` (the shared prewarm
/// tree in CI, or a scenario's `data/runtimes` locally). Best-effort: any source
/// that isn't present yields `None`. `os` re-reads `examine`; `rocm`/`vllm`/
/// `lemonade` come from the active runtime, so they're populated only once the SDK
/// / an engine has been installed on this platform.
#[must_use]
pub fn collect_versions(runtimes_dir: Option<&std::path::Path>) -> PlatformVersions {
    let mut v = PlatformVersions::default();

    // OS distro from `examine` — ALWAYS collected (every platform, incl. mock and
    // hosts without a shared runtime dir). Isolated throwaway root reads the host.
    if let Ok(tmp) = tempfile::TempDir::with_prefix("rocm-e2e-ver-") {
        let examine = run_probe(tmp.path(), &["examine"]);
        v.os = examine
            .lines()
            .find_map(|l| l.trim().strip_prefix("distro:"))
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty());
    }

    // ROCm/vLLM/lemonade versions come from the installed managed runtime. Only
    // available when CI provides a persistent runtimes dir (E2E_SHARED_RUNTIMES_DIR)
    // — mock has no runtime, and per-scenario isolated installs are gone by now.
    if let Some((rocm, root)) = runtimes_dir.and_then(active_runtime_install_root) {
        v.rocm = Some(rocm);
        // vLLM: parse the version out of `.../site-packages/vllm-<ver>.dist-info`.
        v.vllm = vllm_version_from_venv(&root);
        // lemonade: `<root>/engines/lemonade/runtime/lemond --version`.
        v.lemonade = lemonade_version(&root);
    }

    v
}

/// Read the active managed runtime's `(version, install_root)` from the runtimes
/// registry. Returns `None` when the tree names no single runtime.
///
/// Which runtime that is comes from [`crate::shared_runtime::runtime_key_to_activate`],
/// the same answer the scenarios activate — so the version this report attributes
/// a run to is the version the run actually served on. This used to fall back to
/// the first `read_dir` entry when `active.json` named nothing, which was a
/// coin flip as soon as the pre-warm started keeping a newer runtime alongside
/// the old one: the report could name one ROCm version while the serve used
/// another. Reporting no version is the better failure — an absent field reads
/// as unknown, a wrong one reads as fact.
///
/// The install_root is resolved from `runtimes_dir` (the shared tree we were
/// handed) as `<runtimes_dir>/wheel/<runtime_key>`, NOT from the manifest's own
/// `install_root` field, which records where the runtime was first installed and
/// need not be where it lives now. On MI300X the two coincide (the pre-warm
/// installs in place); on Strix they did not, and trusting the field made
/// `vllm`/`lemonade` probe a dead path and report no versions at all.
///
/// The manifest fallback is load-bearing, not legacy — do not read it as dead
/// code. `MANAGED_RUNTIME_FORMATS` is `["wheel", "tarball"]`, and a tarball
/// runtime lives at `<tree>/tarball/<key>`, which the derived `wheel` path never
/// matches. For those the fallback is the only correct answer. It also still
/// covers a tree predating the `wheel/` layout.
///
/// One shape can no longer reach here: a runtime whose recorded root left the
/// tree entirely. `runtime_key_to_activate` now filters those out, so the
/// fallback yields an in-tree path or nothing. Where every entry is such a
/// corpse this returns `None` and the report simply omits the version — absent
/// reads as unknown, which is the honest outcome; a wrong version reads as fact.
fn active_runtime_install_root(
    runtimes_dir: &std::path::Path,
) -> Option<(String, std::path::PathBuf)> {
    let key = crate::shared_runtime::runtime_key_to_activate(runtimes_dir)?;
    let manifest = runtimes_dir.join("registry").join(format!("{key}.json"));
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(manifest).ok()?).ok()?;
    let version = json.get("version")?.as_str()?.to_owned();
    // Resolve the root inside the shared tree first; fall back to the manifest's
    // recorded install_root only if that derived path doesn't exist.
    let derived = runtimes_dir.join("wheel").join(&key);
    let root = if derived.is_dir() {
        derived
    } else {
        std::path::PathBuf::from(json.get("install_root")?.as_str()?)
    };
    Some((version, root))
}

/// Select the canonical aggregate wheel runtime from `rocm runtimes list` output.
///
/// The list is newest-first, so the first matching key is the runtime installed
/// by the pre-warm refresh when legacy family-keyed entries coexist with it.
pub fn canonical_wheel_runtime_key(inventory: &str) -> Option<&str> {
    inventory.lines().find_map(|line| {
        line.split_whitespace()
            .find(|field| field.contains("-wheel-multi-arch-"))
    })
}

/// Parse the vLLM version from the `vllm-<ver>.dist-info` directory in the
/// runtime venv's site-packages (works without importing vllm).
///
/// The venv layout differs by OS: Linux/macOS put site-packages under
/// `lib/python3.X/site-packages` (the minor version varies), Windows under
/// `Lib/site-packages`. Locate it by probing both rather than hardcoding one, so
/// vLLM is found on every platform where it's installed.
fn vllm_version_from_venv(install_root: &std::path::Path) -> Option<String> {
    site_packages_dirs(install_root)
        .into_iter()
        .find_map(|site| {
            std::fs::read_dir(site).ok()?.find_map(|e| {
                let name = e.ok()?.file_name().into_string().ok()?;
                // "vllm-0.23.0+rocm723.dist-info" -> "0.23.0+rocm723"
                name.strip_prefix("vllm-")?
                    .strip_suffix(".dist-info")
                    .map(str::to_owned)
            })
        })
}

/// Candidate `site-packages` directories for a runtime venv, across OS layouts.
/// Windows: `Lib/site-packages`. Unix: `lib/python3.X/site-packages` for
/// whatever python minor version the runtime shipped (globbed, not hardcoded).
fn site_packages_dirs(install_root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    // Windows layout.
    dirs.push(install_root.join("Lib").join("site-packages"));
    // Unix layout: enumerate lib/python3.* so the minor version isn't pinned.
    if let Ok(entries) = std::fs::read_dir(install_root.join("lib")) {
        for e in entries.flatten() {
            let p = e.path();
            if p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("python3"))
            {
                dirs.push(p.join("site-packages"));
            }
        }
    }
    dirs
}

/// Ask the embedded lemonade server for its version (`lemond --version` →
/// "lemond version 10.6.0"). `None` if lemonade isn't installed in this runtime.
///
/// The binary is `lemond` on Unix and `lemond.exe` on Windows — try both.
fn lemonade_version(install_root: &std::path::Path) -> Option<String> {
    let runtime = install_root.join("engines/lemonade/runtime");
    let lemond = [runtime.join("lemond"), runtime.join("lemond.exe")]
        .into_iter()
        .find(|p| p.is_file())?;
    let out = std::process::Command::new(&lemond)
        .arg("--version")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    // "lemond version 10.6.0" -> "10.6.0"; fall back to the last whitespace token.
    text.split_whitespace().last().map(str::to_owned)
}

/// Cached, probed once per process.
pub fn host_capability() -> &'static HostCapability {
    static CAP: OnceLock<HostCapability> = OnceLock::new();
    CAP.get_or_init(probe_host_capability)
}

/// Spawn the real binary in an isolated env and build a [`HostCapability`].
/// Deliberately does NOT reuse a scenario's isolated root — the probe runs
/// before any scenario, in its own throwaway temp dir.
fn probe_host_capability() -> HostCapability {
    let tmp =
        tempfile::TempDir::with_prefix("rocm-e2e-probe-").expect("failed to create probe temp dir");
    let root = tmp.path();

    // Parse the HUMAN `examine` text, not `--json`: the two disagree on GPU
    // detection on the self-hosted runners (the JSON `Examination` reported
    // has_amd_gpu:false / no gfx on a real MI300X, while the human text — the
    // signal the scenarios themselves trust via `detected_gfx_target:` — reports
    // it correctly). Keying on the human text keeps the probe consistent with
    // what the scenarios see.
    let examine = run_probe(root, &["examine"]);
    let engines = run_probe(root, &["engines", "list"]);

    let ExamineFacts {
        os_family,
        is_wsl,
        gfx_target,
        driver_status,
    } = parse_examine_text(&examine);
    let has_amd_gpu = host_has_usable_gpu(gfx_target.as_deref(), is_wsl, &driver_status);
    let available_engines = parse_engines_list(&engines);
    let effective_serve_engine = effective_serve_engine(gfx_target.as_deref(), &os_family);
    let platform_slug =
        derive_platform_slug(has_amd_gpu, gfx_target.as_deref(), &os_family, is_wsl);

    HostCapability {
        os_family,
        is_wsl,
        gfx_target,
        has_amd_gpu,
        available_engines,
        effective_serve_engine,
        platform_slug,
    }
}

/// Run `rocm <args>` with an isolated config/data/cache root, returning stdout
/// (empty string on any failure — the probe must never panic the suite).
fn run_probe(root: &std::path::Path, args: &[&str]) -> String {
    let mut cmd = std::process::Command::new(rocm_binary());
    cmd.args(args);
    cmd.env("ROCM_CLI_CONFIG_DIR", root.join("config"));
    cmd.env("ROCM_CLI_DATA_DIR", root.join("data"));
    cmd.env("ROCM_CLI_CACHE_DIR", root.join("cache"));
    match cmd.output() {
        Ok(out) => String::from_utf8_lossy(&out.stdout).into_owned(),
        Err(_) => String::new(),
    }
}

/// The `examine` fields the capability probe reads.
struct ExamineFacts {
    os_family: String,
    is_wsl: bool,
    gfx_target: Option<String>,
    /// `driver_status:` — the CLI's own verdict on whether the runtime can
    /// reach the device. Only consulted on WSL (see [`host_has_usable_gpu`]).
    driver_status: String,
}

/// `driver_status` value meaning ROCm's WSL passthrough path is complete.
///
/// The CLI sets this only when `/dev/dxg`, dxcore, `librocdxg.so` and its
/// ldconfig entry are all present; the other WSL states (`wsl_rocdxg_missing`,
/// `wsl_gpu_plumbing_missing`) mean the runtime cannot reach the GPU.
const WSL_DRIVER_READY: &str = "wsl_rocdxg_ready";

/// Extract the probe's facts from the human `rocm examine` text (format string
/// in rocm-core `ExamineSummary`): lines like `  os: linux`,
/// `  detected_gfx_target: gfx942`, `  wsl: false`, `  driver_status: ...`. A
/// missing/placeholder (`<unknown>`, empty, `none`) gfx target yields `None`
/// (→ treated as no GPU). Tolerant: an unrecognized dump degrades to a
/// mock-like host.
fn parse_examine_text(text: &str) -> ExamineFacts {
    let mut os_family = "other".to_owned();
    let mut is_wsl = false;
    let mut gfx_target = None;
    let mut driver_status = String::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("os:") {
            os_family = v.trim().to_ascii_lowercase();
        } else if let Some(v) = line.strip_prefix("detected_gfx_target:") {
            let v = v.trim();
            if !v.is_empty() && v != "<unknown>" && v != "none" {
                gfx_target = Some(v.to_owned());
            }
        } else if let Some(v) = line.strip_prefix("wsl:") {
            is_wsl = matches!(v.trim(), "true" | "yes" | "1");
        } else if let Some(v) = line.strip_prefix("driver_status:") {
            driver_status = v.trim().to_owned();
        }
    }
    ExamineFacts {
        os_family,
        is_wsl,
        gfx_target,
        driver_status,
    }
}

/// Whether `@requires-gpu` scenarios can actually run on this host.
///
/// A detected gfx target is enough on a native host. It is NOT enough under
/// WSL: the target is reported from the Windows-side driver even when ROCm has
/// no path to the device, so a distro missing `librocdxg.so` advertises
/// `gfx1151` while `rocm serve` refuses with "no usable AMD GPU". Taking the
/// target at face value there runs every GPU scenario against a host that
/// cannot serve one — they fail on their premise rather than resolving to
/// not-applicable, which is exactly what the capability probe exists to avoid.
fn host_has_usable_gpu(gfx_target: Option<&str>, is_wsl: bool, driver_status: &str) -> bool {
    let hip = std::env::var_os("HIP_VISIBLE_DEVICES");
    let rocr = std::env::var_os("ROCR_VISIBLE_DEVICES");
    host_has_usable_gpu_with_mask(
        gfx_target,
        is_wsl,
        driver_status,
        selected_visibility_mask(hip.as_deref(), rocr.as_deref()),
    )
}

/// Match the product's visibility-mask precedence: HIP wins when present,
/// including an explicitly empty value; ROCR is the fallback.
fn selected_visibility_mask<'a>(
    hip: Option<&'a std::ffi::OsStr>,
    rocr: Option<&'a std::ffi::OsStr>,
) -> Option<&'a std::ffi::OsStr> {
    hip.or(rocr)
}

/// Pure form of [`host_has_usable_gpu`] for contract tests.
///
/// The harness knows whether a gfx target exists, but not the product's complete
/// device topology. As in `rocm_core::has_usable_amd_gpu`, only an authoritative
/// empty mask proves zero usable devices; nonempty/opaque masks stay eligible and
/// let the product perform ordinal/UUID validation at launch time.
fn host_has_usable_gpu_with_mask(
    gfx_target: Option<&str>,
    is_wsl: bool,
    driver_status: &str,
    visibility_mask: Option<&std::ffi::OsStr>,
) -> bool {
    gfx_target.is_some()
        && (!is_wsl || driver_status == WSL_DRIVER_READY)
        && visibility_mask.is_none_or(|mask| !mask.is_empty())
}

/// Parse engine names from `rocm engines list`. Engine rows are the lines whose
/// first non-space token is a known engine name (optionally prefixed by the `*`
/// default marker), before the indented `adapter:`/`runtime:` detail lines.
fn parse_engines_list(text: &str) -> Vec<String> {
    let mut engines = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start_matches(['*', ' ']);
        if trimmed.is_empty() {
            continue;
        }
        let first = trimmed.split_whitespace().next().unwrap_or("");
        if matches!(first, "lemonade" | "vllm") && !engines.iter().any(|e| e == first) {
            engines.push(first.to_owned());
        }
    }
    engines
}

/// Stable platform identity from hardware and host environment. WSL is a
/// distinct platform even without a GPU, so its report never collides with the
/// ordinary hosted mock column. Otherwise no AMD GPU → "mock"; GPU hosts use a
/// coarse slug from the gfx family (data-center → "mi300x"; Strix gfx115x →
/// "strix-halo"), falling back to the normalized family. The OS is appended for
/// families that ship on more than one OS (Strix Halo runs both Ubuntu and
/// Windows on the same gfx1151), so those become distinct grid columns rather
/// than colliding into one.
fn derive_platform_slug(
    has_amd_gpu: bool,
    gfx_target: Option<&str>,
    os_family: &str,
    is_wsl: bool,
) -> String {
    if is_wsl {
        return match gfx_target {
            Some(t) => format!("{}-wsl", platform_hardware_slug(t)),
            None => "wsl".to_owned(),
        };
    }
    if !has_amd_gpu {
        return "mock".to_owned();
    }
    match gfx_target {
        Some(t) => {
            let hardware = platform_hardware_slug(t);
            if hardware == "strix-halo" {
                // Same silicon on Ubuntu and Windows — disambiguate by OS.
                format!("{hardware}-{}", os_normalized(os_family))
            } else {
                hardware
            }
        }
        None => "mock".to_owned(),
    }
}

fn platform_hardware_slug(gfx_target: &str) -> String {
    let family = normalize_family(gfx_target);
    if family.ends_with("-dcgpu") {
        "mi300x".to_owned()
    } else if family.starts_with("gfx115") {
        "strix-halo".to_owned()
    } else {
        family
    }
}

/// Short OS token for a platform slug: "windows" / "linux" / else the raw value.
fn os_normalized(os_family: &str) -> String {
    let o = os_family.trim().to_ascii_lowercase();
    if o.contains("windows") {
        "windows".to_owned()
    } else if o.contains("linux") {
        "linux".to_owned()
    } else {
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_wheel_runtime_key_ignores_markers_and_legacy_entries() {
        let inventory = "registered ROCm runtimes\n  active_runtime_key: <unset>\n  installed:\n    release-wheel-gfx94x-dcgpu-7-13-0 runtime_id=therock-release:gfx94X-dcgpu\n  * release-wheel-multi-arch-7-14-0 runtime_id=therock-release:gfx94X-dcgpu\n";
        assert_eq!(
            canonical_wheel_runtime_key(inventory),
            Some("release-wheel-multi-arch-7-14-0")
        );
    }

    #[test]
    fn canonical_wheel_runtime_key_returns_none_without_canonical_entry() {
        assert_eq!(
            canonical_wheel_runtime_key(
                "  release-wheel-gfx94x-dcgpu-7-13-0 runtime_id=therock-release:gfx94X-dcgpu"
            ),
            None
        );
    }

    // Drift guard (decision #1): these pin the re-implemented rule to the
    // product's known behaviour. When task #16 lands a product probe field,
    // this same table becomes the consistency check (harness rule == probe).

    #[test]
    fn mi300x_dcgpu_prefers_vllm() {
        assert_eq!(effective_serve_engine(Some("gfx942"), "linux"), "vllm");
        assert_eq!(
            effective_serve_engine(Some("gfx94X-dcgpu"), "linux"),
            "vllm"
        );
        assert_eq!(effective_serve_engine(Some("gfx950"), "linux"), "vllm");
    }

    #[test]
    fn legacy_dcgpu_set_prefers_vllm() {
        for t in ["gfx906", "gfx908", "gfx90a"] {
            assert_eq!(effective_serve_engine(Some(t), "linux"), "vllm", "{t}");
        }
    }

    #[test]
    fn strix_halo_defaults_to_lemonade() {
        // gfx1151 is NOT a vLLM-preferred family → default engine (lemonade),
        // on Linux AND Windows.
        assert_eq!(effective_serve_engine(Some("gfx1151"), "linux"), "lemonade");
        assert_eq!(
            effective_serve_engine(Some("gfx1151"), "windows"),
            "lemonade"
        );
    }

    #[test]
    fn native_windows_never_prefers_vllm() {
        // Even a data-center family cannot use vLLM on native Windows.
        assert_eq!(
            effective_serve_engine(Some("gfx942"), "windows"),
            "lemonade"
        );
    }

    #[test]
    fn no_gpu_defaults_to_lemonade() {
        assert_eq!(effective_serve_engine(None, "other"), "lemonade");
    }

    #[test]
    fn engine_available_respects_platform() {
        let strix = HostCapability {
            os_family: "windows".to_owned(),
            is_wsl: false,
            gfx_target: Some("gfx1151".to_owned()),
            has_amd_gpu: true,
            available_engines: vec!["lemonade".to_owned(), "vllm".to_owned()],
            effective_serve_engine: "lemonade".to_owned(),
            platform_slug: "strix-halo".to_owned(),
        };
        assert!(strix.engine_available("lemonade"));
        // vLLM adapter is "built-in" but cannot start on Windows / non-dcgpu.
        assert!(!strix.engine_available("vllm"));

        let mi300x = HostCapability {
            os_family: "linux".to_owned(),
            is_wsl: false,
            gfx_target: Some("gfx942".to_owned()),
            has_amd_gpu: true,
            available_engines: vec!["lemonade".to_owned(), "vllm".to_owned()],
            effective_serve_engine: "vllm".to_owned(),
            platform_slug: "mi300x".to_owned(),
        };
        assert!(mi300x.engine_available("vllm"));
        assert!(mi300x.engine_available("lemonade"));
    }

    #[test]
    fn parses_examine_text_gpu_host() {
        // Human `rocm examine` dump (subset of the real format).
        let text = "\
rocm examine
  os: linux
  arch: x86_64
  detected_gfx_target: gfx942
  detected_therock_family: gfx94X-dcgpu
  wsl: false
  driver_status: ok
";
        let facts = parse_examine_text(text);
        assert_eq!(facts.os_family, "linux");
        assert!(!facts.is_wsl);
        assert_eq!(facts.gfx_target.as_deref(), Some("gfx942"));
        assert_eq!(facts.driver_status, "ok");
    }

    #[test]
    fn parses_examine_text_mock_host() {
        // No GPU: detected_gfx_target is the <unknown> placeholder.
        let text = "\
rocm examine
  os: other
  detected_gfx_target: <unknown>
  wsl: false
";
        let facts = parse_examine_text(text);
        assert_eq!(facts.os_family, "other");
        assert_eq!(facts.gfx_target, None);
    }

    /// A WSL distro advertises the Windows-side gfx target whether or not ROCm
    /// can reach it, so the driver verdict is what decides. Without this, every
    /// `@requires-gpu` scenario runs on a host where `serve` cannot start and
    /// fails on its premise instead of resolving to not-applicable.
    #[test]
    fn wsl_gpu_requires_a_ready_rocdxg_path() {
        // Real values from the self-hosted WSL runner before librocdxg was
        // installed: /dev/dxg present, the ROCm passthrough library missing.
        assert!(!host_has_usable_gpu_with_mask(
            Some("gfx1151"),
            true,
            "wsl_rocdxg_missing",
            None
        ));
        assert!(!host_has_usable_gpu_with_mask(
            Some("gfx1151"),
            true,
            "wsl_gpu_plumbing_missing",
            None
        ));
        // Complete passthrough: the GPU is usable and the scenarios must run.
        assert!(host_has_usable_gpu_with_mask(
            Some("gfx1151"),
            true,
            WSL_DRIVER_READY,
            None
        ));
        // A native host is unaffected by the driver verdict.
        assert!(host_has_usable_gpu_with_mask(
            Some("gfx942"),
            false,
            "amdgpu_available",
            None
        ));
        assert!(host_has_usable_gpu_with_mask(
            Some("gfx942"),
            false,
            "",
            None
        ));
        // No gfx target is still no GPU, WSL or not.
        assert!(!host_has_usable_gpu_with_mask(
            None,
            true,
            WSL_DRIVER_READY,
            None
        ));
        assert!(!host_has_usable_gpu_with_mask(
            None,
            false,
            "amdgpu_available",
            None
        ));
    }

    #[test]
    fn gpu_capability_honors_the_product_visibility_mask_precedence() {
        use std::ffi::OsStr;

        let empty = OsStr::new("");
        let device_zero = OsStr::new("0");
        assert_eq!(
            selected_visibility_mask(Some(empty), Some(device_zero)),
            Some(empty),
            "HIP_VISIBLE_DEVICES must take precedence even when explicitly empty"
        );
        assert_eq!(
            selected_visibility_mask(None, Some(empty)),
            Some(empty),
            "ROCR_VISIBLE_DEVICES applies when HIP_VISIBLE_DEVICES is unset"
        );

        assert!(!host_has_usable_gpu_with_mask(
            Some("gfx1151"),
            true,
            WSL_DRIVER_READY,
            Some(empty)
        ));
        assert!(!host_has_usable_gpu_with_mask(
            Some("gfx942"),
            false,
            "amdgpu_available",
            Some(empty)
        ));
        assert!(host_has_usable_gpu_with_mask(
            Some("gfx1151"),
            true,
            WSL_DRIVER_READY,
            Some(device_zero)
        ));
    }

    #[test]
    fn parses_engines_list() {
        let text = "\
Local model engines
  Built-in engines are included with rocm-cli.
* lemonade   default embedded Lemonade server with ROCm llama.cpp backend
    adapter: built-in
    runtime: not found
  vllm       Linux/WSL ROCm GPU serving engine through external vLLM
    adapter: built-in
    runtime: not found
  protocol: 0.1.0
";
        assert_eq!(parse_engines_list(text), vec!["lemonade", "vllm"]);
    }

    #[test]
    fn platform_slug_derivation() {
        assert_eq!(derive_platform_slug(false, None, "other", false), "mock");
        assert_eq!(
            derive_platform_slug(true, Some("gfx942"), "linux", false),
            "mi300x"
        );
        // Strix Halo: same gfx1151 silicon on both OSes → distinct slugs so the
        // report grid gets a column per platform, not a collision.
        assert_eq!(
            derive_platform_slug(true, Some("gfx1151"), "linux", false),
            "strix-halo-linux"
        );
        assert_eq!(
            derive_platform_slug(true, Some("gfx1151"), "windows", false),
            "strix-halo-windows"
        );
        // Hosted WSL has no GPU but must not collide with the ordinary mock
        // column. A future WSL GPU lane also remains distinct from native Linux.
        assert_eq!(derive_platform_slug(false, None, "linux", true), "wsl");
        assert_eq!(
            derive_platform_slug(true, Some("gfx1151"), "linux", true),
            "strix-halo-wsl"
        );
    }

    fn write_manifest(runtimes_dir: &std::path::Path, key: &str, version: &str) {
        let registry = runtimes_dir.join("registry");
        std::fs::create_dir_all(&registry).expect("create registry");
        let install_root = runtimes_dir.join("wheel").join(key);
        std::fs::create_dir_all(&install_root).expect("create install root");
        std::fs::write(
            registry.join(format!("{key}.json")),
            serde_json::json!({
                "runtime_key": key,
                "version": version,
                "install_root": install_root,
            })
            .to_string(),
        )
        .expect("write manifest");
    }

    /// The report must name the ROCm version the run actually served on. The
    /// pre-warm keeps a newer runtime beside the old one, so picking whichever
    /// manifest `read_dir` yielded first could attribute a run to the version it
    /// did NOT use — and read as fact.
    #[test]
    fn reports_the_active_runtimes_version_when_several_are_installed() {
        let tmp = tempfile::TempDir::with_prefix("capability-").expect("temp dir");
        let dir = tmp.path();
        write_manifest(dir, "release-wheel-gfx94x-dcgpu-7-13-0", "7.13.0");
        write_manifest(dir, "release-wheel-multi-arch-7-14-0", "7.14.0");
        std::fs::write(
            dir.join("active.json"),
            r#"{"runtime_key": "release-wheel-multi-arch-7-14-0"}"#,
        )
        .expect("write marker");

        let (version, root) = active_runtime_install_root(dir).expect("a runtime is named");
        assert_eq!(version, "7.14.0");
        assert_eq!(
            root,
            dir.join("wheel").join("release-wheel-multi-arch-7-14-0")
        );
    }

    /// Several runtimes and no marker: report nothing rather than guess. An
    /// absent version reads as unknown; a wrong one reads as fact.
    #[test]
    fn reports_no_version_when_the_tree_names_no_runtime() {
        let tmp = tempfile::TempDir::with_prefix("capability-").expect("temp dir");
        let dir = tmp.path();
        write_manifest(dir, "release-wheel-gfx94x-dcgpu-7-13-0", "7.13.0");
        write_manifest(dir, "release-wheel-multi-arch-7-14-0", "7.14.0");

        assert!(active_runtime_install_root(dir).is_none());
    }
}
