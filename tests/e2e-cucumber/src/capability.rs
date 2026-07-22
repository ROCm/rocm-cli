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
//! IMPORTANT — the effective serve engine is currently RE-IMPLEMENTED here (see
//! [`effective_serve_engine`]) because no `rocm` command exposes it directly
//! (`examine`'s `default_engine` is a constant `"lemonade"` decoy, not what
//! `serve` selects). This duplicates the product's `select_serve_engine` /
//! `preferred_serve_engine_for_therock_family` logic and WILL drift if the
//! product changes engine support. Task #16 tracks adding a product probe
//! (`examine --json` → `effective_serve_engine`) so the harness can read the
//! product's own decision instead. Until then, the unit tests below guard drift.

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
    /// Whether an AMD GPU was detected (a real `detected_gfx_target` is present).
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
/// registry: prefer the runtime named by `active.json`, else the sole installed
/// manifest. Returns `None` when nothing is installed.
///
/// The install_root is resolved from `runtimes_dir` (the shared tree we were
/// handed) as `<runtimes_dir>/wheel/<runtime_key>`, NOT from the manifest's own
/// `install_root` field. That field records the absolute path where the runtime
/// was first installed — on Strix a per-scenario temp dir that no longer exists
/// by report time — so trusting it made `vllm`/`lemonade` probe a dead path and
/// come back `None`. On MI300X the two coincide (prewarm installs in place),
/// which is why it worked there but not on Strix. Falls back to the manifest
/// path if the derived one is absent, for any tree that predates the wheel layout.
fn active_runtime_install_root(
    runtimes_dir: &std::path::Path,
) -> Option<(String, std::path::PathBuf)> {
    let registry = runtimes_dir.join("registry");
    let entries: Vec<std::path::PathBuf> = std::fs::read_dir(&registry)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    // Prefer the active runtime's key if active.json names one.
    let active_key = std::fs::read_to_string(runtimes_dir.join("active.json"))
        .ok()
        .and_then(|t| {
            serde_json::from_str::<serde_json::Value>(&t)
                .ok()?
                .get("runtime_key")?
                .as_str()
                .map(str::to_owned)
        });
    let pick = entries
        .iter()
        .find(|p| {
            active_key
                .as_deref()
                .is_some_and(|k| p.file_stem().and_then(|s| s.to_str()) == Some(k))
        })
        .or_else(|| entries.first())?;
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(pick).ok()?).ok()?;
    let version = json.get("version")?.as_str()?.to_owned();
    // Runtime key = the manifest file stem (e.g. release-wheel-gfx1151-7-13-0).
    let key = pick.file_stem().and_then(|s| s.to_str());
    // Resolve the root inside the shared tree first; fall back to the manifest's
    // recorded install_root only if that derived path doesn't exist.
    let derived = key.map(|k| runtimes_dir.join("wheel").join(k));
    let root = match derived {
        Some(d) if d.is_dir() => d,
        _ => std::path::PathBuf::from(json.get("install_root")?.as_str()?),
    };
    Some((version, root))
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

    let (os_family, is_wsl, gfx_target) = parse_examine_text(&examine);
    let has_amd_gpu = gfx_target.is_some();
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

/// Extract `(os_family, is_wsl, first_gfx_target)` from the human `rocm examine`
/// text (format string in rocm-core `ExamineSummary`): lines like `  os: linux`,
/// `  detected_gfx_target: gfx942`, `  wsl: false`. A missing/placeholder
/// (`<unknown>`, empty, `none`) gfx target yields `None` (→ treated as no GPU).
/// Tolerant: an unrecognized dump degrades to a mock-like host.
fn parse_examine_text(text: &str) -> (String, bool, Option<String>) {
    let mut os_family = "other".to_owned();
    let mut is_wsl = false;
    let mut gfx_target = None;
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
        }
    }
    (os_family, is_wsl, gfx_target)
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
        let (os, wsl, gfx) = parse_examine_text(text);
        assert_eq!(os, "linux");
        assert!(!wsl);
        assert_eq!(gfx.as_deref(), Some("gfx942"));
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
        let (os, _, gfx) = parse_examine_text(text);
        assert_eq!(os, "other");
        assert_eq!(gfx, None);
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
}
