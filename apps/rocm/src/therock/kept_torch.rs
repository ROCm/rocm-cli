// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Keeping the torch a runtime already holds when an SDK install runs over it.
//!
//! The wheel install hands `uv` one transaction: `rocm`, `torch`, `torchvision`
//! and `torchaudio`, all from the SDK index. Run over an environment that already
//! holds a different torch (the one an engine installed, or one the user built)
//! that transaction replaces the stack, and `triton` with it. The alignment
//! opt-out, `ROCM_CLI_DISABLE_TORCH_ALIGNMENT`, only governs the step that runs
//! after the install, so without this module the opt-out was honoured everywhere
//! except in the command that rewrites torch first.
//!
//! Set, the opt-out leaves the torch stack out of the transaction when the
//! environment already has torch, and installs only the ROCm packages. An
//! environment with no torch gets the full stack, since there is nothing to keep.
//!
//! Builds on `process` for running the interpreter; nothing in `process` depends
//! on this module.

use super::process::capture_python_stdout_with_env;
use anyhow::{Context, Result};
use std::path::Path;

/// The packages whose requirements the opt-out removes from the SDK install.
///
/// `triton` is not listed because the install never names it: it arrives as a
/// dependency of `torch`, so leaving `torch` out leaves it where it is.
const TORCH_SPEC_PACKAGES: [&str; 3] = ["torch", "torchvision", "torchaudio"];

/// The torch stack an environment held before the install, with the versions it
/// held them at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct KeptTorch {
    installed: Vec<(String, String)>,
}

impl KeptTorch {
    /// The stack as it is named in the plan and the progress line, for example
    /// `torch 2.11.0+gitd0c8b1f, torchvision 0.24.1+d801a34`.
    pub(super) fn describe(&self) -> String {
        self.installed
            .iter()
            .map(|(name, version)| format!("{name} {version}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Decide what to keep: nothing unless the user opted out *and* the environment
/// has a torch to keep.
///
/// `installed` is asked only when the opt-out is set, so an install that did not
/// opt out never launches the interpreter for an answer it would not use. A probe
/// that fails keeps nothing: the environment is then treated as the empty one the
/// install has always assumed, which is the behaviour the opt-out is a request to
/// change, not a promise to refuse the install over.
pub(super) fn kept_torch_stack(
    opted_out: bool,
    installed: impl FnOnce() -> Result<Vec<(String, String)>>,
) -> Option<KeptTorch> {
    if !opted_out {
        return None;
    }
    let installed = installed().ok()?;
    installed
        .iter()
        .any(|(name, _)| name == "torch")
        .then_some(KeptTorch { installed })
}

/// The requirements the install hands `uv`: all of them, or all but the torch
/// stack when one is being kept.
pub(super) fn install_specs(specs: &[String], kept: Option<&KeptTorch>) -> Vec<String> {
    if kept.is_none() {
        return specs.to_vec();
    }
    specs
        .iter()
        .filter(|spec| !TORCH_SPEC_PACKAGES.contains(&spec_package_name(spec)))
        .cloned()
        .collect()
}

/// The distribution a requirement names: what comes before an extras bracket, a
/// version operator or an environment marker.
fn spec_package_name(spec: &str) -> &str {
    let end = spec
        .find(['[', '=', '<', '>', '!', '~', ';', ' '])
        .unwrap_or(spec.len());
    spec[..end].trim()
}

/// The plan line that says the torch stack is not being installed, made from the
/// same value that filtered the requirements.
pub(super) fn plan_line(kept: &KeptTorch) -> String {
    format!(
        "  torch_kept: ROCM_CLI_DISABLE_TORCH_ALIGNMENT is set and this runtime already holds {}; the torch, torchvision and torchaudio requirements are left out of the install",
        kept.describe()
    )
}

/// Ask the environment's own interpreter which torch packages it holds.
///
/// Distribution metadata, not an import: importing torch loads the GPU libraries
/// and can fail or stall for reasons unrelated to what is installed. An
/// environment that does not exist yet has no interpreter to ask, and yields an
/// empty stack.
pub(super) fn installed_torch_stack(env_python: &Path) -> Result<Vec<(String, String)>> {
    if !env_python.exists() {
        return Ok(Vec::new());
    }
    let text = capture_python_stdout_with_env(
        env_python,
        INSTALLED_TORCH_SCRIPT,
        &[],
        "read the installed torch stack",
    )
    .with_context(|| {
        format!(
            "failed to read the installed torch stack via {}",
            env_python.display()
        )
    })?;
    parse_installed_stack(&text)
}

fn parse_installed_stack(output: &str) -> Result<Vec<(String, String)>> {
    let versions: std::collections::BTreeMap<String, String> =
        serde_json::from_str(output.trim()).context("failed to parse the installed torch stack")?;
    // Fixed order, so the line printed for it reads the same on every run.
    Ok(rocm_core::SDK_TORCH_STACK_PACKAGES
        .iter()
        .filter_map(|name| {
            versions
                .get(*name)
                .map(|version| ((*name).to_owned(), version.clone()))
        })
        .collect())
}

const INSTALLED_TORCH_SCRIPT: &str = r#"
import importlib.metadata as md
import json

out = {}
for name in ("torch", "torchvision", "torchaudio", "triton"):
    try:
        out[name] = md.version(name)
    except md.PackageNotFoundError:
        pass
print(json.dumps(out))
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn specs() -> Vec<String> {
        [
            "rocm[libraries,device-gfx942]==7.14.1",
            "torch[device-gfx942]==2.11.0+rocm7.14.1",
            "torchvision[device-gfx942]==0.26.0+rocm7.14.1",
            "torchaudio==2.11.0+rocm7.14.1",
        ]
        .map(str::to_owned)
        .to_vec()
    }

    fn held(names: &[(&str, &str)]) -> Result<Vec<(String, String)>> {
        Ok(names
            .iter()
            .map(|(name, version)| ((*name).to_owned(), (*version).to_owned()))
            .collect())
    }

    #[test]
    fn without_the_opt_out_nothing_is_kept_and_the_environment_is_not_asked() {
        let kept = kept_torch_stack(false, || panic!("the interpreter must not be launched"));
        assert_eq!(kept, None);
        assert_eq!(install_specs(&specs(), kept.as_ref()), specs());
    }

    #[test]
    fn an_opted_out_environment_with_torch_installs_only_the_rocm_packages() {
        let kept = kept_torch_stack(true, || {
            held(&[("torch", "2.11.0+gitd0c8b1f"), ("triton", "3.6.0")])
        })
        .expect("a torch to keep");
        assert_eq!(kept.describe(), "torch 2.11.0+gitd0c8b1f, triton 3.6.0");
        assert_eq!(
            install_specs(&specs(), Some(&kept)),
            vec!["rocm[libraries,device-gfx942]==7.14.1".to_owned()],
            "torch, torchvision and torchaudio are all left to the user"
        );
    }

    #[test]
    fn an_opted_out_environment_without_torch_gets_the_full_stack() {
        for installed in [held(&[]), held(&[("triton", "3.6.0")])] {
            let kept = kept_torch_stack(true, || installed);
            assert_eq!(kept, None, "there is no torch to keep");
            assert_eq!(install_specs(&specs(), kept.as_ref()), specs());
        }
    }

    #[test]
    fn an_environment_that_cannot_be_read_keeps_nothing() {
        let kept = kept_torch_stack(true, || anyhow::bail!("the interpreter would not start"));
        assert_eq!(kept, None);
    }

    #[test]
    fn only_the_torch_requirements_are_dropped() {
        let kept = KeptTorch {
            installed: vec![("torch".to_owned(), "2.11.0".to_owned())],
        };
        let specs = [
            "rocm[libraries]==7.14.1",
            "rocm-sdk-devel==7.14.1",
            "torch-geometric==2.6.0",
            "torchaudio==2.11.0",
        ]
        .map(str::to_owned)
        .to_vec();
        assert_eq!(
            install_specs(&specs, Some(&kept)),
            vec![
                "rocm[libraries]==7.14.1".to_owned(),
                "rocm-sdk-devel==7.14.1".to_owned(),
                "torch-geometric==2.6.0".to_owned(),
            ],
            "a package that merely starts with `torch` is not part of the stack"
        );
    }

    #[test]
    fn a_requirement_is_named_by_what_precedes_its_extras_and_version() {
        for (spec, name) in [
            ("torch==2.11.0", "torch"),
            ("torch[device-gfx942]==2.11.0+rocm7.14.1", "torch"),
            ("torchvision>=0.20", "torchvision"),
            ("torchaudio ; platform_system == 'Linux'", "torchaudio"),
            ("rocm", "rocm"),
        ] {
            assert_eq!(spec_package_name(spec), name, "{spec}");
        }
    }

    #[test]
    fn the_installed_stack_is_read_in_a_fixed_order_and_skips_what_is_absent() {
        let stack = parse_installed_stack(
            r#"{"triton": "3.6.0", "torchaudio": "2.9.0+eaa9e4e", "torch": "2.11.0+gitd0c8b1f"}"#,
        )
        .expect("a stack");
        assert_eq!(
            stack,
            vec![
                ("torch".to_owned(), "2.11.0+gitd0c8b1f".to_owned()),
                ("torchaudio".to_owned(), "2.9.0+eaa9e4e".to_owned()),
                ("triton".to_owned(), "3.6.0".to_owned()),
            ]
        );
        assert!(parse_installed_stack("not json").is_err());
    }

    #[test]
    fn the_plan_line_names_what_is_kept_and_what_is_left_out() {
        let kept = kept_torch_stack(true, || {
            held(&[
                ("torch", "2.11.0+gitd0c8b1f"),
                ("torchvision", "0.24.1+d801a34"),
            ])
        })
        .expect("a torch to keep");
        let line = plan_line(&kept);
        assert!(line.contains("torch 2.11.0+gitd0c8b1f, torchvision 0.24.1+d801a34"));
        assert!(line.contains("ROCM_CLI_DISABLE_TORCH_ALIGNMENT"));
        // The e2e step `the SDK install leaves the torch the runtime held as it was`
        // reads the torch version out of this line by the words around it, and checks
        // it against the runtime's interpreter. That step is what proves the torch was
        // kept; this only pins the shape the step parses.
        assert!(
            line.contains(" already holds torch 2.11.0+gitd0c8b1f,"),
            "{line}"
        );
    }

    /// The interpreter really is asked: a stand-in environment whose `python` is a
    /// wrapper that exposes one fake `torch` distribution, read back through the
    /// same script an install runs. Parsing the script's JSON is tested above;
    /// this is what ties the script to the parser.
    #[test]
    #[cfg(unix)]
    fn the_installed_stack_is_read_from_a_real_interpreter() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "rocm-cli-kept-torch-{}-{}",
            std::process::id(),
            rocm_core::unix_time_millis()
        ));
        let dist_info = root.join("site").join("torch-2.9.0+gittest.dist-info");
        std::fs::create_dir_all(&dist_info).unwrap();
        std::fs::write(
            dist_info.join("METADATA"),
            "Metadata-Version: 2.1\nName: torch\nVersion: 2.9.0+gittest\n",
        )
        .unwrap();
        let python = root.join("python");
        std::fs::write(
            &python,
            format!(
                "#!/bin/sh\nPYTHONPATH='{}' exec python3 \"$@\"\n",
                root.join("site").display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755)).unwrap();

        let stack = installed_torch_stack(&python).expect("the interpreter answers");
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(
            stack,
            vec![("torch".to_owned(), "2.9.0+gittest".to_owned())],
            "only the torch distribution the environment has is reported"
        );
    }

    #[test]
    fn a_missing_environment_holds_no_torch() {
        let stack = installed_torch_stack(Path::new("/nonexistent/venv/bin/python"))
            .expect("a missing environment is empty, not an error");
        assert!(stack.is_empty());
    }
}
