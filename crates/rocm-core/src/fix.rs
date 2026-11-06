// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Apply remediations for diagnosed ROCm failure modes.
//!
//! Rust port of the `rocm-doctor` skill's `apply_fix.py`. Only small, safe,
//! well-bounded fixes are auto-applicable (the runners in `fix/runners.rs`);
//! everything else is advisory and only prints its plan. The consent model
//! mirrors the Python: print the exact change, honor `--dry-run`, refuse on a
//! non-interactive shell without `--yes`, and otherwise confirm before mutating
//! anything.
//!
//! Exit codes are named in [`exit`] and published in the catalog manifest, so
//! the number, its meaning, and what a reader is told it means all come from one
//! place.
//!
//! This file is the module root. The exit codes, [`FixOptions`] and [`FixClass`]
//! live in `fix/options.rs`, and the consent decision, the timeouts and the
//! `fail!` helper in `fix/consent.rs`; the names are re-exported from here, so
//! `crate::fix::exit` and `rocm_core::fix::FixClass` keep resolving. The
//! runners, one per auto-applicable fix, are in `fix/runners.rs`. The closed
//! catalog (`RECIPES`, `FixRecipe`, the lookups by fix-id and the test-only
//! oracles) is in `fix/catalog.rs`, its rendering in `fix/render.rs` and its
//! published JSON contract in `fix/manifest.rs`. What stays here is the
//! dispatch: [`apply`], and the plan it makes and acts on.

mod catalog;
mod consent;
mod manifest;
mod options;
mod render;
mod runners;

use catalog::{FixRecipe, find_recipe};
use consent::{current_os, fail};
use render::{
    diagnosis_position_refusal, looks_like_a_diagnosis_position, print_recipe, verify_steps,
};
use std::io::Write;

pub(crate) use catalog::{COMGR_CONFLICT_NEITHER_OPTION_RECOMMENDED, is_catalog_id};
pub use catalog::{class_on, needs_on};
pub use manifest::{
    CATALOG_CONTRACT_VERSION, CatalogManifest, ManifestEntry, ManifestPlatform, catalog_manifest,
    catalog_manifest_json,
};
pub use options::{FixClass, FixOptions, exit};
pub(crate) use render::format_flags;
pub use render::list_recipes;
// The catalog's test-only oracles, which the `diagnose` module holds its own copies of
// a recipe to.
#[cfg(test)]
pub(crate) use catalog::{
    OLDEST_ROCM_MAJOR_THE_CLI_INSTALLS, assert_engine_shell_boundary_is_labelled,
    assert_needs_reboot_matches_the_catalog, assert_plan_matches_the_catalog_copy,
    fails_in_windows_powershell, recipe_applies_on, recipe_linux_verify,
    recipe_verify_notes_and_run_commands, torch_rocm_indexes_named_in,
};
#[cfg(all(test, windows))]
pub(crate) use catalog::{ShellRun, run_typed_in_cmd_and_powershell};

/// Apply (or print) the fix identified by `fix_id`. Returns the process exit code.
#[must_use]
pub fn apply(fix_id: &str, opts: &FixOptions) -> i32 {
    let Some(recipe) = find_recipe(fix_id) else {
        fail!("Unknown fix-id: {fix_id}");
        if looks_like_a_diagnosis_position(fix_id) {
            // `rocm diagnose` ranks findings `#1`, `#2`, and users reach for that
            // number here. It is a position in one report, not a name -- and it
            // does not line up with the catalog's `fix-N` names either, so a
            // bare "unknown id" left them with nothing to correct.
            for line in diagnosis_position_refusal(fix_id) {
                fail!("{line}");
            }
        } else {
            fail!("Run `rocm diagnose` to see which fix-id applies.");
        }
        return exit::UNKNOWN_ID;
    };
    act_on(recipe, opts)
}

/// What `rocm fix <id>` will do with an entry on this host.
///
/// Every decision lives in [`plan_of`] and every message lives in [`act_on`], so
/// the two can be checked apart. The split exists because several of these end
/// in exit code 0: a test that watched only the code could not tell
/// [`Plan::Unfixable`] from [`Plan::PrintSteps`], and the difference matters —
/// the second tells the user to copy commands, which an unfixable entry does not
/// have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plan {
    /// The entry does not apply on the running OS.
    WrongMachine,
    /// No reliable fix exists. Name the problem and change nothing.
    Unfixable,
    /// Hand off to the recipe's runner.
    Run,
    /// The catalog says the CLI carries this out, and nothing can.
    MissingRunner,
    /// Print the steps for the user to carry out.
    PrintSteps,
}

/// Decide what to do with `recipe` here. No output, no side effects.
fn plan_of(recipe: &FixRecipe) -> Plan {
    let Some(platform) = recipe.platform(current_os()) else {
        return Plan::WrongMachine;
    };
    if platform.class == FixClass::DiagnoseOnly {
        return Plan::Unfixable;
    }
    // The class says what happens to the *machine*; it does not say whether a
    // runner runs. `fix-2-unset-override` is print-only on Linux and still has
    // real work to do there -- it reads the live value and scans five rc files
    // to name the ones that set it. Gating the runner on the class would throw
    // that away and print a generic "copy the commands above" in its place.
    if recipe.runner.is_some() {
        return Plan::Run;
    }
    if platform.class.applies_itself() {
        return Plan::MissingRunner;
    }
    Plan::PrintSteps
}

/// Everything `apply` does once the id has resolved to a recipe.
///
/// Split out from [`apply`] so the class gate can be reached with a recipe the
/// catalog does not contain, which keeps every class's branch reachable from a
/// test however few entries the catalog happens to carry for it.
/// [`FixClass::DiagnoseOnly`] had no member at all until
/// `info-20-gpu-no-rocm-support`, the first; the seam is what exercised its
/// branch before that, and still covers any class the catalog later empties.
#[must_use]
fn act_on(recipe: &FixRecipe, opts: &FixOptions) -> i32 {
    print_recipe(recipe);
    println!();

    // Exhaustive on purpose. An arm removed here is a compile error rather than
    // a silent fall-through to the next branch, which is how a disabled guard
    // would otherwise keep returning the same exit code and pass its own test.
    match plan_of(recipe) {
        Plan::WrongMachine => {
            fail!(
                "This fix only applies on: {}. Running OS is: {}.",
                recipe.os_scope().join(", "),
                current_os()
            );
            exit::NOT_APPLICABLE
        }
        Plan::Unfixable => {
            // Not an error, so not a nonzero code: the command did everything it
            // could, which is name the problem. NOT_APPLICABLE would say "not
            // applicable *here*", a claim that would send a caller looking for
            // another machine to run it on.
            println!("This problem has no reliable fix, so this command changes nothing.");
            println!("The explanation above is the whole of what is known about it.");
            exit::OK
        }
        Plan::Run => recipe
            .runner
            .expect("plan_of returns Run only when a runner is present")(opts),
        Plan::MissingRunner => {
            // A recipe the catalog says the CLI carries out, with nothing to
            // carry it out with -> INTERNAL, not FAILED (which is reserved for
            // "attempted, and the command failed"). Held by
            // `a_recipe_that_acts_has_a_runner`.
            fail!("Internal error: a recipe the catalog says is applied here has no runner.");
            exit::INTERNAL
        }
        Plan::PrintSteps => {
            println!("This fix is print-only (manual change required).");
            println!("Copy the commands above, run them yourself, then verify with:");
            for step in verify_steps(recipe) {
                println!("{step}");
            }
            exit::OK
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_is_linux;
    use catalog::{RECIPES, on};
    use render::verify_lines;

    /// Whether a recipe applies on the platform the test is running on.
    ///
    /// Gating on `runtime_is_linux()` stopped being the same question once WSL
    /// became its own family: a WSL host is Linux, but a Linux-scoped recipe is
    /// correctly refused there.
    fn recipe_applies_here(fix_id: &str) -> bool {
        find_recipe(fix_id).is_some_and(|r| r.platform(current_os()).is_some())
    }

    /// A recipe that exists only here, to reach a class the catalog has no
    /// member of.
    fn unfixable_recipe(class: FixClass) -> FixRecipe {
        FixRecipe {
            fix_id: "fix-0-not-in-the-catalog",
            title: "A problem that can be named but not repaired",
            rationale: "Synthetic. Exists to drive the class gate in `act_on`.",
            commands: &[],
            needs_sudo: false,
            needs_reboot: false,
            needs_relogin: false,
            verify: "",
            verify_linux: None,
            notes: &[],
            doc_url: None,
            applies_on: Box::leak(Box::new([on(current_os(), class)])),
            runner: None,
        }
    }

    /// `DIAGNOSE-ONLY` reports success, because naming the problem *is* the
    /// whole job.
    ///
    /// The distinction this pins is 0 against 3. Exit 3 is "not applicable on
    /// this machine", which would send a caller looking for a different machine
    /// to run the fix on; there is no such machine, because there is no fix.
    /// Driven through the synthetic recipe rather than through
    /// `info-20-gpu-no-rocm-support`, the catalog's one `DIAGNOSE-ONLY` entry,
    /// so the exit code stays pinned to the class and not to whichever entries
    /// happen to carry it.
    #[test]
    fn a_problem_with_no_reliable_fix_reports_success_and_does_not_look_portable() {
        let opts = FixOptions::default();
        assert_eq!(
            act_on(&unfixable_recipe(FixClass::DiagnoseOnly), &opts),
            0,
            "an unfixable problem is not an error and is not a wrong-machine result"
        );
        // The code alone does not pin the branch: print-only exits 0 as well, so
        // an assertion on the code would survive the unfixable arm being
        // disabled, and the user would be told to copy commands that do not
        // exist. This is the part that only holds while the arm is reached.
        assert_eq!(
            plan_of(&unfixable_recipe(FixClass::DiagnoseOnly)),
            Plan::Unfixable,
            "DIAGNOSE-ONLY must not fall through to the print-the-steps path"
        );
        for actionable in [FixClass::Auto, FixClass::NeedsArgument, FixClass::PrintOnly] {
            assert_ne!(
                plan_of(&unfixable_recipe(actionable)),
                Plan::Unfixable,
                "{actionable:?} has something to offer and must not be treated as unfixable"
            );
        }
    }

    /// The same recipe, reclassified, takes a different path — so the assertion
    /// above is about the class and not about the synthetic recipe's emptiness.
    ///
    /// Without this pairing, `act_on` could ignore the class entirely and both
    /// this test and the one above would still pass: a runner-less print-only
    /// recipe also returns 0. What separates them is the OS gate, which a
    /// recipe scoped to another platform fails.
    #[test]
    fn the_class_gate_is_reached_only_after_the_platform_gate() {
        let opts = FixOptions::default();
        let elsewhere = FixRecipe {
            applies_on: Box::leak(Box::new([on(
                if current_os() == "windows" {
                    "linux"
                } else {
                    "windows"
                },
                FixClass::DiagnoseOnly,
            )])),
            ..unfixable_recipe(FixClass::DiagnoseOnly)
        };
        assert_eq!(
            act_on(&elsewhere, &opts),
            3,
            "a recipe that does not apply here is a wrong-machine result, whatever its class"
        );
    }

    /// A Linux verify line is only worth having on a recipe that applies on
    /// Linux or WSL; on any other it could never be run.
    #[test]
    fn a_linux_verify_line_belongs_to_a_recipe_that_applies_on_linux_or_wsl() {
        let with_one: Vec<&FixRecipe> = RECIPES
            .iter()
            .filter(|r| r.verify_linux.is_some())
            .collect();
        assert!(
            !with_one.is_empty(),
            "no recipe carries a Linux verify line"
        );
        for r in with_one {
            assert!(
                r.platform("linux").is_some() || r.platform("wsl").is_some(),
                "{} has a Linux verify line but does not apply on Linux or WSL",
                r.fix_id
            );
        }
    }

    #[test]
    fn a_print_only_plan_labels_its_linux_verify_step() {
        let mut recipe = unfixable_recipe(FixClass::PrintOnly);
        recipe.verify = "hipcc --version";
        assert_eq!(verify_steps(&recipe), ["  $ hipcc --version"]);
        recipe.verify_linux = Some("rocminfo | head -n 5");
        assert_eq!(
            verify_steps(&recipe),
            [
                "  $ hipcc --version",
                "  $ rocminfo | head -n 5   # Linux, WSL"
            ]
        );
        recipe.verify = "";
        recipe.verify_linux = None;
        assert!(verify_steps(&recipe).is_empty() && verify_lines(&recipe).is_empty());
    }

    #[test]
    fn the_published_exit_codes_are_the_ones_apply_returns() {
        // Reachable without a machine in a particular state, so they are worth
        // pinning here: an unknown id and a recipe for another platform.
        let manifest = catalog_manifest();
        assert_eq!(
            apply("fix-does-not-exist", &FixOptions::default()),
            manifest.exit_codes["unknown_id"],
            "an unknown id must exit with the code the manifest publishes for it"
        );
        assert_eq!(
            apply(fix_id_for_another_platform(), &FixOptions::default()),
            manifest.exit_codes["not_applicable"],
            "a recipe for another platform must exit with the published code"
        );
        // The map is a contract in itself: dropping a name would leave a reader
        // unable to interpret a code it still receives.
        for name in [
            "ok",
            "internal",
            "unknown_id",
            "not_applicable",
            "failed",
            "declined",
        ] {
            assert!(
                manifest.exit_codes.contains_key(name),
                "the published exit-code map is missing `{name}`"
            );
        }
    }

    /// A catalog entry that cannot apply on the platform running the test.
    fn fix_id_for_another_platform() -> &'static str {
        RECIPES
            .iter()
            .find(|r| r.platform(current_os()).is_none())
            .map(|r| r.fix_id)
            .expect("every platform has an entry belonging to a different one")
    }

    #[test]
    fn unknown_fix_id_returns_2() {
        let code = apply("fix-does-not-exist", &FixOptions::default());
        assert_eq!(code, 2);
    }

    #[test]
    fn dry_run_never_mutates_and_returns_zero_for_auto_linux_fix() {
        if !recipe_applies_here("fix-2-unset-override") {
            return;
        }
        // fix-2 unset-override is print-only on linux (no mutation regardless);
        // a dry-run must report success without changing anything.
        let opts = FixOptions {
            dry_run: true,
            ..FixOptions::default()
        };
        let code = apply("fix-2-unset-override", &opts);
        assert_eq!(code, 0);
    }

    #[test]
    fn fix_9_without_device_index_is_print_only_and_returns_zero() {
        if !recipe_applies_here("fix-9-igpu-dgpu") {
            // Refused at the OS gate here (WSL is its own family), which is a
            // different assertion -- covered by the inapplicable-recipe test.
            return;
        }
        // Regression: the missing `--device-index` branch only prints the
        // query that identifies the dGPU, so it is a print-only preview and
        // must return 0 -- not the environment/OS code 3. A dry-run without the
        // argument must likewise succeed, since the runner never mutates.
        for dry_run in [false, true] {
            let opts = FixOptions {
                dry_run,
                ..FixOptions::default()
            };
            let code = apply("fix-9-igpu-dgpu", &opts);
            assert_eq!(
                code, 0,
                "fix-9 without --device-index (dry_run={dry_run}) must be a print-only success"
            );
        }
    }

    #[test]
    fn print_only_fix_returns_zero() {
        // Pick a recipe that applies on THIS platform rather than naming a Linux
        // one: the assertion is about print-only recipes succeeding, and hunting
        // for an applicable one keeps it meaningful on every lane.
        let fix_id = RECIPES
            .iter()
            .find(|r| {
                r.platform(current_os())
                    .is_some_and(|p| p.class == FixClass::PrintOnly)
            })
            .map(|r| r.fix_id)
            .expect("every supported platform has at least one print-only recipe");
        let code = apply(fix_id, &FixOptions::default());
        assert_eq!(code, 0, "{fix_id} is print-only here and must succeed");
    }

    #[test]
    fn windows_only_fix_refused_on_linux() {
        if !runtime_is_linux() {
            return;
        }
        let code = apply("fix-13-hip-sdk-missing", &FixOptions::default());
        assert_eq!(code, 3);
    }

    #[test]
    fn a_position_argument_is_refused_with_the_same_exit_code() {
        // Still 2: this is clearer wording on an existing refusal, not a new
        // behaviour that a script could start depending on.
        assert_eq!(apply("#1", &FixOptions::default()), 2);
        assert_eq!(apply("bogus", &FixOptions::default()), 2);

        // ROCMAI-543: the two lines `apply` writes to stderr on this path must
        // be plain ASCII, or a Windows console on the OEM code page renders the
        // punctuation as mojibake like `ΓÇö`. The exit code above says nothing
        // about them, and `fail!` writes past any in-process capture.
        let refusal = diagnosis_position_refusal("#1");
        for line in &refusal {
            assert!(
                line.is_ascii(),
                "the position-argument refusal must be ASCII-only (ROCMAI-543): {line}"
            );
        }
        // Both halves still have to do their job: name what the user typed, and
        // name where the real id is printed.
        assert!(refusal[0].contains("#1"), "refusal: {refusal:?}");
        assert!(
            refusal[1].contains("id:") && refusal[1].contains("apply with:"),
            "the refusal must keep pointing at the `id:`/`apply with:` lines \
             `rocm diagnose` prints: {refusal:?}"
        );
    }

    /// What [`scan_exit_code_literals`] found in a source file.
    #[derive(Default)]
    struct ExitCodeScan {
        /// The `-> i32` functions it inspected, so a test can notice when a
        /// reorder or rename leaves it looking at fewer of them than it should.
        checked: Vec<String>,
        /// `function: literal` for each integer literal reported as a result.
        bare: Vec<String>,
    }

    /// Whether a `cfg` predicate holds only in test builds: `test`, or an
    /// `all(..)` with such a member, or an `any(..)` of which every member is
    /// such. `not(..)` and every other predicate are not.
    fn cfg_requires_test(meta: &syn::Meta) -> bool {
        match meta {
            syn::Meta::Path(path) => path.is_ident("test"),
            syn::Meta::List(list) => {
                let Ok(members) = syn::parse::Parser::parse2(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                    list.tokens.clone(),
                ) else {
                    return false;
                };
                if list.path.is_ident("all") {
                    members.iter().any(cfg_requires_test)
                } else if list.path.is_ident("any") {
                    !members.is_empty() && members.iter().all(cfg_requires_test)
                } else {
                    false
                }
            }
            syn::Meta::NameValue(_) => false,
        }
    }

    /// Whether `attrs` carry a `#[cfg(..)]` that holds only in test builds, such
    /// as `#[cfg(test)]` or `#[cfg(all(test, windows))]`.
    fn is_cfg_test(attrs: &[syn::Attribute]) -> bool {
        attrs.iter().any(|attr| {
            attr.path().is_ident("cfg")
                && attr
                    .parse_args::<syn::Meta>()
                    .is_ok_and(|meta| cfg_requires_test(&meta))
        })
    }

    /// The text of `expr` when it is an integer literal in any spelling
    /// (`4`, `4_i32`, `0x4`, `-1`).
    fn int_literal(expr: &syn::Expr) -> Option<String> {
        match expr {
            syn::Expr::Lit(lit) => match &lit.lit {
                syn::Lit::Int(int) => Some(int.to_string()),
                _ => None,
            },
            syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
                int_literal(&unary.expr).map(|text| format!("-{text}"))
            }
            syn::Expr::Paren(paren) => int_literal(&paren.expr),
            _ => None,
        }
    }

    /// Whether `expr` can evaluate to an integer literal: it is one, or an `if`,
    /// `match` or block with a literal result in some branch.
    fn yields_literal(expr: &syn::Expr) -> bool {
        let mut branches = FunctionResults::default();
        branches.result(expr);
        !branches.literals.is_empty()
    }

    fn pattern_name(pat: &syn::Pat) -> Option<String> {
        match pat {
            syn::Pat::Ident(ident) => Some(ident.ident.to_string()),
            syn::Pat::Type(typed) => pattern_name(&typed.pat),
            _ => None,
        }
    }

    /// The results of one function body: the integer literals it returns, and
    /// the names it returns that were set to an integer literal.
    #[derive(Default)]
    struct FunctionResults {
        literals: Vec<String>,
        returned_names: Vec<String>,
        literal_locals: std::collections::HashSet<String>,
    }

    impl FunctionResults {
        /// Record `expr`, a value the function hands back: look through the
        /// branches of an `if` or `match` and the block of a tail expression.
        fn result(&mut self, expr: &syn::Expr) {
            if let Some(text) = int_literal(expr) {
                self.literals.push(text);
                return;
            }
            match expr {
                syn::Expr::Path(path) => {
                    if let Some(name) = path.path.get_ident() {
                        self.returned_names.push(name.to_string());
                    }
                }
                syn::Expr::Paren(paren) => self.result(&paren.expr),
                syn::Expr::Block(block) => self.tail(&block.block),
                syn::Expr::Unsafe(block) => self.tail(&block.block),
                syn::Expr::If(branch) => {
                    self.tail(&branch.then_branch);
                    if let Some((_, otherwise)) = &branch.else_branch {
                        self.result(otherwise);
                    }
                }
                syn::Expr::Match(matched) => {
                    for arm in &matched.arms {
                        self.result(&arm.body);
                    }
                }
                _ => {}
            }
        }

        fn tail(&mut self, block: &syn::Block) {
            if let Some(syn::Stmt::Expr(expr, None)) = block.stmts.last() {
                self.result(expr);
            }
        }
    }

    impl<'ast> syn::visit::Visit<'ast> for FunctionResults {
        fn visit_expr_return(&mut self, ret: &'ast syn::ExprReturn) {
            if let Some(expr) = &ret.expr {
                self.result(expr);
            }
            syn::visit::visit_expr_return(self, ret);
        }

        fn visit_local(&mut self, local: &'ast syn::Local) {
            if let (Some(name), Some(init)) = (pattern_name(&local.pat), &local.init)
                && yields_literal(&init.expr)
            {
                self.literal_locals.insert(name);
            }
            syn::visit::visit_local(self, local);
        }

        fn visit_expr_assign(&mut self, assign: &'ast syn::ExprAssign) {
            if let syn::Expr::Path(target) = &*assign.left
                && let Some(name) = target.path.get_ident()
                && yields_literal(&assign.right)
            {
                self.literal_locals.insert(name.to_string());
            }
            syn::visit::visit_expr_assign(self, assign);
        }

        // A `return` inside a closure or a nested function is that function's
        // result, not this one's.
        fn visit_expr_closure(&mut self, _: &'ast syn::ExprClosure) {}

        fn visit_item_fn(&mut self, _: &'ast syn::ItemFn) {}
    }

    fn scan_function(sig: &syn::Signature, body: &syn::Block, scan: &mut ExitCodeScan) {
        let returns_i32 = matches!(
            &sig.output,
            syn::ReturnType::Type(_, ty)
                if matches!(&**ty, syn::Type::Path(path) if path.path.is_ident("i32"))
        );
        if !returns_i32 {
            return;
        }
        let name = sig.ident.to_string();
        let mut results = FunctionResults::default();
        syn::visit::Visit::visit_block(&mut results, body);
        results.tail(body);
        for literal in &results.literals {
            scan.bare.push(format!("{name}: {literal}"));
        }
        for local in &results.returned_names {
            if results.literal_locals.contains(local) {
                scan.bare
                    .push(format!("{name}: `{local}` is set to an integer literal"));
            }
        }
        scan.checked.push(name);
    }

    fn scan_items(items: &[syn::Item], scan: &mut ExitCodeScan) {
        for item in items {
            match item {
                syn::Item::Fn(function) if !is_cfg_test(&function.attrs) => {
                    scan_function(&function.sig, &function.block, scan);
                }
                syn::Item::Impl(imp) if !is_cfg_test(&imp.attrs) => {
                    for item in &imp.items {
                        if let syn::ImplItem::Fn(method) = item {
                            scan_function(&method.sig, &method.block, scan);
                        }
                    }
                }
                syn::Item::Mod(module) if !is_cfg_test(&module.attrs) => {
                    if let Some((_, items)) = &module.content {
                        scan_items(items, scan);
                    }
                }
                _ => {}
            }
        }
    }

    /// Find the integer literals that a free or `impl` function returning `i32`
    /// reports as its result, in `source`, wherever in the file it sits.
    ///
    /// It parses the source, so the spelling of the literal does not matter
    /// (`4i32`, `4_i32`, `0x4`, `-1`), nor does the layout (a one-line `match`,
    /// a `return 3` with no semicolon, a comment). A literal counts when it is
    /// the operand of a `return`, the function's tail expression, an arm of a
    /// tail `if` or `match`, or a local that was set to such a value and is then
    /// returned by name (tracked by name, without scoping, so an unrelated local
    /// of the same name can be flagged: a spurious failure, never a miss).
    ///
    /// What it does not see: trait default methods, a function nested in a
    /// function body, a return type spelled through an alias, a literal reaching
    /// the result through a cast or arithmetic, a helper that returns it, or a
    /// value read from a struct. So this is a net under the `exit::` constants,
    /// not a proof that no runner can return a number of its own.
    fn scan_exit_code_literals(source: &str) -> ExitCodeScan {
        let file = syn::parse_file(source).expect("the source parses");
        let mut scan = ExitCodeScan::default();
        scan_items(&file.items, &mut scan);
        scan
    }

    #[test]
    fn the_bare_exit_code_lint_flags_each_way_of_writing_one() {
        let scan = scan_exit_code_literals(
            r"
            fn tail() -> i32 { 0 }
            fn returned() -> i32 { return 3; }
            fn no_semicolon(x: bool) -> i32 { if x { return 3 } exit::OK }
            fn branches(x: bool) -> i32 { if x { 5 } else { exit::OK } }
            fn one_line_match(r: Result<(), ()>) -> i32 { match r { Ok(_) => 0, Err(_) => 4 } }
            fn suffixed() -> i32 { return 4i32; }
            fn underscored() -> i32 { return 4_i32; }
            fn hex() -> i32 { return 0x4; }
            fn negative() -> i32 { return -1; }
            fn via_local() -> i32 { let rc = 4; rc }
            fn via_typed_local() -> i32 { let rc: i32 = 4; rc }
            fn via_assignment() -> i32 { let mut rc = exit::OK; rc = 4; rc }
            fn via_if_local(c: bool) -> i32 { let code = if c { 3 } else { exit::OK }; return code; }
            fn via_match_local(rc: i32) -> i32 { let code = match rc { 1 => exit::OK, _ => 4 }; code }
            fn in_else(c: bool) -> i32 { if c { exit::OK } else { 4 } }
            fn in_block_arm(r: Result<(), ()>) -> i32 { match r { Ok(_) => exit::OK, Err(_) => { 4 } } }
            fn in_unsafe() -> i32 { unsafe { 4 } }
            fn parenthesised() -> i32 { (4) }
            fn returned_parenthesised() -> i32 { return (4); }
            fn parenthesised_branch(c: bool) -> i32 { (if c { 4 } else { exit::OK }) }
            struct Runner;
            impl Runner { fn method(&self) -> i32 { 3 } }
            mod inner { pub fn nested() -> i32 { 5 } }
            #[cfg(not(test))]
            fn not_test() -> i32 { 3 }
            #[cfg(any(test, windows))]
            fn maybe_test() -> i32 { 3 }
            ",
        );
        assert_eq!(
            scan.bare,
            [
                "tail: 0",
                "returned: 3",
                "no_semicolon: 3",
                "branches: 5",
                "one_line_match: 0",
                "one_line_match: 4",
                "suffixed: 4i32",
                "underscored: 4_i32",
                "hex: 0x4",
                "negative: -1",
                "via_local: `rc` is set to an integer literal",
                "via_typed_local: `rc` is set to an integer literal",
                "via_assignment: `rc` is set to an integer literal",
                "via_if_local: `code` is set to an integer literal",
                "via_match_local: `code` is set to an integer literal",
                "in_else: 4",
                "in_block_arm: 4",
                "in_unsafe: 4",
                "parenthesised: 4",
                "returned_parenthesised: 4",
                "parenthesised_branch: 4",
                "method: 3",
                "nested: 5",
                "not_test: 3",
                "maybe_test: 3",
            ]
        );
    }

    #[test]
    fn the_bare_exit_code_lint_leaves_named_constants_and_unrelated_integers_alone() {
        let scan = scan_exit_code_literals(
            r"
            fn named(x: bool) -> i32 { if x { return exit::FAILED; } exit::OK }
            fn compares(rc: i32) -> i32 { if rc != 0 { return exit::FAILED; } exit::OK }
            fn other_integers() -> i32 {
                let width = 80;
                let _ = width;
                for _ in 0..3 {}
                exit::OK
            }
            fn commented() -> i32 {
                // return 5; is what this used to say
                exit::OK
            }
            fn closure() -> i32 {
                let f = |x: i32| { return 7; };
                f(1);
                exit::OK
            }
            fn not_a_runner() -> usize { 3 }
            fn nested() -> i32 { fn inner() -> i32 { 7 } exit::OK }
            #[cfg(test)]
            mod tests { fn in_tests() -> i32 { 3 } }
            #[cfg(all(test, windows))]
            mod windows_tests { fn in_tests() -> i32 { 3 } }
            #[cfg(test)]
            fn test_only() -> i32 { 3 }
            #[cfg(test)]
            struct TestRunner;
            #[cfg(test)]
            impl TestRunner { fn method(&self) -> i32 { 3 } }
            ",
        );
        assert_eq!(scan.bare, Vec::<String>::new());
        assert_eq!(
            scan.checked,
            [
                "named",
                "compares",
                "other_integers",
                "commented",
                "closure",
                "nested"
            ]
        );
    }

    /// Every runner reports its result through `exit::`, so the published
    /// manifest and the code that produces the codes are meant not to come to
    /// disagree. [`scan_exit_code_literals`] says what the check can and cannot
    /// see; `rc != 0` on a child's own status is not a runner result and is not
    /// matched.
    #[test]
    fn fix_runners_return_named_exit_constants_not_integers() {
        let mut scan = scan_exit_code_literals(include_str!("fix.rs"));
        // The module root and its children are one body of code, so the scan
        // covers `fix.rs` and every file in `fix/`. The list below is held to
        // the directory, so a child added without being listed fails the test
        // rather than silently going unscanned.
        let children: [(&str, &str); 6] = [
            ("catalog.rs", include_str!("fix/catalog.rs")),
            ("consent.rs", include_str!("fix/consent.rs")),
            ("manifest.rs", include_str!("fix/manifest.rs")),
            ("options.rs", include_str!("fix/options.rs")),
            ("render.rs", include_str!("fix/render.rs")),
            ("runners.rs", include_str!("fix/runners.rs")),
        ];
        let mut on_disk: Vec<String> =
            std::fs::read_dir(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/fix"))
                .expect("the fix/ directory is readable")
                .map(|entry| {
                    entry
                        .expect("a directory entry is readable")
                        .file_name()
                        .to_string_lossy()
                        .into_owned()
                })
                .filter(|name| {
                    std::path::Path::new(name)
                        .extension()
                        .is_some_and(|ext| ext == "rs")
                })
                .collect();
        on_disk.sort();
        let listed: Vec<&str> = children.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            on_disk, listed,
            "every file in fix/ must be listed here, or the exit-code lint skips it"
        );
        for (_, source) in children {
            let more = scan_exit_code_literals(source);
            scan.checked.extend(more.checked);
            scan.bare.extend(more.bare);
        }
        // A reorder or a rename must not leave the scan looking at fewer
        // runners than it should without anyone noticing.
        for runner in [
            "apply",
            "act_on",
            "run_render_group",
            "run_unset_override",
            "run_unset_override_linux",
            "report_persistent_override",
            "run_unset_override_windows",
            "report_and_clear_override_windows",
            "run_path_export",
            "run_path_export_linux",
            "run_path_export_windows",
            "path_export_windows",
            "run_hip_visible_devices",
            "run_hip_visible_devices_linux",
            "pin_device_in_rc_file",
            "run_hip_visible_devices_windows",
        ] {
            assert!(
                scan.checked.iter().any(|name| name == runner),
                "the scan no longer inspects `{runner}`; checked: {:?}",
                scan.checked
            );
        }
        assert_eq!(
            scan.bare,
            Vec::<String>::new(),
            "use exit::OK, exit::NOT_APPLICABLE, exit::FAILED or exit::DECLINED"
        );
    }
}
