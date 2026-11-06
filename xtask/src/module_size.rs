// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Guard the size of every source file: no file may exceed
//! [`MAX_PRODUCTION_LINES`] production lines.
//!
//! There is one limit, for every file. There are no per-file ceilings, baseline,
//! bless flag or allowlist, and no comment or attribute that opts a file out, and
//! none should be added: a PR that would push a file past the limit splits it in
//! the same change, and changing the limit is a reviewed edit to the constant
//! below.
//!
//! # What counts as a source file
//!
//! Every `.rs` file reachable through `mod` declarations from the root of a
//! library, binary, proc-macro or build-script target of a workspace package
//! (the roots come from `cargo metadata`), except:
//!
//! * a file with a `tests`, `benches` or `examples` directory anywhere in its
//!   path (this includes the end-to-end harness crate under `tests/`);
//! * a file reached only through a `mod` declaration gated on a test-only cfg
//!   (`#[cfg(test)] mod x;`, `#[cfg(all(test, unix))] mod x;`, with or without a
//!   `#[path]`), or whose own first line is `#![cfg(test)]`. Proptest and
//!   property-test files are excluded this way, not by name.
//!
//! An external `#[cfg(test)] mod tests;` therefore never charges its lines to the
//! parent: the child is not a source file, and the parent loses only the lines of
//! the declaration. A `.rs` file under a package's `src/` that no `mod`
//! declaration reaches is an error rather than a file that escapes the count.
//!
//! # What counts as a production line
//!
//! A physical line of a source file (blank lines and comments included) that is
//! not covered by a test-gated item. An item is test-gated when it carries
//! `#[test]` (or any attribute path ending in `test`, such as `#[tokio::test]`)
//! or a `#[cfg(..)]` that holds only in test builds: `test`, `all(..)` with a
//! test-only term (`all(test, unix)`), or `any(..)` whose every term is
//! test-only. `not(..)`, `any(test, unix)` and `cfg_attr` are not test-gated.
//! Gated items are modules, functions, impls, uses and the rest, and the gated
//! members of an impl or trait. An item's lines run from its first outer
//! attribute (doc comments included) to its closing token.
//!
//! Counting is exact because it parses each file with `syn` instead of scanning
//! for braces, so braces in string literals, raw strings, comments, and
//! `cfg(all(test, ..))` spellings cannot confuse it. It does not see inside macro
//! bodies or function bodies, so a `#[cfg(test)]` item nested in either counts as
//! production. A file `syn` cannot parse fails the check; it is never skipped.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use syn::ext::IdentExt;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{Attribute, Expr, ExprLit, ImplItem, Item, Lit, Meta, Token, TraitItem};

use crate::paths;

/// The production-line limit that applies to every source file. A file of exactly
/// this many lines passes. 1,500 keeps a file readable in one sitting and leaves
/// room to add the tests every fix requires, which the count excludes.
pub const MAX_PRODUCTION_LINES: usize = 1_500;

/// Directory names that mark a file as test-only, whatever else is in its path.
const EXCLUDED_DIRS: &[&str] = &["tests", "benches", "examples"];

/// Cargo target kinds whose root file starts a module tree that ships.
const ROOT_KINDS: &[&str] = &[
    "lib",
    "rlib",
    "dylib",
    "cdylib",
    "staticlib",
    "proc-macro",
    "bin",
    "custom-build",
];

/// A file over the limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Offender {
    path: String,
    lines: usize,
}

/// What a run measured.
#[derive(Debug)]
pub(crate) struct Outcome {
    checked: usize,
    offenders: Vec<Offender>,
}

/// `cargo xtask check-module-size`: fail, naming each file and its count, if any
/// source file is over [`MAX_PRODUCTION_LINES`].
pub fn run() -> Result<()> {
    let outcome = evaluate()?;
    if outcome.offenders.is_empty() {
        println!("{}", passed_text(&outcome));
        return Ok(());
    }
    bail!("{}", failed_text(&outcome))
}

/// `cargo xtask check-module-size-advisory`: the same check, but it only reports.
/// This is the wrapper the local hook runs while the limit is not yet blocking; it
/// cannot fail a commit, including when the check itself cannot run.
pub fn run_advisory() -> Result<()> {
    advisory_to(evaluate(), &mut std::io::stdout())
}

fn advisory_to(result: Result<Outcome>, out: &mut impl Write) -> Result<()> {
    let text = match result {
        Ok(outcome) if outcome.offenders.is_empty() => passed_text(&outcome),
        Ok(outcome) => format!(
            "warning: not blocking yet, but this will fail once the limit is enforced\n{}",
            failed_text(&outcome)
        ),
        Err(error) => format!("warning: the module-size check could not run: {error:#}"),
    };
    writeln!(out, "{text}").context("failed to write the module-size report")?;
    Ok(())
}

fn passed_text(outcome: &Outcome) -> String {
    format!(
        "module-size: {} source files checked, none over {MAX_PRODUCTION_LINES} production lines",
        outcome.checked
    )
}

fn failed_text(outcome: &Outcome) -> String {
    let mut text = format!(
        "{} source file(s) over the limit of {MAX_PRODUCTION_LINES} production lines (of {} checked):",
        outcome.offenders.len(),
        outcome.checked
    );
    for offender in &outcome.offenders {
        let _ = write!(
            text,
            "\n  {}: {} production lines ({} over)",
            offender.path,
            offender.lines,
            offender.lines - MAX_PRODUCTION_LINES
        );
    }
    text
}

fn evaluate() -> Result<Outcome> {
    let stdout = paths::run_cargo_metadata(&["--no-deps"])?;
    let metadata: Metadata =
        serde_json::from_slice(&stdout).context("failed to parse `cargo metadata` output")?;
    let packages: Vec<Package> = metadata.packages.iter().map(Package::from).collect();
    let files = collect(&metadata.workspace_root, &packages)?;
    Ok(Outcome {
        checked: files.len(),
        offenders: offenders(&files, MAX_PRODUCTION_LINES),
    })
}

/// Files strictly over `limit`, largest first.
fn offenders(files: &[(String, usize)], limit: usize) -> Vec<Offender> {
    let mut over: Vec<Offender> = files
        .iter()
        .filter(|(_, lines)| *lines > limit)
        .map(|(path, lines)| Offender {
            path: path.clone(),
            lines: *lines,
        })
        .collect();
    over.sort_by(|a, b| b.lines.cmp(&a.lines).then_with(|| a.path.cmp(&b.path)));
    over
}

#[derive(Deserialize)]
struct Metadata {
    workspace_root: PathBuf,
    packages: Vec<MetaPackage>,
}

#[derive(Deserialize)]
struct MetaPackage {
    manifest_path: PathBuf,
    targets: Vec<MetaTarget>,
}

#[derive(Deserialize)]
struct MetaTarget {
    kind: Vec<String>,
    src_path: PathBuf,
}

/// A workspace package: where its sources live and the files its module trees
/// start from.
struct Package {
    src_dir: PathBuf,
    roots: Vec<PathBuf>,
}

impl From<&MetaPackage> for Package {
    fn from(package: &MetaPackage) -> Self {
        let dir = package
            .manifest_path
            .parent()
            .unwrap_or_else(|| Path::new(""));
        Self {
            src_dir: dir.join("src"),
            roots: package
                .targets
                .iter()
                .filter(|target| target.kind.iter().any(|k| ROOT_KINDS.contains(&k.as_str())))
                .map(|target| target.src_path.clone())
                .collect(),
        }
    }
}

/// Production line count of every source file in `packages`, keyed by its path
/// relative to `root` with `/` separators.
fn collect(root: &Path, packages: &[Package]) -> Result<Vec<(String, usize)>> {
    let mut walk = Walk {
        root: normalize(root),
        counted: Vec::new(),
        reached: BTreeSet::new(),
    };
    for package in packages {
        for target_root in &package.roots {
            walk.visit(target_root, true, false)?;
        }
    }

    let mut stray = Vec::new();
    for package in packages {
        for file in rust_files_under(&package.src_dir)? {
            let file = normalize(&file);
            if !walk.reached.contains(&file) && !is_excluded(&walk.relative(&file)) {
                stray.push(walk.relative(&file));
            }
        }
    }
    if !stray.is_empty() {
        stray.sort();
        bail!(
            "source file(s) that no `mod` declaration reaches, so they would escape the size count \
             (declare them, or delete them):\n  {}",
            stray.join("\n  ")
        );
    }

    walk.counted.sort();
    Ok(walk.counted)
}

struct Walk {
    root: PathBuf,
    counted: Vec<(String, usize)>,
    reached: BTreeSet<PathBuf>,
}

impl Walk {
    fn relative(&self, file: &Path) -> String {
        let relative = file.strip_prefix(&self.root).unwrap_or(file);
        relative.to_string_lossy().replace('\\', "/")
    }

    /// Visit `file`: count it unless it is test-only, then follow its external
    /// `mod` declarations. `mod_rs_like` is true for a target root, a `mod.rs`, and
    /// a file loaded by `#[path]`, whose child modules sit beside it rather than in
    /// a directory named after it. Everything below a gated file is gated.
    fn visit(&mut self, file: &Path, mod_rs_like: bool, gated: bool) -> Result<()> {
        let file = normalize(file);
        if !self.reached.insert(file.clone()) {
            return Ok(());
        }
        let shown = self.relative(&file);
        let source =
            fs::read_to_string(&file).with_context(|| format!("failed to read {shown}"))?;
        let analysis = analyze(&source).with_context(|| format!("failed to parse {shown}"))?;

        let gated = gated || analysis.gated_file;
        if !gated && !is_excluded(&shown) {
            let total = source.lines().count();
            self.counted
                .push((shown.clone(), production_lines(total, &analysis)));
        }

        for child in &analysis.children {
            let path =
                resolve_child(&file, mod_rs_like, child).with_context(|| format!("in {shown}"))?;
            let child_mod_rs_like =
                child.path_attr.is_some() || path.file_name().is_some_and(|n| n == "mod.rs");
            self.visit(&path, child_mod_rs_like, gated || child.gated)?;
        }
        Ok(())
    }
}

/// Whether a root-relative `/` path is under a test-only directory.
fn is_excluded(relative: &str) -> bool {
    relative
        .split('/')
        .any(|component| EXCLUDED_DIRS.contains(&component))
}

/// The file a `mod name;` declaration refers to.
fn resolve_child(parent: &Path, parent_mod_rs_like: bool, child: &Child) -> Result<PathBuf> {
    let dir = parent.parent().unwrap_or_else(|| Path::new(""));
    let mut base = if parent_mod_rs_like {
        dir.to_path_buf()
    } else {
        dir.join(parent.file_stem().unwrap_or_default())
    };
    for name in &child.inline {
        base.push(name);
    }

    if let Some(attr) = &child.path_attr {
        let at = if child.inline.is_empty() { dir } else { &base };
        return Ok(normalize(&at.join(attr)));
    }

    let flat = base.join(format!("{}.rs", child.name));
    let nested = base.join(&child.name).join("mod.rs");
    match (flat.is_file(), nested.is_file()) {
        (true, false) => Ok(flat),
        (false, true) => Ok(nested),
        (true, true) => bail!(
            "`mod {}` is ambiguous: both {} and {} exist",
            child.name,
            flat.display(),
            nested.display()
        ),
        (false, false) => bail!(
            "`mod {};` has no file: looked for {} and {}",
            child.name,
            flat.display(),
            nested.display()
        ),
    }
}

/// Resolve `.` and `..` lexically, so `a/b/../c` and `a/c` name one file.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn rust_files_under(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    if !dir.is_dir() {
        return Ok(found);
    }
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in
            fs::read_dir(&next).with_context(|| format!("failed to read {}", next.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }
    Ok(found)
}

/// What parsing one file yields.
#[derive(Default)]
struct Analysis {
    /// The file opens with `#![cfg(test)]` (or another test-only inner cfg).
    gated_file: bool,
    /// 1-based inclusive line ranges covered by test-gated items.
    test_ranges: Vec<(usize, usize)>,
    /// External `mod x;` declarations, in source order.
    children: Vec<Child>,
}

struct Child {
    name: String,
    path_attr: Option<String>,
    /// Names of the inline `mod a { .. }` blocks the declaration sits in.
    inline: Vec<String>,
    gated: bool,
}

fn analyze(source: &str) -> Result<Analysis> {
    let file =
        syn::parse_file(source).map_err(|e| anyhow!("{e} (line {})", e.span().start().line))?;
    let mut analysis = Analysis {
        gated_file: attrs_are_test_gated(&file.attrs)?,
        ..Analysis::default()
    };
    walk_items(&file.items, &mut Vec::new(), false, &mut analysis)?;
    Ok(analysis)
}

/// Collect the gated line ranges and the external module declarations under
/// `items`. `gated` is true inside a gated item: it is already covered by that
/// item's range, but a `mod x;` inside it still has to be recorded as gated so
/// that its file is known to be reached.
fn walk_items(
    items: &[Item],
    inline: &mut Vec<String>,
    gated: bool,
    analysis: &mut Analysis,
) -> Result<()> {
    for item in items {
        let item_gated = gated || attrs_are_test_gated(item_attrs(item))?;
        if item_gated && !gated {
            analysis.test_ranges.push(lines_of(item));
        }
        match item {
            Item::Mod(module) => match &module.content {
                Some((_, inner)) => {
                    inline.push(module.ident.unraw().to_string());
                    walk_items(inner, inline, item_gated, analysis)?;
                    inline.pop();
                }
                None => analysis.children.push(Child {
                    name: module.ident.unraw().to_string(),
                    path_attr: path_attribute(&module.attrs)?,
                    inline: inline.clone(),
                    gated: item_gated,
                }),
            },
            Item::Impl(imp) if !item_gated => {
                for member in &imp.items {
                    if attrs_are_test_gated(impl_item_attrs(member))? {
                        analysis.test_ranges.push(lines_of(member));
                    }
                }
            }
            Item::Trait(tr) if !item_gated => {
                for member in &tr.items {
                    if attrs_are_test_gated(trait_item_attrs(member))? {
                        analysis.test_ranges.push(lines_of(member));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn lines_of(node: &impl Spanned) -> (usize, usize) {
    let span = node.span();
    (span.start().line, span.end().line)
}

/// Lines of the file not covered by a gated range. Two gated items can share a
/// line, so ranges are merged before they are subtracted.
fn production_lines(total: usize, analysis: &Analysis) -> usize {
    if analysis.gated_file {
        return 0;
    }
    let mut ranges = analysis.test_ranges.clone();
    ranges.sort_unstable();
    let mut covered = 0;
    let mut end_so_far = 0;
    for (start, end) in ranges {
        let end = end.min(total);
        let start = start.max(end_so_far + 1);
        if start <= end {
            covered += end - start + 1;
            end_so_far = end;
        }
    }
    total - covered
}

/// Production lines of one file's source text.
#[cfg(test)]
fn production_lines_of(source: &str) -> Result<usize> {
    Ok(production_lines(source.lines().count(), &analyze(source)?))
}

fn path_attribute(attrs: &[Attribute]) -> Result<Option<String>> {
    for attr in attrs {
        if !attr.path().is_ident("path") {
            continue;
        }
        if let Meta::NameValue(name_value) = &attr.meta
            && let Expr::Lit(ExprLit {
                lit: Lit::Str(path),
                ..
            }) = &name_value.value
        {
            return Ok(Some(path.value()));
        }
        bail!("cannot read a `#[path]` attribute that is not `path = \"..\"`");
    }
    Ok(None)
}

/// Whether any attribute makes an item exist only in test builds.
fn attrs_are_test_gated(attrs: &[Attribute]) -> Result<bool> {
    for attr in attrs {
        let path = attr.path();
        if path.is_ident("cfg") {
            let predicate: Meta = attr
                .parse_args()
                .map_err(|e| anyhow!("cannot parse a `#[cfg(..)]` predicate: {e}"))?;
            if cfg_is_test_only(&predicate)? {
                return Ok(true);
            }
        } else if path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "test")
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether a cfg predicate can only hold in a test build.
fn cfg_is_test_only(predicate: &Meta) -> Result<bool> {
    let Meta::List(list) = predicate else {
        return Ok(matches!(predicate, Meta::Path(path) if path.is_ident("test")));
    };
    let combinator = if list.path.is_ident("all") {
        true
    } else if list.path.is_ident("any") {
        false
    } else {
        return Ok(false);
    };
    let terms = list
        .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
        .map_err(|e| anyhow!("cannot parse a cfg combinator's terms: {e}"))?;
    if combinator {
        for term in &terms {
            if cfg_is_test_only(term)? {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    for term in &terms {
        if !cfg_is_test_only(term)? {
            return Ok(false);
        }
    }
    Ok(!terms.is_empty())
}

fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(i) => &i.attrs,
        Item::Enum(i) => &i.attrs,
        Item::ExternCrate(i) => &i.attrs,
        Item::Fn(i) => &i.attrs,
        Item::ForeignMod(i) => &i.attrs,
        Item::Impl(i) => &i.attrs,
        Item::Macro(i) => &i.attrs,
        Item::Mod(i) => &i.attrs,
        Item::Static(i) => &i.attrs,
        Item::Struct(i) => &i.attrs,
        Item::Trait(i) => &i.attrs,
        Item::TraitAlias(i) => &i.attrs,
        Item::Type(i) => &i.attrs,
        Item::Union(i) => &i.attrs,
        Item::Use(i) => &i.attrs,
        _ => &[],
    }
}

fn impl_item_attrs(item: &ImplItem) -> &[Attribute] {
    match item {
        ImplItem::Const(i) => &i.attrs,
        ImplItem::Fn(i) => &i.attrs,
        ImplItem::Type(i) => &i.attrs,
        ImplItem::Macro(i) => &i.attrs,
        _ => &[],
    }
}

fn trait_item_attrs(item: &TraitItem) -> &[Attribute] {
    match item {
        TraitItem::Const(i) => &i.attrs,
        TraitItem::Fn(i) => &i.attrs,
        TraitItem::Type(i) => &i.attrs,
        TraitItem::Macro(i) => &i.attrs,
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` one-line production items.
    fn filler(n: usize) -> String {
        (0..n).fold(String::new(), |mut text, i| {
            let _ = writeln!(text, "fn f{i}() {{}}");
            text
        })
    }

    fn write(root: &Path, relative: &str, text: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("a file has a parent")).expect("create dirs");
        fs::write(path, text).expect("write fixture");
    }

    /// One package whose only target root is `src/lib.rs`.
    fn lib_package(root: &Path) -> Vec<Package> {
        vec![Package {
            src_dir: root.join("src"),
            roots: vec![root.join("src/lib.rs")],
        }]
    }

    fn counted(root: &Path) -> Result<Vec<(String, usize)>> {
        collect(root, &lib_package(root))
    }

    // --- counting rules -----------------------------------------------------

    #[test]
    fn an_inline_test_module_is_not_counted() {
        let source = format!(
            "{}#[cfg(test)]\nmod tests {{\n{}}}\n",
            filler(1200),
            filler(900)
        );
        assert_eq!(production_lines_of(&source).unwrap(), 1200);
    }

    #[test]
    fn a_test_module_gated_on_all_test_and_unix_is_not_counted() {
        let source = format!(
            "{}#[cfg(all(test, unix))]\nmod tests {{\n{}}}\n",
            filler(1200),
            filler(900)
        );
        assert_eq!(production_lines_of(&source).unwrap(), 1200);
    }

    #[test]
    fn a_cfg_that_can_hold_outside_test_builds_is_counted() {
        for gate in [
            "any(test, unix)",
            "not(test)",
            "unix",
            "all(unix, not(test))",
        ] {
            let source = format!("#[cfg({gate})]\nmod m {{\n{}}}\n", filler(10));
            assert_eq!(production_lines_of(&source).unwrap(), 13, "cfg({gate})");
        }
    }

    #[test]
    fn braces_and_cfg_text_inside_literals_and_comments_do_not_change_the_count() {
        let source = "\
fn a() -> &'static str { \"}}}} #[cfg(test)] mod t {\" }
const RAW: &str = r#\"
} }
#[cfg(test)]
mod tests {
\"#;
// #[cfg(test)] mod nope {
/* #[cfg(test)]
mod nope2 { */
fn b() {}
";
        assert_eq!(production_lines_of(source).unwrap(), source.lines().count());
    }

    #[test]
    fn a_doc_comment_on_a_gated_item_goes_with_it() {
        let source = format!(
            "{}/// Helpers.\n#[cfg(test)]\nmod tests {{\n}}\n",
            filler(5)
        );
        assert_eq!(production_lines_of(&source).unwrap(), 5);
    }

    #[test]
    fn gated_functions_uses_and_members_are_not_counted() {
        let source = "\
fn real() {}
#[cfg(test)]
use std::fmt;
#[cfg(test)]
fn helper() {
    let _ = 1;
}
#[test]
fn a_test() {}
#[tokio::test]
async fn an_async_test() {}
struct S;
impl S {
    fn keep(&self) {}
    #[cfg(test)]
    fn only_in_tests(&self) {
        let _ = 2;
    }
}
trait T {
    fn kept();
    #[cfg(test)]
    fn gated();
}
";
        // real, struct S, the impl without its gated fn (3 lines), the trait
        // without its gated fn (3 lines).
        assert_eq!(production_lines_of(source).unwrap(), 1 + 1 + 3 + 3);
    }

    #[test]
    fn a_gated_item_inside_a_gated_item_is_subtracted_once() {
        let source = format!(
            "{}#[cfg(test)]\nmod outer {{\n#[cfg(test)]\nmod inner {{\n{}}}\n}}\n",
            filler(7),
            filler(20)
        );
        assert_eq!(production_lines_of(&source).unwrap(), 7);
    }

    #[test]
    fn gated_items_sharing_a_line_cover_it_once() {
        let source = "#[cfg(test)] fn a() {} #[cfg(test)] fn b() {}\nfn real() {}\n";
        assert_eq!(production_lines_of(source).unwrap(), 1);
    }

    #[test]
    fn an_inner_cfg_test_attribute_makes_the_whole_file_test_only() {
        let source = format!("#![cfg(test)]\n{}", filler(50));
        assert_eq!(production_lines_of(&source).unwrap(), 0);
    }

    #[test]
    fn a_file_that_does_not_parse_is_an_error_not_a_skip() {
        let error = production_lines_of("fn broken( {\n").unwrap_err();
        assert!(format!("{error:#}").contains("line 1"), "{error:#}");
    }

    #[test]
    fn the_cfg_evaluator_only_accepts_what_holds_solely_in_test_builds() {
        let table = [
            ("test", true),
            ("all(test, unix)", true),
            ("all(unix, test)", true),
            ("all(test, feature = \"x\")", true),
            ("all(unix, windows)", false),
            ("any(test)", true),
            ("any(test, test)", true),
            ("any(test, unix)", false),
            ("any()", false),
            ("not(test)", false),
            ("all(unix, not(test))", false),
            ("feature = \"test\"", false),
            ("unix", false),
        ];
        for (text, expected) in table {
            let meta: Meta = syn::parse_str(text).unwrap();
            assert_eq!(cfg_is_test_only(&meta).unwrap(), expected, "cfg({text})");
        }
    }

    // --- the limit ----------------------------------------------------------

    #[test]
    fn a_file_of_exactly_the_limit_passes_and_one_line_more_fails() {
        let files = [
            ("at_limit.rs".to_string(), MAX_PRODUCTION_LINES),
            ("over.rs".to_string(), MAX_PRODUCTION_LINES + 1),
        ];
        assert_eq!(
            offenders(&files, MAX_PRODUCTION_LINES),
            vec![Offender {
                path: "over.rs".to_string(),
                lines: MAX_PRODUCTION_LINES + 1,
            }]
        );
    }

    #[test]
    fn a_real_file_just_over_the_limit_is_reported_by_name_with_its_count() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/lib.rs", &filler(MAX_PRODUCTION_LINES + 1));
        let files = counted(dir.path()).unwrap();
        let over = offenders(&files, MAX_PRODUCTION_LINES);
        assert_eq!(over.len(), 1);
        let text = failed_text(&Outcome {
            checked: files.len(),
            offenders: over,
        });
        assert!(
            text.contains("src/lib.rs: 1501 production lines (1 over)"),
            "{text}"
        );

        write(dir.path(), "src/lib.rs", &filler(MAX_PRODUCTION_LINES));
        let files = counted(dir.path()).unwrap();
        assert!(offenders(&files, MAX_PRODUCTION_LINES).is_empty());
    }

    #[test]
    fn offenders_are_listed_largest_first() {
        let files = [
            ("b.rs".to_string(), 1600),
            ("a.rs".to_string(), 1600),
            ("c.rs".to_string(), 1900),
        ];
        let paths: Vec<String> = offenders(&files, 1500)
            .into_iter()
            .map(|o| o.path)
            .collect();
        assert_eq!(paths, ["c.rs", "a.rs", "b.rs"]);
    }

    // --- which files count --------------------------------------------------

    #[test]
    fn an_external_test_module_is_neither_counted_nor_charged_to_its_parent() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "src/lib.rs",
            "fn real() {}\n#[cfg(test)]\nmod tests;\n",
        );
        write(dir.path(), "src/tests.rs", &filler(3000));
        let files = counted(dir.path()).unwrap();
        assert_eq!(files, [("src/lib.rs".to_string(), 1)]);
    }

    #[test]
    fn an_external_module_gated_on_all_test_and_unix_with_a_path_is_not_counted() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "src/lib.rs",
            "fn real() {}\n#[cfg(all(test, unix))]\n#[path = \"../shared/props.rs\"]\nmod props;\n",
        );
        write(dir.path(), "shared/props.rs", &filler(3000));
        let files = counted(dir.path()).unwrap();
        assert_eq!(files, [("src/lib.rs".to_string(), 1)]);
    }

    #[test]
    fn modules_below_a_gated_file_are_reached_and_not_counted() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/lib.rs", "#[cfg(test)]\nmod t;\n");
        write(dir.path(), "src/t.rs", "mod u;\n");
        write(dir.path(), "src/t/u.rs", &filler(2000));
        let files = counted(dir.path()).unwrap();
        assert_eq!(files, [("src/lib.rs".to_string(), 0)]);
    }

    #[test]
    fn an_external_module_inside_a_gated_inline_module_is_not_counted() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "src/lib.rs",
            "fn real() {}\n#[cfg(test)]\nmod harness {\n    mod cases;\n}\n",
        );
        write(dir.path(), "src/harness/cases.rs", &filler(3000));
        let files = counted(dir.path()).unwrap();
        assert_eq!(files, [("src/lib.rs".to_string(), 1)]);
    }

    #[test]
    fn child_modules_resolve_the_way_rustc_resolves_them() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "src/lib.rs",
            "mod flat;\nmod nested;\nmod a {\n    mod inline_child;\n}\n",
        );
        write(dir.path(), "src/flat.rs", "mod below_flat;\n");
        write(dir.path(), "src/flat/below_flat.rs", "fn x() {}\n");
        write(dir.path(), "src/nested/mod.rs", "mod beside_mod_rs;\n");
        write(dir.path(), "src/nested/beside_mod_rs.rs", "fn y() {}\n");
        write(dir.path(), "src/a/inline_child.rs", "fn z() {}\n");
        let names: Vec<String> = counted(dir.path())
            .unwrap()
            .into_iter()
            .map(|(path, _)| path)
            .collect();
        assert_eq!(
            names,
            [
                "src/a/inline_child.rs",
                "src/flat.rs",
                "src/flat/below_flat.rs",
                "src/lib.rs",
                "src/nested/beside_mod_rs.rs",
                "src/nested/mod.rs",
            ]
        );
    }

    #[test]
    fn a_file_under_a_tests_directory_is_not_a_source_file() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "tests/harness/src/lib.rs", &filler(4000));
        let packages = vec![Package {
            src_dir: dir.path().join("tests/harness/src"),
            roots: vec![dir.path().join("tests/harness/src/lib.rs")],
        }];
        assert!(collect(dir.path(), &packages).unwrap().is_empty());
    }

    #[test]
    fn a_source_file_nothing_declares_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/lib.rs", "fn a() {}\n");
        write(dir.path(), "src/stray.rs", &filler(4000));
        let error = counted(dir.path()).unwrap_err();
        assert!(format!("{error:#}").contains("src/stray.rs"), "{error:#}");
    }

    #[test]
    fn a_declared_module_with_no_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/lib.rs", "mod missing;\n");
        let error = counted(dir.path()).unwrap_err();
        assert!(format!("{error:#}").contains("missing"), "{error:#}");
    }

    // --- the advisory wrapper ----------------------------------------------

    #[test]
    fn the_advisory_wrapper_reports_offenders_and_still_succeeds() {
        let outcome = Outcome {
            checked: 3,
            offenders: vec![Offender {
                path: "src/big.rs".to_string(),
                lines: 1600,
            }],
        };
        let mut out = Vec::new();
        advisory_to(Ok(outcome), &mut out).expect("advisory never fails on a violation");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("src/big.rs: 1600 production lines"), "{text}");
        assert!(text.contains("not blocking yet"), "{text}");
    }

    #[test]
    fn the_advisory_wrapper_succeeds_even_when_the_check_cannot_run() {
        let mut out = Vec::new();
        advisory_to(Err(anyhow!("cargo metadata failed")), &mut out).expect("advisory never fails");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("could not run"), "{text}");
        assert!(text.contains("cargo metadata failed"), "{text}");
    }
}
