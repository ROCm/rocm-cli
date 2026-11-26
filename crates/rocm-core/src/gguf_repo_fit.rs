// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Every quantization of a GGUF repository, each with its own fit.
//!
//! [`crate::gguf_fit`] answers for one file. A Hugging Face GGUF repository
//! publishes the same model at a dozen or more quantizations that differ
//! hugely in size, and a user who does not know which of them their machine
//! can run has nothing to name. This module answers the repository: a header
//! per quantization, read through [`crate::gguf_source`] under one shared
//! budget, each sized by [`estimate_fit`] against the same host.
//!
//! A header per file, never one header scaled by the others' sizes. The
//! weights are a file's size less *its* `data_offset`, and the input embedding
//! is clamped against *its* next tensor's offset (`sizing` in
//! [`crate::gguf_fit`]); both differ per file, so a borrowed header would
//! state one quantization's breakdown with another's confidence. A
//! quantization that could not be reached inside the budget is reported as not
//! estimated, with the reason, and never guessed.

use crate::diagnose::Fix;
use crate::gguf_fit::{
    CannotEstimateReason, ContextEstimate, ESTIMATE_CAVEAT, EstimateOutcome, FitRequest,
    KvCacheType, SERVE_CONTEXT_NOTE, estimate_fit, shown_total_need, tokens,
};
use crate::gguf_source::{
    GgufRef, GgufSource, NotProbed, QuantProbeOutcome, RepositoryProbe, is_safe_variant,
    parse_gguf_ref,
};
use crate::model_readiness::{
    ContextRequest, DEFAULT_CONTEXT_SIZE, HostFacts, ModelReadiness, UndeterminedReason, format_gib,
};
use serde::{Deserialize, Serialize};

const GIB: f64 = (1024 * 1024 * 1024) as f64;

/// Why one quantization carries no estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotEstimatedReason {
    /// The budget the whole repository read shares ran out first.
    BudgetSpent,
    /// The repository publishes more quantizations than one command reads.
    TooManyQuantizations,
    /// The file was reached and its header could not be read.
    HeaderUnreadable,
}

/// One quantization of a repository: either an estimate, or why there is none.
///
/// Tagged `answer` rather than `outcome`, and flattened into [`QuantFit`], so
/// one row is one flat JSON object: the estimate's own `outcome` (fits /
/// does not fit / cannot estimate) keeps that name, and `answer` says whether
/// there is an estimate at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "answer")]
pub enum QuantFitOutcome {
    /// The header was read and sized against this host.
    Estimated(Box<ContextEstimate>),
    /// No figures, and why.
    NotEstimated {
        reason: NotEstimatedReason,
        /// The specific cause, in words.
        detail: String,
    },
}

/// The fit of one quantization a repository publishes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantFit {
    /// The GGUF file in the repository, by full name.
    pub file: String,
    /// The quantization label read off that name (`Q4_K_M`, `IQ4_XS`, `F16`),
    /// when the name carries one. `None` for a file named some other way: the
    /// row then shows the file name, never a guessed label.
    pub quantization: Option<String>,
    /// The size of the whole file in bytes, when it is known.
    pub size_bytes: Option<u64>,
    #[serde(flatten)]
    pub outcome: QuantFitOutcome,
}

impl QuantFit {
    /// What to call this row: its quantization label, or its file name when the
    /// name carries no label.
    #[must_use]
    pub fn label(&self) -> &str {
        self.quantization.as_deref().unwrap_or(&self.file)
    }

    /// The estimate this row carries, whether or not it worked any figures out.
    fn estimate(&self) -> Option<&ContextEstimate> {
        match &self.outcome {
            QuantFitOutcome::Estimated(estimate) => Some(estimate),
            QuantFitOutcome::NotEstimated { .. } => None,
        }
    }

    /// The estimate, when this quantization has one with figures in it.
    fn sized(&self) -> Option<&ContextEstimate> {
        self.estimate()
            .filter(|estimate| estimate.total_bytes.is_some())
    }

    /// Whether this row's figures were put beside a figure for this host's
    /// memory, whichever way the comparison then came out -- including a row
    /// that was given no verdict because the formula does not measure its
    /// layout. The one question [`ContextEstimate::compared_with_memory`] asks,
    /// asked of a row.
    #[must_use]
    pub fn compared_with_memory(&self) -> bool {
        self.estimate()
            .is_some_and(ContextEstimate::compared_with_memory)
    }

    /// Whether this quantization fits on the host it was sized against.
    #[must_use]
    pub fn fits(&self) -> bool {
        self.sized()
            .is_some_and(|estimate| estimate.outcome == EstimateOutcome::Fits)
    }
}

/// Every quantization a repository publishes, with its fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepositoryFit {
    /// `owner/repo`, as the user wrote it.
    pub repository: String,
    /// One entry per quantization, smallest first, so the one a modest machine
    /// can actually run is at the top.
    pub quantizations: Vec<QuantFit>,
    /// How many GGUF files in the repository are multimodal projectors, which
    /// are not language models and are not sized.
    pub projectors: usize,
    /// The context size every row was sized for.
    pub context_size: u64,
}

impl RepositoryFit {
    /// How many quantizations were sized against this host.
    #[must_use]
    pub fn estimated(&self) -> usize {
        self.quantizations
            .iter()
            .filter(|quant| quant.sized().is_some())
            .count()
    }

    /// How many had their GGUF header fetched and parsed, whether or not a size
    /// came out of it.
    ///
    /// Apart from [`Self::estimated`], which counts rows that reached figures.
    /// A header can be read and still yield none -- one shard of a split model
    /// carries the metadata but not the weights, and a projector is not a
    /// language model -- so reporting `estimated` as the number of headers read
    /// called those rows unread, and offered the remedy for a row nobody had
    /// looked at.
    #[must_use]
    pub fn read(&self) -> usize {
        self.quantizations
            .iter()
            .filter(|quant| quant.estimate().is_some())
            .count()
    }

    /// How many were compared with memory this host was measured to have and
    /// came out of it with a verdict.
    ///
    /// Apart from [`Self::estimated`], because a host whose GPU memory could
    /// not be read still gets every quantization's figures -- but none of them
    /// a verdict, and "0 of 14 fit" would read as a firm no rather than as a
    /// comparison that never happened.
    ///
    /// Apart from [`Self::compared_with_memory`], because a row whose layout
    /// the formula does not measure (mixture-of-experts, state-space, a header
    /// missing a width) is deliberately given no verdict even though both
    /// figures are there: its own row says "not compared", so counting it in
    /// the denominator of "N of M fit" would contradict the table underneath.
    #[must_use]
    pub fn compared(&self) -> usize {
        self.quantizations
            .iter()
            .filter(|quant| {
                quant.estimate().is_some_and(|estimate| {
                    estimate.compared_with_memory()
                        && estimate.outcome != EstimateOutcome::CannotEstimate
                })
            })
            .count()
    }

    /// How many rows have figures of their own beside a figure for this host's
    /// memory -- the rows a memory reading taken from the wrong device would
    /// have falsified, whether or not they were given a verdict.
    #[must_use]
    pub fn compared_with_memory(&self) -> usize {
        self.quantizations
            .iter()
            .filter(|quant| quant.compared_with_memory())
            .count()
    }

    /// How many fit.
    #[must_use]
    pub fn fitting(&self) -> usize {
        self.quantizations.iter().filter(|q| q.fits()).count()
    }
}

/// Whether `readiness` is an answer about a bare `owner/repo`: a reference the
/// catalog does not carry, naming a repository rather than one file.
///
/// The one place that decides it, so the spinner the CLI draws (`hub_read` in
/// `apps/rocm/src/diagnose_model.rs`) and the probe itself cannot disagree
/// about whether a request is about to be made.
#[must_use]
pub fn names_a_bare_repository(readiness: &ModelReadiness) -> Option<(String, String)> {
    if readiness.canonical_model_id.is_some()
        || readiness.undetermined_reason != Some(UndeterminedReason::ModelNotCurated)
    {
        return None;
    }
    match parse_gguf_ref(&readiness.model_ref) {
        Some(GgufRef::Hub {
            owner,
            repo,
            variant: None,
        }) => Some((owner, repo)),
        _ => None,
    }
}

/// Answer a bare `owner/repo` by sizing every quantization it publishes.
///
/// Does nothing for any other answer, so a caller can call it unconditionally
/// beside [`crate::model_readiness::attach_context_estimate`], which leaves a
/// bare repository alone.
pub fn attach_repository_fit(
    readiness: &mut ModelReadiness,
    request: &ContextRequest,
    host: &HostFacts,
    source: &dyn GgufSource,
) {
    let Some((owner, repo)) = names_a_bare_repository(readiness) else {
        return;
    };
    let fit_request = FitRequest {
        context_size: request.context_size.unwrap_or(DEFAULT_CONTEXT_SIZE),
        context_size_defaulted: request.context_size.is_none(),
        kv_cache_type: request.kv_cache_type.unwrap_or(KvCacheType::F16),
    };
    match source.probe_quantizations(&owner, &repo) {
        Ok(probe) => {
            let fit = repository_fit(&readiness.model_ref, probe, &fit_request, host);
            apply_repository_fit(readiness, fit);
        }
        Err(error) => {
            readiness.evidence.push(format!(
                "the files of `{}` could not be listed, so none of its quantizations was sized: \
                 {error:#}",
                readiness.model_ref
            ));
            readiness.route = None;
            readiness.fix = Some(Fix {
                summary: "act on the reason above (a misspelt repository, no route to Hugging \
                          Face, a busy hub to wait for, or a gated repository), then ask again"
                    .to_owned(),
                verify: format!("rocm diagnose --model {}", readiness.model_ref),
                notes: vec![
                    "a gated or private repository cannot be read, because no token is sent: \
                     download a file yourself and ask about its local path"
                        .to_owned(),
                ],
                ..Fix::default()
            });
        }
    }
}

/// Turn what the probe found into the per-quantization answer, sizing each
/// header that was read against `host`.
#[must_use]
pub fn repository_fit(
    repository: &str,
    probe: RepositoryProbe,
    request: &FitRequest,
    host: &HostFacts,
) -> RepositoryFit {
    let quantizations = probe
        .quants
        .into_iter()
        .map(|quant| {
            let quantization = quantization_label(&quant.file);
            let outcome = match (quant.outcome, quant.size) {
                (QuantProbeOutcome::Read(header), Some(size)) => {
                    let estimate = estimate_fit(
                        &header,
                        size,
                        &format!("{repository}: {}", quant.file),
                        request,
                        host.accelerator_memory,
                        host.system_ram_gib,
                    );
                    QuantFitOutcome::Estimated(Box::new(estimate))
                }
                // A header with no size cannot be sized: the weights are the
                // file's size less its header. The probe only produces this
                // pairing if a `HEAD` answered without a length.
                (QuantProbeOutcome::Read(_), None) => QuantFitOutcome::NotEstimated {
                    reason: NotEstimatedReason::HeaderUnreadable,
                    detail: "the hub did not report how large this file is, so its weights \
                             cannot be sized"
                        .to_owned(),
                },
                (QuantProbeOutcome::NotRead { reason, detail }, _) => {
                    QuantFitOutcome::NotEstimated {
                        reason: match reason {
                            NotProbed::BudgetSpent => NotEstimatedReason::BudgetSpent,
                            NotProbed::TooManyQuantizations => {
                                NotEstimatedReason::TooManyQuantizations
                            }
                            NotProbed::Unreadable => NotEstimatedReason::HeaderUnreadable,
                        },
                        detail,
                    }
                }
            };
            QuantFit {
                file: quant.file,
                quantization,
                size_bytes: quant.size,
                outcome,
            }
        })
        .collect();
    RepositoryFit {
        repository: repository.to_owned(),
        quantizations,
        projectors: probe.projectors,
        context_size: request.context_size,
    }
}

/// Put `fit` on `readiness`, with the evidence that reads it out and the plan
/// that follows from it.
fn apply_repository_fit(readiness: &mut ModelReadiness, fit: RepositoryFit) {
    // The route sends a user to the upstream tracker because the catalog lacks
    // the model. The repository was read directly, so that is no longer the
    // question.
    readiness.route = None;
    readiness.fix = None;
    let total = fit.quantizations.len();
    if total == 0 {
        readiness.evidence.push(if fit.projectors > 0 {
            format!(
                "every GGUF file in `{}` is a multimodal projector (mmproj), not a language \
                 model, so there is nothing to size",
                fit.repository
            )
        } else {
            format!(
                "`{}` holds no GGUF file, so there is nothing to size",
                fit.repository
            )
        });
        readiness.quantizations = Some(fit);
        return;
    }
    let estimated = fit.estimated();
    let read = fit.read();
    readiness.evidence.push(format!(
        "`{}` names a repository, so every quantization in it was answered: it publishes {total} \
         language-model {}, of which {read} had {} read and {estimated} could be sized from it; \
         the weights were not downloaded",
        fit.repository,
        plural(total, "quantization", "quantizations"),
        plural(read, "a GGUF header", "GGUF headers"),
    ));
    // Two different reasons a row carries no figures, and only one of them has
    // this remedy. A row nobody read -- past the probe cap, or past the budget
    // -- is sized by naming it alone, because that asks for the one header this
    // command skipped. A row that *was* read and still has no figures is a file
    // the formula cannot size at all: one shard of a split model carries the
    // metadata and not the weights, and `rocm diagnose --model owner/repo:QUANT`
    // on that shard says exactly the same thing again. Offering the remedy for
    // both sent the second group round a loop.
    let unread = total - read;
    if unread > 0 {
        readiness.evidence.push(format!(
            "{unread} of the {total} {} not read, for the reason on its row; naming one as \
             `{}:QUANT` reads and sizes it on its own",
            plural(unread, "was", "were"),
            fit.repository
        ));
    }
    let read_but_unsized = read - estimated;
    if read_but_unsized > 0 {
        readiness.evidence.push(format!(
            "{read_but_unsized} of the {total} {} read but could not be sized, for the reason on \
             its row; naming one on its own gives the same answer, because the file itself is \
             not one this estimate can size",
            plural(read_but_unsized, "was", "were"),
        ));
    }
    // "0 of 14 fit" and "none of them was compared" are different answers, and
    // only the second is true on a host whose memory could not be read. Said
    // once here rather than on every row, which would repeat one fact about
    // the host a dozen times.
    if estimated > 0 && fit.compared() == 0 {
        readiness.evidence.push(format!(
            "none of them was compared with this host: {}, so each row gives what the \
             quantization needs and no verdict",
            fit.quantizations
                .iter()
                .find_map(|quant| match &quant.outcome {
                    QuantFitOutcome::Estimated(estimate) => Some(uncompared(estimate)),
                    QuantFitOutcome::NotEstimated { .. } => None,
                })
                .unwrap_or("nothing was compared")
        ));
    } else if fit.compared() > 0 && fit.fitting() == 0 {
        readiness.evidence.push(format!(
            "no quantization that was compared fits in the memory this host offers at {}",
            tokens(fit.context_size)
        ));
    }
    if estimated > 0 {
        readiness.warnings.push(ESTIMATE_CAVEAT.to_owned());
        readiness.warnings.push(SERVE_CONTEXT_NOTE.to_owned());
    }
    // The plan is the quantization to look at next: the largest that fits, so
    // the user keeps the most quality their host can hold, or the smallest of
    // all when none does. Asking about it by name is what produces the full
    // breakdown -- the per-quantization rows carry a total, not the parts.
    if let Some(next) = fit
        .quantizations
        .iter()
        .rev()
        .find(|quant| quant.fits())
        .or_else(|| fit.quantizations.first())
        && let Some(reference) = quantization_reference(&fit.repository, next)
    {
        // "The largest that fits" is only true of the rows that were read: a
        // row the cap or the budget cut off was never compared, and a larger
        // one among them could fit too. Where one went unread the claim is
        // narrowed to what was actually looked at, rather than promising a
        // maximum over files this command never saw.
        let largest = if fit.quantizations.len() > fit.estimated() {
            "the largest of those read that fits"
        } else {
            "the largest that fits"
        };
        readiness.fix = Some(Fix {
            summary: format!(
                "ask about one quantization by name for the parts behind its total ({} is {})",
                next.label(),
                if next.fits() {
                    largest
                } else {
                    "the smallest in the repository"
                }
            ),
            verify: format!("rocm diagnose --model {reference}"),
            ..Fix::default()
        });
    }
    readiness.quantizations = Some(fit);
}

/// "1 quantization" against "3 quantizations", without a stray "(s)".
const fn plural<'a>(count: usize, one: &'a str, many: &'a str) -> &'a str {
    if count == 1 { one } else { many }
}

/// How to name one quantization on the command line, when it can be named at
/// all: the label when the file carries one, else the file name as the listing
/// gives it, and only when that name is one `rocm diagnose --model` would
/// accept. A plan that printed an unquotable name would name a command that
/// does not run.
///
/// The file keeps any sub-directory it came with. `select_gguf_file` in
/// [`crate::gguf_source`] refuses a bare name two directories share ("matches
/// several files; name one exactly, as `owner/repo:dir/file.gguf`"), which is
/// exactly the form the listing already gave, and `is_safe_variant` accepts it.
fn quantization_reference(repository: &str, quant: &QuantFit) -> Option<String> {
    let variant = quant
        .quantization
        .clone()
        .unwrap_or_else(|| quant.file.clone());
    is_safe_variant(&variant).then(|| format!("{repository}:{variant}"))
}

/// The quantization label a GGUF file name carries, if any.
///
/// The last `-` or `.` separated segment that is shaped like one, because the
/// label is conventionally last (`qwen2.5-7b-instruct-q4_k_m.gguf`) while an
/// earlier segment can look like one by accident. `None` for a name that
/// carries none, so a row shows the file name rather than a guess.
#[must_use]
pub fn quantization_label(name: &str) -> Option<String> {
    let file = name.rsplit('/').next()?;
    let stem = file.get(..file.len().checked_sub(".gguf".len())?)?;
    stem.split(['-', '.'])
        .rfind(|segment| looks_like_quantization(segment))
        .map(str::to_owned)
}

/// Whether one name segment is shaped like a GGUF quantization label.
///
/// `Q` (optionally after `I` or `T`) then one or two digits then an optional
/// `_`-joined tail, or a plain float width. The digit requirement is what keeps
/// `Qwen2` out: a model family whose name starts with `Q` would otherwise be
/// printed as this file's quantization.
fn looks_like_quantization(segment: &str) -> bool {
    let upper = segment.to_ascii_uppercase();
    if matches!(upper.as_str(), "F16" | "F32" | "BF16") {
        return true;
    }
    let Some(rest) = upper
        .strip_prefix("IQ")
        .or_else(|| upper.strip_prefix("TQ"))
        .or_else(|| upper.strip_prefix('Q'))
    else {
        return false;
    };
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    let tail = &rest[digits..];
    (1..=2).contains(&digits)
        && (tail.is_empty() || tail.starts_with('_'))
        && tail.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// A file size as the rows show it.
fn file_size(bytes: Option<u64>) -> String {
    bytes.map_or_else(
        || "size unknown".to_owned(),
        |size| format_gib(size as f64 / GIB),
    )
}

/// Why a quantization with figures got no verdict, short enough to sit on a
/// row. The reasons that are facts about the host repeat on every row of the
/// table, so each is stated in a few words; `apply_repository_fit` says the
/// whole of it once, in the evidence above the table.
const fn uncompared(estimate: &ContextEstimate) -> &'static str {
    match estimate.cannot_estimate_reason {
        Some(CannotEstimateReason::NoGpu) => "no GPU is visible to ROCm",
        Some(CannotEstimateReason::AcceleratorMemoryUnknown) => {
            "this host's GPU memory could not be read"
        }
        Some(CannotEstimateReason::UnifiedMemoryUnreadable) => {
            "this host's shared memory pool could not be read"
        }
        Some(CannotEstimateReason::UnmeasuredLayout | CannotEstimateReason::MissingMetadata) => {
            "this model's layout is not one the formula measures, so the figure can be too low"
        }
        _ => "nothing was compared",
    }
}

/// What one row says about its quantization.
fn verdict(quant: &QuantFit) -> String {
    match &quant.outcome {
        QuantFitOutcome::Estimated(estimate) => {
            match (estimate.outcome, shown_total_need(estimate)) {
                (EstimateOutcome::Fits, Some(need)) => format!("fits: about {need} needed"),
                (EstimateOutcome::DoesNotFit, Some(need)) => {
                    format!("does not fit: about {need} needed")
                }
                // Figures but no verdict: the need is still the useful half, so it
                // leads and the reason follows it.
                (EstimateOutcome::CannotEstimate, Some(need)) => {
                    format!(
                        "about {need} needed, not compared: {}",
                        uncompared(estimate)
                    )
                }
                // No figures at all: only what this file is says anything.
                (_, None) => format!(
                    "not estimated: {}",
                    estimate
                        .detail
                        .as_deref()
                        .unwrap_or("the figures could not be worked out from this header")
                ),
            }
        }
        QuantFitOutcome::NotEstimated { detail, .. } => format!("not estimated: {detail}"),
    }
}

/// The per-quantization table, in the report's own voice: a heading naming what
/// every row was sized for, then one row per quantization, smallest first.
///
/// The first line is the heading and the rest are its rows; the caller indents
/// them as it indents a plan's commands.
#[must_use]
pub fn repository_fit_lines(fit: &RepositoryFit) -> Vec<String> {
    let mut lines = vec![format!(
        "quantizations of {}, smallest first, at {}:",
        fit.repository,
        tokens(fit.context_size)
    )];
    let label_width = fit
        .quantizations
        .iter()
        .map(|quant| quant.label().chars().count())
        .max()
        .unwrap_or(0);
    let sizes: Vec<String> = fit
        .quantizations
        .iter()
        .map(|q| file_size(q.size_bytes))
        .collect();
    let size_width = sizes
        .iter()
        .map(|size| size.chars().count())
        .max()
        .unwrap_or(0);
    for (quant, size) in fit.quantizations.iter().zip(&sizes) {
        lines.push(format!(
            "{:<label_width$}  {:>size_width$}  {}",
            quant.label(),
            size,
            verdict(quant)
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::parse_gguf_header;
    use crate::gguf::test_builder::llama_like;
    use crate::gguf_source::{LiveGgufSource, QuantProbe};
    use crate::model_readiness::AcceleratorMemory;
    use std::io::{Cursor, Read as _, Write as _};
    use std::net::TcpListener;
    use std::thread;

    fn host(gpu_gib: f64) -> HostFacts {
        HostFacts {
            accelerator_memory: AcceleratorMemory::Dedicated(gpu_gib),
            system_ram_gib: Some(64.0),
        }
    }

    /// A host whose memory is not a figure at all: one whose GPU memory could
    /// not be read, and one with no GPU. Every quantization is still sized
    /// against it, and none of them is compared with it.
    fn host_without_memory(accelerator_memory: AcceleratorMemory) -> HostFacts {
        HostFacts {
            accelerator_memory,
            system_ram_gib: Some(64.0),
        }
    }

    fn request() -> FitRequest {
        FitRequest {
            context_size: 4096,
            context_size_defaulted: true,
            kv_cache_type: KvCacheType::F16,
        }
    }

    fn read(name: &str, size: u64) -> QuantProbe {
        let header = parse_gguf_header(&mut Cursor::new(llama_like(10).file(0))).unwrap();
        QuantProbe {
            file: name.to_owned(),
            size: Some(header.data_offset + size),
            outcome: QuantProbeOutcome::Read(Box::new(header)),
        }
    }

    /// One shard of a split model: the header is read and parsed, and
    /// `estimate_fit` then declines to size it because the weights are spread
    /// over files this one does not name. Read, and still no figures -- the
    /// shape that must not be reported as unread, and must not be offered the
    /// "name it on its own" remedy.
    fn read_shard(name: &str, size: u64) -> QuantProbe {
        let header = parse_gguf_header(&mut Cursor::new(
            llama_like(10).u32("split.count", 3).file(0),
        ))
        .unwrap();
        QuantProbe {
            file: name.to_owned(),
            size: Some(header.data_offset + size),
            outcome: QuantProbeOutcome::Read(Box::new(header)),
        }
    }

    /// The same file as [`read`], declared a mixture-of-experts model: the
    /// figures are worked out and the host's memory is read, and `estimate_fit`
    /// then withholds the verdict because the compute buffer was not measured
    /// for experts. Both figures present, no verdict -- the shape
    /// [`RepositoryFit::compared`] must not count.
    fn read_unjudgeable(name: &str, size: u64) -> QuantProbe {
        let header = parse_gguf_header(&mut Cursor::new(
            llama_like(10).u32("llama.expert_count", 4).file(0),
        ))
        .unwrap();
        assert!(header.moe, "premise: the fixture has to be a MoE header");
        QuantProbe {
            file: name.to_owned(),
            size: Some(header.data_offset + size),
            outcome: QuantProbeOutcome::Read(Box::new(header)),
        }
    }

    /// A "not in the catalog" answer about `model_ref`, as the readiness check
    /// reaches one.
    fn not_curated(model_ref: &str) -> ModelReadiness {
        let readiness = crate::model_readiness::assess_model_readiness(
            model_ref,
            &crate::model_readiness::ModelCatalogSource::Available(
                &crate::builtin_model_recipe_registry(),
            ),
            &host(24.0),
            &|_| crate::model_readiness::HostEngineChoice {
                engine: "lemonade".to_owned(),
                source: "test".to_owned(),
                unsupported_here: None,
            },
        );
        assert_eq!(
            readiness.undetermined_reason,
            Some(UndeterminedReason::ModelNotCurated),
            "premise: {model_ref} must be a catalog miss"
        );
        readiness
    }

    fn unread(name: &str, size: u64, reason: NotProbed) -> QuantProbe {
        QuantProbe {
            file: name.to_owned(),
            size: Some(size),
            outcome: QuantProbeOutcome::NotRead {
                reason,
                detail: "the budget ran out".to_owned(),
            },
        }
    }

    /// A source that answers a repository probe from a canned list, and counts
    /// how often each entry point was reached.
    struct FakeRepository {
        probe: std::cell::RefCell<Option<RepositoryProbe>>,
        probes: std::cell::Cell<usize>,
    }

    impl FakeRepository {
        fn new(probe: RepositoryProbe) -> Self {
            Self {
                probe: std::cell::RefCell::new(Some(probe)),
                probes: std::cell::Cell::new(0),
            }
        }
    }

    impl GgufSource for FakeRepository {
        fn open(&self, _reference: &GgufRef) -> anyhow::Result<crate::gguf_source::OpenedGguf> {
            panic!("a repository must never be opened as one file")
        }

        fn probe_quantizations(
            &self,
            _owner: &str,
            _repo: &str,
        ) -> anyhow::Result<RepositoryProbe> {
            self.probes.set(self.probes.get() + 1);
            self.probe
                .borrow_mut()
                .take()
                .ok_or_else(|| anyhow::anyhow!("probed twice"))
        }
    }

    /// The arm as the CLI reaches it: a bare repository is probed and answered
    /// with a row per quantization, and nothing else is.
    #[test]
    fn only_a_bare_repository_is_probed_and_it_is_answered_with_its_quantizations() {
        let probe = RepositoryProbe {
            quants: vec![read("m-Q4_K_M.gguf", 4 << 30)],
            projectors: 0,
        };
        let source = FakeRepository::new(probe);
        let mut readiness = not_curated("o/r");
        attach_repository_fit(
            &mut readiness,
            &ContextRequest::default(),
            &host(24.0),
            &source,
        );
        assert_eq!(source.probes.get(), 1);
        let fit = readiness.quantizations.as_ref().expect("a repository fit");
        assert_eq!(fit.estimated(), 1, "{fit:?}");
        assert!(readiness.route.is_none(), "{:?}", readiness.route);
        assert!(
            readiness.warnings.iter().any(|w| w == ESTIMATE_CAVEAT),
            "{:?}",
            readiness.warnings
        );
        // Every other reference leaves this alone, so it can be called
        // unconditionally beside the one-file estimate.
        for other in ["o/r:Q4_K_M", "models/m.gguf", "not-a-gguf-at-all"] {
            let mut untouched = not_curated(other);
            let before = untouched.clone();
            attach_repository_fit(
                &mut untouched,
                &ContextRequest::default(),
                &host(24.0),
                &FakeRepository::new(RepositoryProbe {
                    quants: Vec::new(),
                    projectors: 0,
                }),
            );
            assert_eq!(untouched, before, "{other}");
        }
    }

    /// A repository whose listing could not be read says so and offers a plan,
    /// rather than rendering an empty table that reads like a complete answer.
    #[test]
    fn a_repository_that_could_not_be_listed_says_so_rather_than_listing_nothing() {
        struct Unreachable;
        impl GgufSource for Unreachable {
            fn open(&self, _reference: &GgufRef) -> anyhow::Result<crate::gguf_source::OpenedGguf> {
                unreachable!()
            }
        }
        let mut readiness = not_curated("o/r");
        attach_repository_fit(
            &mut readiness,
            &ContextRequest::default(),
            &host(24.0),
            &Unreachable,
        );
        assert!(readiness.quantizations.is_none(), "{readiness:?}");
        assert!(
            readiness
                .evidence
                .iter()
                .any(|line| line.contains("could not be listed, so none of its quantizations")),
            "{:?}",
            readiness.evidence
        );
        assert_eq!(
            readiness.fix.as_ref().map(|fix| fix.verify.clone()),
            Some("rocm diagnose --model o/r".to_owned())
        );
    }

    /// Each quantization is sized from its own file's size, so two rows of one
    /// repository carry different weights -- which is the whole reason a header
    /// is read per file rather than one being scaled by the listing's sizes.
    #[test]
    fn each_quantization_is_sized_from_its_own_file() {
        let probe = RepositoryProbe {
            quants: vec![read("m-Q4_K_M.gguf", 4 << 30), read("m-Q8_0.gguf", 8 << 30)],
            projectors: 0,
        };
        let fit = repository_fit("o/r", probe, &request(), &host(24.0));
        let weights: Vec<u64> = fit
            .quantizations
            .iter()
            .map(|quant| match &quant.outcome {
                QuantFitOutcome::Estimated(estimate) => estimate.weights_bytes.unwrap(),
                QuantFitOutcome::NotEstimated { .. } => panic!("{quant:?} was not estimated"),
            })
            .collect();
        assert_eq!(weights.len(), 2, "{fit:?}");
        assert!(
            weights[1] > weights[0] + (3 << 30),
            "the larger file must carry the larger weights: {weights:?}"
        );
        assert_eq!(fit.quantizations[0].quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(fit.quantizations[1].quantization.as_deref(), Some("Q8_0"));
    }

    /// A quantization the budget cut off is a row saying so, not a missing row
    /// and not a figure derived from a neighbour's header.
    #[test]
    fn a_quantization_past_the_budget_is_reported_as_not_estimated() {
        let probe = RepositoryProbe {
            quants: vec![
                read("m-Q4_K_M.gguf", 4 << 30),
                unread("m-Q8_0.gguf", 8 << 30, NotProbed::BudgetSpent),
            ],
            projectors: 0,
        };
        let fit = repository_fit("o/r", probe, &request(), &host(24.0));
        assert_eq!(fit.quantizations.len(), 2, "{fit:?}");
        assert_eq!(fit.estimated(), 1, "{fit:?}");
        assert!(matches!(
            fit.quantizations[1].outcome,
            QuantFitOutcome::NotEstimated {
                reason: NotEstimatedReason::BudgetSpent,
                ..
            }
        ));
        let rendered = repository_fit_lines(&fit).join("\n");
        assert!(
            rendered.contains("Q8_0") && rendered.contains("not estimated: the budget ran out"),
            "{rendered}"
        );
        // The unanswered row carries no total of its own.
        assert!(
            !rendered.lines().last().unwrap().contains("needed"),
            "{rendered}"
        );
    }

    /// The rows read smallest-first and say which fit, so the one a modest host
    /// can run is at the top.
    #[test]
    fn the_rows_are_smallest_first_and_name_the_quantization_and_its_size() {
        let probe = RepositoryProbe {
            quants: vec![read("m-Q4_K_M.gguf", 4 << 30), read("m-F16.gguf", 60 << 30)],
            projectors: 0,
        };
        let fit = repository_fit("o/r", probe, &request(), &host(24.0));
        let lines = repository_fit_lines(&fit);
        assert!(lines[0].starts_with("quantizations of o/r, smallest first, at 4096 tokens:"));
        assert!(lines[1].starts_with("Q4_K_M"), "{lines:?}");
        assert!(
            lines[1].contains("GiB") && lines[1].contains("fits: about"),
            "{lines:?}"
        );
        assert!(lines[2].starts_with("F16  "), "{lines:?}");
        assert!(lines[2].contains("does not fit: about"), "{lines:?}");
        assert_eq!(fit.fitting(), 1, "{fit:?}");
    }

    /// A label is only read off a name that carries one; a family name that
    /// starts with `Q` is not mistaken for a quantization.
    #[test]
    fn a_quantization_label_is_read_only_where_the_name_carries_one() {
        for (name, label) in [
            ("qwen2.5-7b-instruct-q4_k_m.gguf", Some("q4_k_m")),
            ("dir/Model.IQ4_XS.gguf", Some("IQ4_XS")),
            ("m-Q4_0_4_4.gguf", Some("Q4_0_4_4")),
            ("m-f16.gguf", Some("f16")),
            ("Qwen2-7B.gguf", None),
            ("model.gguf", None),
        ] {
            assert_eq!(quantization_label(name).as_deref(), label, "{name}");
        }
    }

    /// A repository whose every GGUF is a projector reads as that, not as an
    /// empty answer that could be mistaken for a complete one.
    #[test]
    fn a_repository_with_no_language_model_says_so_rather_than_listing_nothing() {
        let fit = repository_fit(
            "o/r",
            RepositoryProbe {
                quants: Vec::new(),
                projectors: 2,
            },
            &request(),
            &host(24.0),
        );
        let mut readiness = not_curated("o/r");
        apply_repository_fit(&mut readiness, fit);
        assert!(
            readiness
                .evidence
                .iter()
                .any(|line| line.contains("is a multimodal projector (mmproj)")),
            "{:?}",
            readiness.evidence
        );
        assert!(readiness.fix.is_none(), "{:?}", readiness.fix);
    }

    /// A host that can hold none of them is told so in as many words: a table
    /// of "does not fit" rows with no summary reads like a list to pick from.
    #[test]
    fn a_repository_where_nothing_fits_says_so_and_still_names_a_next_step() {
        let probe = RepositoryProbe {
            quants: vec![read("m-Q4_K_M.gguf", 40 << 30)],
            projectors: 0,
        };
        let fit = repository_fit("o/r", probe, &request(), &host(8.0));
        let mut readiness = not_curated("o/r");
        apply_repository_fit(&mut readiness, fit);
        assert!(
            readiness
                .evidence
                .iter()
                .any(|line| line.contains("no quantization that was compared fits")),
            "{:?}",
            readiness.evidence
        );
        assert_eq!(
            readiness.fix.as_ref().map(|fix| fix.verify.clone()),
            Some("rocm diagnose --model o/r:Q4_K_M".to_owned()),
            "{:?}",
            readiness.fix
        );
    }

    /// The plan names the largest quantization that fits, so the user keeps the
    /// most quality their host can hold -- not the first row that happened to
    /// pass, which on a capable machine is the one with the least quality in
    /// the repository.
    #[test]
    fn the_plan_names_the_largest_quantization_that_fits_not_the_smallest() {
        let three = || {
            vec![
                read("m-Q4_K_M.gguf", 4 << 30),
                read("m-Q6_K.gguf", 8 << 30),
                read("m-F16.gguf", 60 << 30),
            ]
        };
        let fit = repository_fit(
            "o/r",
            RepositoryProbe {
                quants: three(),
                projectors: 0,
            },
            &request(),
            &host(24.0),
        );
        assert_eq!(
            fit.fitting(),
            2,
            "premise: two of the three have to fit, or there is no larger one to prefer: {fit:?}"
        );
        let mut readiness = not_curated("o/r");
        apply_repository_fit(&mut readiness, fit);
        let plan = readiness.fix.as_ref().expect("a plan");
        assert_eq!(plan.verify, "rocm diagnose --model o/r:Q6_K", "{plan:?}");
        assert!(
            plan.summary.contains("Q6_K is the largest that fits"),
            "{}",
            plan.summary
        );
        // With a row that was never read, "the largest that fits" would be a
        // claim about a file this command never looked at, so the plan narrows
        // it to the ones it did read.
        let mut quants = three();
        quants.push(unread("m-Q8_0.gguf", 9 << 30, NotProbed::BudgetSpent));
        let partial = repository_fit(
            "o/r",
            RepositoryProbe {
                quants,
                projectors: 0,
            },
            &request(),
            &host(24.0),
        );
        let mut readiness = not_curated("o/r");
        apply_repository_fit(&mut readiness, partial);
        let plan = readiness.fix.as_ref().expect("a plan");
        assert_eq!(plan.verify, "rocm diagnose --model o/r:Q6_K", "{plan:?}");
        assert!(
            plan.summary
                .contains("Q6_K is the largest of those read that fits"),
            "{}",
            plan.summary
        );
    }

    /// A host whose memory is not a figure gets every quantization's need and
    /// no verdict, and is told that once, in the host's own terms. "0 of 14
    /// fit" would read as a firm no about a comparison that never happened.
    #[test]
    fn a_host_whose_memory_was_not_read_is_told_that_nothing_was_compared() {
        for (memory, because) in [
            (
                AcceleratorMemory::Unknown,
                "this host's GPU memory could not be read",
            ),
            (AcceleratorMemory::None, "no GPU is visible to ROCm"),
        ] {
            let fit = repository_fit(
                "o/r",
                RepositoryProbe {
                    quants: vec![read("m-Q4_K_M.gguf", 4 << 30), read("m-F16.gguf", 60 << 30)],
                    projectors: 0,
                },
                &request(),
                &host_without_memory(memory),
            );
            assert_eq!(fit.estimated(), 2, "{memory:?}: {fit:?}");
            assert_eq!(fit.compared(), 0, "{memory:?}: {fit:?}");
            let mut readiness = not_curated("o/r");
            apply_repository_fit(&mut readiness, fit);
            assert!(
                readiness.evidence.iter().any(|line| line.contains(&format!(
                    "none of them was compared with this host: {because}"
                ))),
                "{memory:?}: {:?}",
                readiness.evidence
            );
            // ...and never the sentence for a comparison that did happen.
            assert!(
                !readiness
                    .evidence
                    .iter()
                    .any(|line| line.contains("no quantization that was compared fits")),
                "{memory:?}: {:?}",
                readiness.evidence
            );
            let rows =
                repository_fit_lines(readiness.quantizations.as_ref().expect("a fit")).join("\n");
            assert!(
                rows.contains(&format!("not compared: {because}")),
                "{memory:?}: {rows}"
            );
        }
    }

    /// A shard is read and still has no figures, and the evidence has to say
    /// both halves of that.
    ///
    /// Counting the sized rows as "headers read" called the shard unread, and
    /// then offered it the remedy for a row nobody had looked at -- `naming one
    /// as owner/repo:QUANT sizes it on its own`, which on a shard produces the
    /// same refusal a second time. The README documents a shard as the file
    /// this estimate cannot size, so the loop was guaranteed.
    #[test]
    fn a_shard_is_reported_as_read_and_unsizeable_rather_than_unread() {
        let source = FakeRepository::new(RepositoryProbe {
            quants: vec![
                read("m-Q4_K_M.gguf", 4 << 30),
                read_shard("m-Q8_0-00001-of-00003.gguf", 8 << 30),
            ],
            projectors: 0,
        });
        let mut readiness = not_curated("o/r");
        attach_repository_fit(
            &mut readiness,
            &ContextRequest::default(),
            &host(24.0),
            &source,
        );
        let fit = readiness.quantizations.as_ref().expect("a repository fit");
        let shard = &fit.quantizations[1];
        assert_eq!(
            shard
                .estimate()
                .map(|estimate| (estimate.total_bytes, estimate.cannot_estimate_reason)),
            Some((None, Some(CannotEstimateReason::SplitFile))),
            "premise: the second row has to be read and unsizeable: {fit:?}"
        );
        assert_eq!(fit.read(), 2, "both headers were read: {fit:?}");
        assert_eq!(fit.estimated(), 1, "only one of them sized: {fit:?}");

        let evidence = readiness.evidence.join("\n");
        assert!(
            evidence.contains("2 had GGUF headers read and 1 could be sized from it"),
            "the count of headers read must not be the count that sized: {evidence}"
        );
        assert!(
            evidence.contains("1 of the 2 was read but could not be sized"),
            "the shard has to be named as read: {evidence}"
        );
        assert!(
            !evidence.contains("was not read"),
            "nothing here went unread, so no row may be offered that remedy: {evidence}"
        );
        assert!(
            !evidence.contains("`o/r:QUANT` reads and sizes it on its own"),
            "naming a shard on its own repeats the same refusal: {evidence}"
        );
    }

    /// A row the formula will not judge is not one of those compared.
    ///
    /// A mixture-of-experts quantization is sized and its figures are put
    /// beside this host's memory, and `estimate_fit` then withholds the
    /// verdict: the row itself prints "not compared". Counting it would make
    /// the heading say two quantizations were compared directly above a table
    /// saying one of them was not -- so the two counters are separated here,
    /// and each is pinned to the row text it has to agree with.
    #[test]
    fn a_row_the_formula_will_not_judge_is_not_counted_among_those_compared() {
        let fit = repository_fit(
            "o/r",
            RepositoryProbe {
                quants: vec![
                    read("m-Q4_K_M.gguf", 4 << 30),
                    read_unjudgeable("m-Q8_0.gguf", 8 << 30),
                ],
                projectors: 0,
            },
            &request(),
            &host(24.0),
        );
        let unjudgeable = &fit.quantizations[1];
        assert_eq!(
            unjudgeable
                .sized()
                .map(|estimate| (estimate.outcome, estimate.cannot_estimate_reason)),
            Some((
                EstimateOutcome::CannotEstimate,
                Some(CannotEstimateReason::UnmeasuredLayout)
            )),
            "premise: the second row has to be one with figures and no verdict: {fit:?}"
        );
        assert_eq!(fit.estimated(), 2, "both rows carry figures: {fit:?}");
        assert_eq!(
            fit.compared_with_memory(),
            2,
            "both rows had their total put beside this host's memory: {fit:?}"
        );
        assert_eq!(
            fit.compared(),
            1,
            "only the judged row may be counted as compared: {fit:?}"
        );
        assert_eq!(fit.fitting(), 1, "{fit:?}");
        // The row the count has to agree with, in the words it prints.
        let rows = repository_fit_lines(&fit).join("\n");
        assert!(
            rows.contains(
                "not compared: this model's layout is not one the formula measures, so the \
                 figure can be too low"
            ),
            "{rows}"
        );
        // And the heading above that table, which is where the count is read.
        let mut readiness = not_curated("o/r");
        apply_repository_fit(&mut readiness, fit);
        let text = crate::model_readiness::render_model_readiness_text(&readiness);
        assert!(
            text.lines().next().is_some_and(|line| line.ends_with(
                "ESTIMATED: 1 of 1 quantizations fit at 4096 tokens (calibrated on one MI300X, \
                 not a tested minimum)"
            )),
            "{text}"
        );
    }

    /// A repository where no header was read carries no estimate caveat: the
    /// caveat qualifies figures, and there are none here to qualify. Each row
    /// keeps the reason it was not read, rather than one standing in for all.
    #[test]
    fn a_repository_with_no_figures_carries_no_estimate_caveat() {
        let fit = repository_fit(
            "o/r",
            RepositoryProbe {
                quants: vec![
                    unread("m-Q4_K_M.gguf", 4 << 30, NotProbed::Unreadable),
                    unread("m-Q8_0.gguf", 8 << 30, NotProbed::TooManyQuantizations),
                ],
                projectors: 0,
            },
            &request(),
            &host(24.0),
        );
        assert_eq!(fit.estimated(), 0, "{fit:?}");
        assert!(
            matches!(
                fit.quantizations[0].outcome,
                QuantFitOutcome::NotEstimated {
                    reason: NotEstimatedReason::HeaderUnreadable,
                    ..
                }
            ),
            "{fit:?}"
        );
        assert!(
            matches!(
                fit.quantizations[1].outcome,
                QuantFitOutcome::NotEstimated {
                    reason: NotEstimatedReason::TooManyQuantizations,
                    ..
                }
            ),
            "{fit:?}"
        );
        let mut readiness = not_curated("o/r");
        apply_repository_fit(&mut readiness, fit);
        for caveat in [ESTIMATE_CAVEAT, SERVE_CONTEXT_NOTE] {
            assert!(
                !readiness.warnings.iter().any(|warning| warning == caveat),
                "{:?}",
                readiness.warnings
            );
        }
        // Nor the sentence about a comparison, which needs figures too.
        assert!(
            !readiness
                .evidence
                .iter()
                .any(|line| line.contains("none of them was compared")),
            "{:?}",
            readiness.evidence
        );
    }

    /// A file whose name carries no quantization label is named by the whole
    /// path the listing gave, sub-directory and all: the last component alone
    /// is a name two directories can share, which the file selector refuses as
    /// ambiguous, so the plan would print a command that does not run.
    #[test]
    fn a_file_with_no_label_is_named_by_the_whole_path_the_listing_gave() {
        let fit = repository_fit(
            "o/r",
            RepositoryProbe {
                quants: vec![read("dir/model.gguf", 4 << 30)],
                projectors: 0,
            },
            &request(),
            &host(24.0),
        );
        assert_eq!(fit.quantizations[0].quantization, None, "{fit:?}");
        let mut readiness = not_curated("o/r");
        apply_repository_fit(&mut readiness, fit);
        assert_eq!(
            readiness.fix.as_ref().map(|fix| fix.verify.clone()),
            Some("rocm diagnose --model o/r:dir/model.gguf".to_owned()),
            "{:?}",
            readiness.fix
        );
        // And that command names the same file back.
        assert_eq!(
            parse_gguf_ref("o/r:dir/model.gguf"),
            Some(GgufRef::Hub {
                owner: "o".to_owned(),
                repo: "r".to_owned(),
                variant: Some("dir/model.gguf".to_owned()),
            })
        );
    }

    /// A hub on loopback that publishes two quantizations: one the listing gave
    /// no size for, whose `HEAD` says how large it is, and one whose bytes come
    /// back `403`. Both are answers only a real read produces, which is why
    /// this serves HTTP rather than canning a probe.
    fn fake_repository_hub(header: Vec<u8>, size: u64) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let base = format!("http://{}", listener.local_addr().expect("an address"));
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut request = Vec::new();
                let mut byte = [0_u8; 1];
                while !request.ends_with(b"\r\n\r\n") {
                    if stream.read(&mut byte).unwrap_or(0) == 0 {
                        break;
                    }
                    request.push(byte[0]);
                }
                let text = String::from_utf8_lossy(&request).into_owned();
                let method = text.split_whitespace().next().unwrap_or("").to_owned();
                let path = text.split_whitespace().nth(1).unwrap_or("").to_owned();
                let range = text
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("Range: bytes=")
                            .or_else(|| line.strip_prefix("range: bytes="))
                    })
                    .map(str::to_owned);
                if path.starts_with("/api/models/o/r") {
                    // `gated-Q4_K_M.gguf` carries a size, so it sorts first and
                    // is read without a `HEAD`; `unsized-Q8_0.gguf` carries
                    // none, which sorts it last.
                    let body = format!(
                        r#"{{"siblings":[{{"rfilename":"gated-Q4_K_M.gguf","size":{size}}},{{"rfilename":"unsized-Q8_0.gguf"}}]}}"#
                    );
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                } else if path.contains("gated-Q4_K_M.gguf") {
                    let _ = write!(
                        stream,
                        "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                } else if method == "HEAD" {
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n"
                    );
                } else if let Some(range) = range {
                    let (start, end) = range.split_once('-').expect("a byte range");
                    let (start, end): (u64, u64) = (
                        start.parse().expect("a start"),
                        end.parse().expect("an end"),
                    );
                    let body: Vec<u8> = (start..=end)
                        .map(|at| header.get(at as usize).copied().unwrap_or(0))
                        .collect();
                    let _ = write!(
                        stream,
                        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{size}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(&body);
                } else {
                    let _ = write!(
                        stream,
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                }
            }
        });
        base
    }

    /// The read behind the table, against a hub: a file the listing left a size
    /// off is sized from its own `HEAD` rather than dropped, and a file that
    /// came back `403` is reported as unreadable -- which is a different answer
    /// from the shared budget running out, because the budget would mean every
    /// later file failed too and naming this one again would not help.
    #[test]
    fn a_file_with_no_listed_size_is_sized_by_head_and_an_unreadable_one_is_not_the_budget() {
        let size = 4_u64 << 30;
        let base = fake_repository_hub(llama_like(10).header_bytes(), size);
        let probe = LiveGgufSource::with_hub(&base)
            .probe_quantizations("o", "r")
            .expect("the listing is readable");
        assert_eq!(probe.quants.len(), 2, "{probe:?}");
        assert_eq!(probe.quants[1].file, "unsized-Q8_0.gguf", "{probe:?}");
        assert_eq!(
            probe.quants[1].size,
            Some(size),
            "the size a HEAD reported has to be the one the row is sized against: {probe:?}"
        );
        assert!(
            matches!(probe.quants[1].outcome, QuantProbeOutcome::Read(_)),
            "{probe:?}"
        );
        assert!(
            matches!(
                &probe.quants[0].outcome,
                QuantProbeOutcome::NotRead {
                    reason: NotProbed::Unreadable,
                    detail,
                } if detail.contains("gated or private")
            ),
            "a file that could not be read is unreadable, never the budget: {probe:?}"
        );
        let fit = repository_fit("o/r", probe, &request(), &host(24.0));
        assert_eq!(fit.estimated(), 1, "{fit:?}");
        assert!(
            matches!(
                fit.quantizations[0].outcome,
                QuantFitOutcome::NotEstimated {
                    reason: NotEstimatedReason::HeaderUnreadable,
                    ..
                }
            ),
            "{fit:?}"
        );
        let rendered = repository_fit_lines(&fit).join("\n");
        assert!(
            !rendered.contains("ran out before this file's header was reached"),
            "{rendered}"
        );
    }
}
