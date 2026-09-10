//! Grinding (proof-of-work) bits — additive contribution to security.
//!
//! A grind sited immediately before a Fiat–Shamir challenge forces a
//! malicious prover to redo `2^pow_bits` work per resampling attempt, so it
//! adds `pow_bits` to the round-by-round error of the round that challenge
//! opens (ethSTARK [2021/582](https://eprint.iacr.org/2021/582) §5, and
//! [2024/1553](https://eprint.iacr.org/2024/1553) §2 for the round-by-round
//! accounting the composite uses).
//!
//! Which round a grind boosts therefore depends on **where** the protocol
//! grinds. [`GrindingSites`] enumerates those sites so a protocol states them
//! as data instead of the crate hardcoding one placement per protocol.
//!
//! # Vocabulary
//!
//! The same proof-of-work phase carries three names in three places.
//!
//! | transcript step | this crate's field    | report term         |
//! | --------------- | --------------------- | ------------------- |
//! | `ood_pow`       | `out_of_domain`       | `deep-ali`          |
//! | `batch_pow`     | `batch_combination`   | `batch-combination` |
//! | `lookup_pow`    | `lookup_challenge`    | `logup-fingerprint` |
//! | `commit_pow`    | FRI `commit_pow_bits` | `ldt-commit-phase`  |
//! | `query_pow`     | FRI `query_pow_bits`  | `ldt-query-phase`   |
//!
//! A site is the one identity behind all three names.
//!
//! The same table is carried as data, keyed on the protocol name and the step label.
//!
//! # Two places, one number
//!
//! Grinding bits are read twice for two different purposes.
//!
//! ```text
//!     transcript  ->  how much work the prover actually pays
//!     this crate  ->  how many bits the parameter set is reported to buy
//! ```
//!
//! A parameter set where the two disagree is a misreported security level.
//!
//! The check in this module compares them.

use core::fmt::{Display, Formatter, Result as FmtResult};

use serde::Serialize;

use crate::error::ErrorBits;
use crate::fri::FriRegime;
use crate::logup::LOGUP_LABEL;
use crate::report::{BATCH_LABEL, DEEP_LABEL, LDT_COMMIT_LABEL, LDT_QUERY_LABEL};

/// Bits added to the soundness budget by a `pow_bits`-bit grinding round.
/// Equal to `pow_bits` when grinding is honest; provided as a function
/// so future tweaks (multi-round PoW, variable difficulty) stay local.
pub const fn grinding_bits(pow_bits: usize) -> f64 {
    pow_bits as f64
}

/// `error` boosted by a `pow_bits`-bit grind placed before the challenge that
/// round samples. A zero-bit grind is the identity.
pub const fn boost(error: ErrorBits, pow_bits: usize) -> ErrorBits {
    ErrorBits::from_log2(error.bits() + grinding_bits(pow_bits))
}

/// Where a STARK grinds, in bits per site.
///
/// Every field defaults to `0` — a protocol declares only the sites it
/// actually uses, and a default-constructed value is neutral.
///
/// Each site is consumed by the term whose round it opens:
///
/// - [`Self::out_of_domain`] is applied to the DEEP-ALI term by
///   [`crate::stark::proven_security_report`],
///   [`crate::stark::conjectured_security_report`], and
///   [`crate::stark::legacy_security_report`].
/// - [`Self::batch_combination`] is applied to the batched-openings term by
///   [`crate::stark::proven_security_report`],
///   [`crate::stark::conjectured_security_report`], and
///   [`crate::stark::legacy_security_report`].
/// - [`Self::lookup_challenge`] is applied to the LogUp fingerprint term by
///   [`crate::logup::security_term`].
///
/// The low-degree test's own grinding sites (e.g. FRI's query- and
/// commit-phase proof-of-work) are **not** modeled here: a
/// [`crate::ldt::LowDegreeTest`] implementation carries those itself
/// (`FriRegime::query_pow_bits` / `FriRegime::commit_pow_bits`) and folds
/// them into the terms it returns, so the composite never re-applies them —
/// a site in this struct for them would be read back out unchanged, never
/// consulted.
///
/// A protocol grinding at a site this crate does not model builds its own
/// [`crate::report::SecurityTerm`], applies [`boost`] to it, and passes it
/// through `extras` — no change to this struct is needed.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GrindingSites {
    /// Bits ground before the DEEP out-of-domain point is sampled.
    pub out_of_domain: usize,
    /// Bits ground before the challenge that random-linear-combines the
    /// committed codewords into the single low-degree-test instance.
    ///
    /// This is the opening-batching challenge of the polynomial commitment
    /// scheme (`alpha` in `p3_fri::TwoAdicFriPcs::open`), **not** a FRI
    /// folding challenge — those are `FriRegime::commit_pow_bits`, carried by
    /// the low-degree test itself. It is credited only to the term
    /// [`crate::report::BATCH_LABEL`] names, which exists only when more than
    /// one codeword is batched (`InstanceShape::num_batched_functions >= 2`);
    /// with nothing to batch there is no such round and these bits buy
    /// nothing.
    ///
    /// The proven, conjectured, and legacy composites all model this round.
    /// They grade it in different proximity regimes, but the prover pays the
    /// same work regardless.
    pub batch_combination: usize,
    /// Bits ground before the lookup / permutation argument's challenges are
    /// sampled.
    pub lookup_challenge: usize,
}

impl GrindingSites {
    /// No grinding at any site — the neutral element, usable in `const`
    /// contexts where [`Default::default`] is not available.
    pub const NONE: Self = Self {
        out_of_domain: 0,
        batch_combination: 0,
        lookup_challenge: 0,
    };
}

/// One proof-of-work phase of a FRI-backed STARK, named once and for all.
///
/// The transcript, this crate, and the security report each have their own word for one phase.
///
/// A site is the identity the phase keeps across all three.
///
/// That is what lets the mapping between the vocabularies be written down rather than remembered.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GrindingSite {
    /// Grinding before the DEEP out-of-domain point is sampled.
    OutOfDomain,
    /// Grinding before the polynomial commitment scheme's opening-batching challenge.
    BatchCombination,
    /// Grinding before the LogUp argument's `(alpha, beta)` pair is sampled.
    LookupChallenge,
    /// Grinding before each of the low-degree test's folding challenges.
    LdtCommitPhase,
    /// Grinding before the low-degree test's query indices are sampled.
    LdtQueryPhase,
}

impl GrindingSite {
    /// Every site, in the order a grinding budget indexes them.
    pub const ALL: [Self; 5] = [
        Self::OutOfDomain,
        Self::BatchCombination,
        Self::LookupChallenge,
        Self::LdtCommitPhase,
        Self::LdtQueryPhase,
    ];

    /// Label of the security-report term this site's bits are added to.
    ///
    /// A site boosts exactly one term.
    ///
    /// That is what makes the boost an addition rather than a re-derivation.
    #[must_use]
    pub const fn report_label(self) -> &'static str {
        match self {
            Self::OutOfDomain => DEEP_LABEL,
            Self::BatchCombination => BATCH_LABEL,
            Self::LookupChallenge => LOGUP_LABEL,
            Self::LdtCommitPhase => LDT_COMMIT_LABEL,
            Self::LdtQueryPhase => LDT_QUERY_LABEL,
        }
    }

    /// Position of this site in the canonical site order.
    const fn index(self) -> usize {
        match self {
            Self::OutOfDomain => 0,
            Self::BatchCombination => 1,
            Self::LookupChallenge => 2,
            Self::LdtCommitPhase => 3,
            Self::LdtQueryPhase => 4,
        }
    }
}

impl Display for GrindingSite {
    /// Render the site by its own name, not by any of its three aliases.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::OutOfDomain => write!(f, "out-of-domain"),
            Self::BatchCombination => write!(f, "batch-combination"),
            Self::LookupChallenge => write!(f, "lookup-challenge"),
            Self::LdtCommitPhase => write!(f, "ldt-commit-phase"),
            Self::LdtQueryPhase => write!(f, "ldt-query-phase"),
        }
    }
}

/// Whether a protocol describes a grinding step whose difficulty is zero.
///
/// A zero-bit grind is free to produce and vacuous to check.
///
/// Whether it is described at all is therefore a per-protocol choice.
///
/// The choice is not uniform across this workspace, so an absent step cannot be read as zero bits everywhere.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ZeroBitConvention {
    /// The step is described only when the difficulty is positive.
    ///
    /// ```text
    ///     bits == 0  ->  no step at all
    ///     bits >  0  ->  one step carrying `bits`
    /// ```
    Elided,
    /// The step is described whatever the difficulty, zero included.
    ///
    /// Used where the witness travels in the proof unconditionally.
    ///
    /// The proof layout then does not depend on the difficulty.
    Always,
    /// The step is described, carrying its exact difficulty, only while its challenge's phase runs at all.
    ///
    /// An absent step then means the phase did not run, so nothing was ground.
    ///
    /// Nothing is credited either: the term the site would boost is itself absent from the report.
    ///
    /// The lookup argument is the case in point: with no interactions it contributes no term.
    WhenPhaseRuns,
}

/// One row of the grinding vocabulary.
///
/// A row is a protocol's step label, the site that label names, and how the protocol records it at zero bits.
///
/// Rows are keyed on `(protocol, label)` rather than on the label alone.
///
/// ```text
///     "query_pow"  ->  p3-fri, p3-circle, p3-whir, p3-stir all use it
///     "ood_pow"    ->  p3-uni-stark and p3-batch-stark, under two conventions
/// ```
///
/// Labels are namespaced by the protocol name in the transcript seed, so this is not a soundness problem.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GrindingStep {
    /// Protocol name bound into the transcript seed, such as `"p3-fri"`.
    pub protocol: &'static str,
    /// Label the protocol's proof-of-work step carries.
    pub label: &'static str,
    /// Phase this step grinds for.
    pub site: GrindingSite,
    /// How this protocol records the step at zero difficulty.
    pub zero_bits: ZeroBitConvention,
}

/// Every proof-of-work step of every protocol whose security this crate models.
///
/// This is the mapping table between the three vocabularies, as data.
///
/// | protocol         | label        | site              | zero bits           |
/// | ---------------- | ------------ | ----------------- | ------------------- |
/// | `p3-uni-stark`   | `ood_pow`    | out-of-domain     | elided              |
/// | `p3-batch-stark` | `ood_pow`    | out-of-domain     | always              |
/// | `p3-batch-stark` | `lookup_pow` | lookup-challenge  | when the phase runs |
/// | `p3-fri-pcs`     | `batch_pow`  | batch-combination | elided              |
/// | `p3-fri`         | `commit_pow` | ldt-commit-phase  | elided              |
/// | `p3-fri`         | `query_pow`  | ldt-query-phase   | elided              |
///
/// Protocols absent from this table grind without a security model that reads the same numbers back.
///
/// | protocol                     | steps it grinds                                              |
/// | ---------------------------- | ------------------------------------------------------------ |
/// | `p3-circle-pcs`              | `commit_pow`, `query_pow`                                    |
/// | `p3-whir`                    | `sumcheck_pow`, `query_pow`, `final_query_pow`               |
/// | `p3-whir-hvzk`               | `zk_sumcheck_pow`, `base_pow`                                |
/// | `p3-stir`                    | `folding_pow`, `query_pow`, `final_folding_pow`, `final_pow` |
/// | `p3-sumcheck-quadratic`      | `round_pow`                                                  |
/// | `p3-sumcheck-generic-degree` | `round_pow`                                                  |
///
/// None of them declares its sites in bits, so no check can read the credited difficulty back out.
pub const GRINDING_VOCABULARY: [GrindingStep; 6] = [
    GrindingStep {
        protocol: "p3-uni-stark",
        label: "ood_pow",
        site: GrindingSite::OutOfDomain,
        zero_bits: ZeroBitConvention::Elided,
    },
    GrindingStep {
        protocol: "p3-batch-stark",
        label: "ood_pow",
        site: GrindingSite::OutOfDomain,
        zero_bits: ZeroBitConvention::Always,
    },
    GrindingStep {
        protocol: "p3-batch-stark",
        label: "lookup_pow",
        site: GrindingSite::LookupChallenge,
        zero_bits: ZeroBitConvention::WhenPhaseRuns,
    },
    GrindingStep {
        protocol: "p3-fri-pcs",
        label: "batch_pow",
        site: GrindingSite::BatchCombination,
        zero_bits: ZeroBitConvention::Elided,
    },
    GrindingStep {
        protocol: "p3-fri",
        label: "commit_pow",
        site: GrindingSite::LdtCommitPhase,
        zero_bits: ZeroBitConvention::Elided,
    },
    GrindingStep {
        protocol: "p3-fri",
        label: "query_pow",
        site: GrindingSite::LdtQueryPhase,
        zero_bits: ZeroBitConvention::Elided,
    },
];

/// The vocabulary row for one protocol's step label, or `None` when the pair
/// names no site this crate models.
#[must_use]
pub fn grinding_step(protocol: &str, label: &str) -> Option<&'static GrindingStep> {
    GRINDING_VOCABULARY
        .iter()
        .find(|step| step.protocol == protocol && step.label == label)
}

/// One proof-of-work step read back out of an interaction pattern.
///
/// `protocol` is the name that protocol binds into its transcript seed.
///
/// That is what makes `label` unambiguous, since four crates label a step `"query_pow"`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RecordedGrind {
    /// Protocol name bound into the transcript seed the step was read from.
    pub protocol: &'static str,
    /// Label the step carries.
    pub label: &'static str,
    /// Difficulty in bits the step demands.
    pub bits: usize,
}

impl RecordedGrind {
    /// One recorded step.
    #[must_use]
    pub const fn new(protocol: &'static str, label: &'static str, bits: usize) -> Self {
        Self {
            protocol,
            label,
            bits,
        }
    }
}

/// The difficulty the security model credits at every site of one parameter set.
///
/// A FRI-backed protocol splits its grinding across two carriers, and this joins them.
///
/// ```text
///     the STARK's own sites  ->  out-of-domain, batch-combination, lookup-challenge
///     the FRI regime         ->  ldt-commit-phase, ldt-query-phase
/// ```
///
/// The composite boosts the first three.
///
/// The low-degree test folds the last two into its own terms.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GrindingBudget {
    /// Bits per site, in the canonical site order.
    bits: [usize; GrindingSite::ALL.len()],
}

impl GrindingBudget {
    /// No grinding credited at any site.
    pub const NONE: Self = Self {
        bits: [0; GrindingSite::ALL.len()],
    };

    /// The three sites a STARK declares, with the low-degree test's own two at zero.
    ///
    /// Fill those two in afterwards from the low-degree test's regime.
    #[must_use]
    pub const fn from_sites(sites: &GrindingSites) -> Self {
        let mut bits = [0; GrindingSite::ALL.len()];
        bits[GrindingSite::OutOfDomain.index()] = sites.out_of_domain;
        bits[GrindingSite::BatchCombination.index()] = sites.batch_combination;
        bits[GrindingSite::LookupChallenge.index()] = sites.lookup_challenge;
        Self { bits }
    }

    /// Fill in the two sites a FRI low-degree test owns.
    ///
    /// The commit-phase grind repeats once per folding round, at one difficulty.
    ///
    /// The commit-phase error term credits that difficulty once, which is the conservative reading.
    #[must_use]
    pub const fn with_fri(mut self, ldt: &FriRegime) -> Self {
        self.bits[GrindingSite::LdtCommitPhase.index()] = ldt.commit_pow_bits;
        self.bits[GrindingSite::LdtQueryPhase.index()] = ldt.query_pow_bits;
        self
    }

    /// Bits credited at one site.
    #[must_use]
    pub const fn bits(&self, site: GrindingSite) -> usize {
        self.bits[site.index()]
    }

    /// Check the difficulties a transcript records against the ones this budget credits.
    ///
    /// `protocols` names the transcripts `recorded` was read from.
    ///
    /// Without it, a step the model credits and no transcript describes would be
    /// indistinguishable from a protocol that is simply not in the stack.
    ///
    /// A grind repeated once per round appears once per round in `recorded`.
    ///
    /// Every occurrence has to carry the credited difficulty.
    ///
    /// # Errors
    ///
    /// Returns the first disagreement found, naming the site and both numbers.
    pub fn check(
        &self,
        protocols: &[&str],
        recorded: &[RecordedGrind],
    ) -> Result<(), GrindingMismatch> {
        // Every recorded step must belong to a listed protocol and name a known site.
        //
        // Either failure means the caller and this table disagree about what
        // the stack is, which would silently exempt a site from the comparison.
        for grind in recorded {
            if !protocols.contains(&grind.protocol) {
                return Err(GrindingMismatch::UnlistedProtocol {
                    protocol: grind.protocol,
                    label: grind.label,
                    recorded: grind.bits,
                });
            }
            if grinding_step(grind.protocol, grind.label).is_none() {
                return Err(GrindingMismatch::UnknownStep {
                    protocol: grind.protocol,
                    label: grind.label,
                    recorded: grind.bits,
                });
            }
        }

        // Every step a listed protocol can describe must agree with the model.
        for step in GRINDING_VOCABULARY
            .iter()
            .filter(|step| protocols.contains(&step.protocol))
        {
            let credited = self.bits(step.site);

            // Presence is per site.
            // The difficulty is checked on each occurrence.
            let mut described = false;
            for grind in recorded
                .iter()
                .filter(|grind| grind.protocol == step.protocol && grind.label == step.label)
            {
                described = true;
                if grind.bits != credited {
                    return Err(GrindingMismatch::Difficulty {
                        protocol: step.protocol,
                        label: step.label,
                        site: step.site,
                        credited,
                        recorded: grind.bits,
                    });
                }
            }

            // What presence has to look like is the protocol's zero-bit convention.
            match (step.zero_bits, described) {
                // Conventions that describe the step whatever the difficulty.
                (ZeroBitConvention::Always | ZeroBitConvention::WhenPhaseRuns, true) => {}
                // An elided convention describes the step exactly when bits are credited.
                (ZeroBitConvention::Elided, true) if credited > 0 => {}
                (ZeroBitConvention::Elided, true) => {
                    return Err(GrindingMismatch::Unexpected {
                        protocol: step.protocol,
                        label: step.label,
                        site: step.site,
                    });
                }
                // A phase that did not run pays nothing and is credited nothing.
                (ZeroBitConvention::WhenPhaseRuns, false) => {}
                // Nothing credited, nothing described.
                (ZeroBitConvention::Elided, false) if credited == 0 => {}
                // Bits credited that the transcript never demands.
                (ZeroBitConvention::Elided | ZeroBitConvention::Always, false) => {
                    return Err(GrindingMismatch::Missing {
                        protocol: step.protocol,
                        label: step.label,
                        site: step.site,
                        credited,
                    });
                }
            }
        }

        Ok(())
    }
}

/// A way the recorded difficulties and the credited ones can disagree.
///
/// Every variant names the site, and both numbers wherever there are two.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GrindingMismatch {
    /// A step is described at a difficulty the model does not credit.
    Difficulty {
        /// Protocol whose transcript describes the step.
        protocol: &'static str,
        /// Label the step carries.
        label: &'static str,
        /// Site both halves are talking about.
        site: GrindingSite,
        /// Bits the security model credits.
        credited: usize,
        /// Bits the transcript demands.
        recorded: usize,
    },
    /// The model credits bits at a site the transcript never grinds for.
    ///
    /// This is the overstating direction.
    ///
    /// The reported level then includes work no prover pays and no verifier checks.
    Missing {
        /// Protocol whose transcript omits the step.
        protocol: &'static str,
        /// Label the omitted step would carry.
        label: &'static str,
        /// Site the model credits.
        site: GrindingSite,
        /// Bits the security model credits.
        credited: usize,
    },
    /// A zero-bit step is described where the protocol's convention elides it.
    Unexpected {
        /// Protocol whose transcript describes the step.
        protocol: &'static str,
        /// Label the step carries.
        label: &'static str,
        /// Site the step names.
        site: GrindingSite,
    },
    /// A described step whose `(protocol, label)` pair names no modeled site.
    UnknownStep {
        /// Protocol whose transcript describes the step.
        protocol: &'static str,
        /// Label the step carries.
        label: &'static str,
        /// Bits the transcript demands.
        recorded: usize,
    },
    /// A described step from a protocol the caller did not list.
    UnlistedProtocol {
        /// Protocol whose transcript describes the step.
        protocol: &'static str,
        /// Label the step carries.
        label: &'static str,
        /// Bits the transcript demands.
        recorded: usize,
    },
}

impl Display for GrindingMismatch {
    /// Name the site, the step it was read from, and both numbers.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Difficulty {
                protocol,
                label,
                site,
                credited,
                recorded,
            } => write!(
                f,
                "{site} grinding disagrees: the security model credits {credited} bits, \
                 `{protocol}`'s `{label}` step demands {recorded}",
            ),
            Self::Missing {
                protocol,
                label,
                site,
                credited,
            } => write!(
                f,
                "{site} grinding disagrees: the security model credits {credited} bits, \
                 `{protocol}` describes no `{label}` step, so 0 are demanded",
            ),
            Self::Unexpected {
                protocol,
                label,
                site,
            } => write!(
                f,
                "{site} grinding disagrees: the security model credits 0 bits, \
                 and `{protocol}` describes a zero-bit `{label}` step it should elide",
            ),
            Self::UnknownStep {
                protocol,
                label,
                recorded,
            } => write!(
                f,
                "`{protocol}`'s `{label}` step demands {recorded} bits at a site \
                 `GRINDING_VOCABULARY` does not map",
            ),
            Self::UnlistedProtocol {
                protocol,
                label,
                recorded,
            } => write!(
                f,
                "`{protocol}`'s `{label}` step demands {recorded} bits, \
                 and `{protocol}` is not among the protocols checked",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boost_adds_pow_bits_and_zero_is_neutral() {
        let error = ErrorBits::from_log2(80.0);
        assert!((boost(error, 0).bits() - 80.0).abs() < 1e-12);
        assert!((boost(error, 16).bits() - 96.0).abs() < 1e-12);
    }

    #[test]
    fn default_sites_are_neutral() {
        assert_eq!(GrindingSites::default(), GrindingSites::NONE);
    }

    /// The FRI regime this crate's own tests grind against.
    const fn regime(commit_pow_bits: usize, query_pow_bits: usize) -> FriRegime {
        FriRegime {
            log_blowup: 1,
            num_queries: 64,
            log_final_poly_len: 0,
            max_log_arity: 3,
            commit_pow_bits,
            query_pow_bits,
        }
    }

    /// The full FRI-backed uni-STARK stack, in transcript nesting order.
    const UNI_STARK_STACK: [&str; 3] = ["p3-uni-stark", "p3-fri-pcs", "p3-fri"];

    #[test]
    fn every_site_appears_in_the_vocabulary_exactly_as_often_as_a_protocol_names_it() {
        // Invariant: the table maps every site, so no site escapes the comparison.
        for site in GrindingSite::ALL {
            assert!(
                GRINDING_VOCABULARY.iter().any(|step| step.site == site),
                "{site} is modeled but no protocol's step maps to it"
            );
        }

        // No two rows share a key, or the lookup would silently pick one.
        for (index, left) in GRINDING_VOCABULARY.iter().enumerate() {
            for right in &GRINDING_VOCABULARY[index + 1..] {
                assert!(
                    (left.protocol, left.label) != (right.protocol, right.label),
                    "`{}`'s `{}` has two vocabulary rows",
                    left.protocol,
                    left.label
                );
            }
        }
    }

    #[test]
    fn the_same_label_in_two_protocols_names_one_site_under_two_conventions() {
        // Fixture state: `ood_pow` is the label both STARK front-ends use.
        let uni = grinding_step("p3-uni-stark", "ood_pow").expect("uni-STARK grinds before zeta");
        let batch =
            grinding_step("p3-batch-stark", "ood_pow").expect("batch-STARK grinds before zeta");

        // The site is a property of the phase, so the two agree on it.
        assert_eq!(uni.site, batch.site);
        assert_eq!(uni.site, GrindingSite::OutOfDomain);

        // The convention is a property of the proof layout, so they differ.
        assert_eq!(uni.zero_bits, ZeroBitConvention::Elided);
        assert_eq!(batch.zero_bits, ZeroBitConvention::Always);
    }

    #[test]
    fn each_site_boosts_the_report_term_its_own_module_labels() {
        // The report label is the third name of the same phase.
        assert_eq!(GrindingSite::OutOfDomain.report_label(), DEEP_LABEL);
        assert_eq!(GrindingSite::BatchCombination.report_label(), BATCH_LABEL);
        assert_eq!(GrindingSite::LookupChallenge.report_label(), LOGUP_LABEL);
        assert_eq!(
            GrindingSite::LdtCommitPhase.report_label(),
            LDT_COMMIT_LABEL
        );
        assert_eq!(GrindingSite::LdtQueryPhase.report_label(), LDT_QUERY_LABEL);
    }

    #[test]
    fn the_budget_reads_back_both_halves_of_the_model() {
        // Fixture state: a distinct value per site, so a swap is visible.
        let sites = GrindingSites {
            out_of_domain: 3,
            batch_combination: 5,
            lookup_challenge: 7,
        };
        let budget = GrindingBudget::from_sites(&sites).with_fri(&regime(11, 13));

        assert_eq!(budget.bits(GrindingSite::OutOfDomain), 3);
        assert_eq!(budget.bits(GrindingSite::BatchCombination), 5);
        assert_eq!(budget.bits(GrindingSite::LookupChallenge), 7);
        assert_eq!(budget.bits(GrindingSite::LdtCommitPhase), 11);
        assert_eq!(budget.bits(GrindingSite::LdtQueryPhase), 13);

        // `from_sites` alone leaves the low-degree test's own sites at zero.
        let without_ldt = GrindingBudget::from_sites(&sites);
        assert_eq!(without_ldt.bits(GrindingSite::LdtCommitPhase), 0);
        assert_eq!(without_ldt.bits(GrindingSite::LdtQueryPhase), 0);
    }

    #[test]
    fn a_stack_that_grinds_exactly_what_the_model_credits_agrees() {
        let sites = GrindingSites {
            out_of_domain: 8,
            batch_combination: 10,
            ..GrindingSites::NONE
        };
        let budget = GrindingBudget::from_sites(&sites).with_fri(&regime(4, 16));

        // The commit-phase grind repeats once per folding round at one difficulty.
        let recorded = [
            RecordedGrind::new("p3-uni-stark", "ood_pow", 8),
            RecordedGrind::new("p3-fri-pcs", "batch_pow", 10),
            RecordedGrind::new("p3-fri", "commit_pow", 4),
            RecordedGrind::new("p3-fri", "commit_pow", 4),
            RecordedGrind::new("p3-fri", "query_pow", 16),
        ];

        budget
            .check(&UNI_STARK_STACK, &recorded)
            .expect("the two halves agree");
    }

    #[test]
    fn a_credited_site_the_transcript_never_grinds_for_is_the_overstating_direction() {
        // Fixture state: the model credits ten batch bits, the transcript demands none.
        let budget = GrindingBudget::from_sites(&GrindingSites {
            batch_combination: 10,
            ..GrindingSites::NONE
        })
        .with_fri(&regime(0, 0));

        let mismatch = budget
            .check(&UNI_STARK_STACK, &[])
            .expect_err("credited bits nobody pays must be reported");

        assert_eq!(
            mismatch,
            GrindingMismatch::Missing {
                protocol: "p3-fri-pcs",
                label: "batch_pow",
                site: GrindingSite::BatchCombination,
                credited: 10,
            }
        );
    }

    #[test]
    fn a_difficulty_disagreement_names_the_site_and_both_numbers() {
        let budget = GrindingBudget::from_sites(&GrindingSites::NONE).with_fri(&regime(0, 16));
        let recorded = [RecordedGrind::new("p3-fri", "query_pow", 20)];

        let mismatch = budget
            .check(&["p3-fri"], &recorded)
            .expect_err("20 recorded against 16 credited must be reported");

        assert_eq!(
            mismatch,
            GrindingMismatch::Difficulty {
                protocol: "p3-fri",
                label: "query_pow",
                site: GrindingSite::LdtQueryPhase,
                credited: 16,
                recorded: 20,
            }
        );
        // The message is the deliverable, so it carries the site and both numbers.
        let rendered = alloc::format!("{mismatch}");
        assert!(rendered.contains("ldt-query-phase"), "{rendered}");
        assert!(rendered.contains("16"), "{rendered}");
        assert!(rendered.contains("20"), "{rendered}");
    }

    #[test]
    fn the_always_convention_requires_a_zero_bit_step_the_elided_one_forbids() {
        let budget = GrindingBudget::from_sites(&GrindingSites::NONE);

        // batch-STARK always describes the step, so omitting it is a disagreement.
        assert_eq!(
            budget
                .check(&["p3-batch-stark"], &[])
                .expect_err("an always-described step cannot be absent"),
            GrindingMismatch::Missing {
                protocol: "p3-batch-stark",
                label: "ood_pow",
                site: GrindingSite::OutOfDomain,
                credited: 0,
            }
        );

        // Described at zero bits, it agrees.
        budget
            .check(
                &["p3-batch-stark"],
                &[RecordedGrind::new("p3-batch-stark", "ood_pow", 0)],
            )
            .expect("a zero-bit step is what the convention calls for");

        // uni-STARK elides at zero, so describing the step is a disagreement.
        assert_eq!(
            budget
                .check(
                    &["p3-uni-stark"],
                    &[RecordedGrind::new("p3-uni-stark", "ood_pow", 0)],
                )
                .expect_err("an elided step cannot be described at zero bits"),
            GrindingMismatch::Unexpected {
                protocol: "p3-uni-stark",
                label: "ood_pow",
                site: GrindingSite::OutOfDomain,
            }
        );
    }

    #[test]
    fn a_lookup_phase_that_does_not_run_is_neither_paid_for_nor_credited() {
        // Fixture state: a batch with no lookups describes no `lookup_pow` step.
        //
        //     no lookups  ->  no `lookup_pow` step
        //                 ->  `logup::security_term` returns `None`
        //                 ->  no term for `lookup_challenge` to boost
        let budget = GrindingBudget::from_sites(&GrindingSites {
            lookup_challenge: 20,
            ..GrindingSites::NONE
        });

        budget
            .check(
                &["p3-batch-stark"],
                &[RecordedGrind::new("p3-batch-stark", "ood_pow", 0)],
            )
            .expect("a phase that does not run credits nothing");

        // Once the phase runs, the difficulty has to be the credited one.
        assert_eq!(
            budget
                .check(
                    &["p3-batch-stark"],
                    &[
                        RecordedGrind::new("p3-batch-stark", "ood_pow", 0),
                        RecordedGrind::new("p3-batch-stark", "lookup_pow", 8),
                    ],
                )
                .expect_err("8 recorded against 20 credited must be reported"),
            GrindingMismatch::Difficulty {
                protocol: "p3-batch-stark",
                label: "lookup_pow",
                site: GrindingSite::LookupChallenge,
                credited: 20,
                recorded: 8,
            }
        );
    }

    #[test]
    fn a_step_outside_the_checked_stack_is_reported_rather_than_ignored() {
        let budget = GrindingBudget::NONE;

        // WHIR's own query grind names a site this table does not map.
        assert_eq!(
            budget
                .check(
                    &["p3-whir"],
                    &[RecordedGrind::new("p3-whir", "query_pow", 16)]
                )
                .expect_err("an unmapped site must not pass silently"),
            GrindingMismatch::UnknownStep {
                protocol: "p3-whir",
                label: "query_pow",
                recorded: 16,
            }
        );

        // A mapped step from a protocol the caller forgot to list is reported too.
        assert_eq!(
            budget
                .check(
                    &["p3-fri"],
                    &[RecordedGrind::new("p3-fri-pcs", "batch_pow", 4)]
                )
                .expect_err("an unlisted protocol must not pass silently"),
            GrindingMismatch::UnlistedProtocol {
                protocol: "p3-fri-pcs",
                label: "batch_pow",
                recorded: 4,
            }
        );
    }
}
