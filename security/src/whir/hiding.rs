//! Diagnostic ideal-IOP soundness accounting for the HVZK-WHIR base case.
//!
//! The componentwise heterogeneous extension of Construction 7.2 combines one
//! randomized source code and a chronological family of randomized mask codes
//! with one shared challenge. Its two RBR error coordinates have different
//! composition rules:
//!
//! - component MCA errors add, as do the shared-challenge MCA and list terms;
//! - candidate-list sizes multiply, so their base-two logarithms add;
//! - query miss probabilities take a maximum, so their security bits take a
//!   minimum;
//! - the final query PoW is shared by the whole query round and, under the
//!   Fiat--Shamir grinding model, is added once.
//!
//! This module deliberately reports those terms without changing parameter
//! selection. It is not an end-to-end security claim: it excludes the earlier
//! masked-sumcheck and code-switch reductions, Merkle hiding and binding,
//! Fiat--Shamir compilation, random-oracle extraction, hash collisions, and
//! simulation of the commitment view. In particular, a conditional or
//! approximate number must not be presented as proven security.

use alloc::vec::Vec;

use p3_field::Field;
use serde::Serialize;

use crate::assumption::SecurityAssumption;

/// How much theorem support a reported bound has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum HidingBoundClassification {
    /// The displayed component or ideal-IOP composition follows from a theorem.
    ///
    /// This label is local to the term and its documented branch premise. For
    /// example, a query-miss term is proven when the oracle is at least the
    /// reported radius from its code; that does not upgrade a conditional
    /// close-word list or MCA branch in the Capacity regime. Likewise, a
    /// proven aggregate covers only the ideal-IOP composition reported here,
    /// not the reductions and compiler steps excluded by this module.
    Proven,
    /// The term depends on an additional coding or compiler assumption.
    ///
    /// This is used for the capacity-rate coding conjectures and for the
    /// post-PoW query estimate under the Fiat--Shamir grinding model.
    Conditional,
    /// The term is a numerical approximation, not a conservative theorem bound.
    ///
    /// This is used for the current Johnson MCA helper, which retains only the
    /// dominant finite-parameter term.
    Approximation,
    /// No accepted numerical bound is encoded for this geometry.
    Unavailable,
}

/// A probability upper bound represented as `-log2(probability)`.
///
/// `bits` is `None` exactly when the term is unavailable. Available values
/// are capped below at zero because a probability bound larger than one says
/// nothing. `vacuous` records when that cap was needed; it is a certification
/// failure, not evidence of an attack.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct HidingErrorBound {
    /// Security bits, or `None` if no numerical term is available.
    pub bits: Option<f64>,
    /// The theorem status of this term.
    pub classification: HidingBoundClassification,
    /// Whether the underlying probability upper bound is at least one.
    pub vacuous: bool,
}

impl HidingErrorBound {
    fn available(bits: f64, classification: HidingBoundClassification) -> Self {
        let vacuous = bits <= 0.;
        Self {
            bits: Some(bits.max(0.)),
            classification,
            vacuous,
        }
    }

    const fn unavailable() -> Self {
        Self {
            bits: None,
            classification: HidingBoundClassification::Unavailable,
            vacuous: false,
        }
    }

    /// Returns bits only for a non-vacuous theorem-backed term in its
    /// documented branch.
    ///
    /// This does not assert that sibling terms, a selected coding regime, or
    /// the compiled PCS are proven end to end.
    #[must_use]
    pub const fn proven_bits(&self) -> Option<f64> {
        match (self.classification, self.bits, self.vacuous) {
            (HidingBoundClassification::Proven, bits, false) => bits,
            _ => None,
        }
    }
}

/// A candidate-list size upper bound represented as `log2(list_size)`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct HidingListBound {
    /// Base-two logarithm of the list-size bound.
    pub log2_size: Option<f64>,
    /// The theorem status of the list-size bound.
    pub classification: HidingBoundClassification,
}

/// The role of one randomized code in Construction 7.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum HidingCodeRole {
    /// The folded terminal witness oracle.
    Source,
    /// A sumcheck-mask group, numbered in protocol order.
    SumcheckMask {
        /// Sumcheck batch index.
        round: usize,
    },
    /// A code-switch-mask group.
    CodeSwitchMask {
        /// Code-switch round index.
        round: usize,
    },
}

/// Security-relevant geometry of one randomized code component.
///
/// Protocol crates assemble these values from their runtime configuration and
/// pass them to [`hiding_base_case_security_report`]. Keeping this input free
/// of protocol configuration types preserves the dependency direction from a
/// protocol crate to `p3-security`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HidingCodeGeometry {
    role: HidingCodeRole,
    message_len: usize,
    randomness_len: usize,
    domain_size: usize,
    width: usize,
    requested_queries: usize,
}

impl HidingCodeGeometry {
    /// Creates the security-relevant geometry for one randomized code.
    ///
    /// # Panics
    ///
    /// Panics unless `domain_size` is a power of two, the randomized
    /// dimension fits in that domain, and `width` is nonzero and can be
    /// doubled for the paired old/fresh interleaving.
    #[must_use]
    pub const fn new(
        role: HidingCodeRole,
        message_len: usize,
        randomness_len: usize,
        domain_size: usize,
        width: usize,
        requested_queries: usize,
    ) -> Self {
        assert!(
            domain_size.is_power_of_two(),
            "the codeword domain size must be a power of two"
        );
        assert!(
            randomness_len <= domain_size && message_len <= domain_size - randomness_len,
            "the randomized-code dimension must fit in the codeword domain"
        );
        assert!(width > 0, "the interleaving width must be nonzero");
        assert!(
            width <= usize::MAX / 2,
            "the interleaving width must be safe to double"
        );
        Self {
            role,
            message_len,
            randomness_len,
            domain_size,
            width,
            requested_queries,
        }
    }
}

/// Geometry and ideal-IOP bounds for one randomized code component.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct HidingCodeSecurityReport {
    /// The component's protocol role.
    pub role: HidingCodeRole,
    /// Number of message coefficients in one lane.
    pub message_len: usize,
    /// Number of randomizing coefficients in one lane.
    pub randomness_len: usize,
    /// Actual randomized-code dimension, `message_len + randomness_len`.
    pub randomized_dimension: usize,
    /// Next-power-of-two envelope used for conservative RS bounds.
    pub dyadic_dimension: usize,
    /// Codeword length.
    pub domain_size: usize,
    /// Exact randomized-code rate, `(message_len + randomness_len) / domain_size`.
    pub randomized_rate: f64,
    /// Logarithm of the conservative dyadic-envelope inverse rate.
    ///
    /// This is unavailable when the dyadic envelope fills the domain: none
    /// of the configured proximity regimes supplies a positive radius then.
    pub analysis_log_inv_rate: Option<usize>,
    /// Proximity radius selected at the dyadic-envelope rate.
    pub analysis_radius: Option<f64>,
    /// Number of lanes sharing one row-query schedule.
    pub width: usize,
    /// Number of row queries requested by the derived configuration.
    pub requested_queries: usize,
    /// Number of distinct rows the sampler actually opens.
    ///
    /// The sampler saturates at the codeword length.
    pub effective_queries: usize,
    /// Conservative query-miss error before the shared final PoW.
    ///
    /// At saturation this keeps the IID expression instead of reporting the
    /// exact zero miss probability. The sampling bound is proven under the
    /// premise that the pre-query component is at least `analysis_radius`
    /// from its code. Under `CapacityBound`, the separate list/MCA terms
    /// remain conditional; this field does not promote the full regime to
    /// proven.
    pub query_miss: HidingErrorBound,
    /// MCA error for this component. Interleaved groups include the
    /// conservative factor equal to `width`.
    pub mca: HidingErrorBound,
    /// Candidate-list bound for the old/fresh pair. Its interleaving is two
    /// for the source and `2 * width` for a mask group.
    pub paired_list: HidingListBound,
}

/// The shared query-round accounting.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct HidingQueryRoundReport {
    /// Source query-miss bits before PoW.
    pub source_before_pow_bits: Option<f64>,
    /// Mask query-miss bits in chronological group order, before PoW.
    pub masks_before_pow_bits: Vec<Option<f64>>,
    /// Minimum component bits, corresponding to the maximum miss probability.
    pub weakest_before_pow_bits: Option<f64>,
    /// Ideal-IOP query coordinate before any Fiat--Shamir grinding credit.
    ///
    /// This composes the proven far-word sampling branches. It does not
    /// classify the corresponding close-word list/MCA branches.
    pub before_pow: HidingErrorBound,
    /// The one final grind shared by all source and mask queries.
    pub shared_pow_bits: usize,
    /// Source query estimate after applying the shared grind once.
    pub source_with_pow: HidingErrorBound,
    /// Weakest query estimate after applying the shared grind once.
    ///
    /// This is compiler-conditional when `shared_pow_bits > 0`.
    pub with_pow: HidingErrorBound,
}

/// Shared-`gamma` combination-round accounting.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct HidingGammaRoundReport {
    /// Sum of the source and chronological mask-group MCA errors.
    pub mca_sum: HidingErrorBound,
    /// Logarithm of the product of all old/fresh candidate-list bounds.
    pub product_list: HidingListBound,
    /// Product-list size divided by the exact field order.
    pub product_list_over_field: HidingErrorBound,
    /// Sum of the MCA and product-list/field error terms.
    pub combined: HidingErrorBound,
}

/// Non-enforcing report for the ideal interactive hiding base case.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[non_exhaustive]
pub struct HidingBaseCaseSecurityReport {
    /// User-requested security level. The report does not assert it is met.
    pub target_bits: usize,
    /// Coding regime used for all component radii and bounds.
    pub assumption: SecurityAssumption,
    /// Numerical estimate of the exact `log2(EF::order())`.
    ///
    /// This retains the fractional part for diagnostics but is not used as a
    /// theorem bound, because `libm::log2` does not provide directed rounding.
    pub field_order_log2_estimate: f64,
    /// Rigorous integer lower bound `floor(log2(EF::order()))`.
    ///
    /// All field-dependent probability bounds use this value.
    pub field_order_log2_floor: usize,
    /// Folded terminal source-code report.
    pub source: HidingCodeSecurityReport,
    /// Mask groups in chronological commitment order.
    pub mask_groups: Vec<HidingCodeSecurityReport>,
    /// Query-round report.
    pub query_round: HidingQueryRoundReport,
    /// Shared-challenge combination-round report.
    pub gamma_round: HidingGammaRoundReport,
    /// The weaker of the two Construction 7.2 RBR error coordinates.
    ///
    /// This is `min(gamma_round, query_round.before_pow)` in security-bit
    /// notation, not `-log2` of a union of the coordinate errors. Its
    /// classification conservatively combines both coordinates' provenance,
    /// even when the less-proven coordinate is not numerically binding,
    /// because certifying the reported RBR level requires both bounds. It
    /// deliberately excludes `query_round.with_pow`, which assumes the
    /// Fiat--Shamir grinding model, and does not compose with other WHIR or
    /// compiler errors. A `Proven` classification here says only that both
    /// coordinates and this ideal-IOP RBR composition are theorem-backed
    /// within that scope.
    pub round_by_round: HidingErrorBound,
}

/// Reports the heterogeneous HVZK-WHIR base case's ideal-IOP RBR terms.
///
/// The caller supplies protocol-derived source and mask geometry. The field
/// order and every security term are derived here so all users share one
/// implementation of the accounting. This is diagnostic only: it neither
/// rejects a configuration nor certifies a compiled PCS or the preceding
/// hiding reductions.
#[must_use]
pub fn hiding_base_case_security_report<EF: Field>(
    target_bits: usize,
    assumption: SecurityAssumption,
    source: HidingCodeGeometry,
    mask_groups: Vec<HidingCodeGeometry>,
    shared_pow_bits: usize,
) -> HidingBaseCaseSecurityReport {
    build_hiding_base_case_security_report(
        target_bits,
        assumption,
        log2_order_estimate::<EF>(),
        EF::order().bits() as usize - 1,
        source,
        mask_groups,
        shared_pow_bits,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_hiding_base_case_security_report(
    target_bits: usize,
    assumption: SecurityAssumption,
    field_order_log2_estimate: f64,
    field_order_log2_floor: usize,
    source: HidingCodeGeometry,
    mask_groups: Vec<HidingCodeGeometry>,
    shared_pow_bits: usize,
) -> HidingBaseCaseSecurityReport {
    let source = code_report(source, assumption, field_order_log2_floor);
    let mask_groups: Vec<_> = mask_groups
        .into_iter()
        .map(|geometry| code_report(geometry, assumption, field_order_log2_floor))
        .collect();

    let source_before_pow_bits = source.query_miss.bits;
    let masks_before_pow_bits: Vec<Option<f64>> = mask_groups
        .iter()
        .map(|group| group.query_miss.bits)
        .collect();
    let weakest_before_pow_bits = minimum_available(
        core::iter::once(source_before_pow_bits).chain(masks_before_pow_bits.iter().copied()),
    );
    let before_pow = weakest_before_pow_bits.map_or_else(HidingErrorBound::unavailable, |bits| {
        HidingErrorBound::available(bits, HidingBoundClassification::Proven)
    });
    let pow_classification = if shared_pow_bits == 0 {
        HidingBoundClassification::Proven
    } else {
        HidingBoundClassification::Conditional
    };
    let source_with_pow = source_before_pow_bits
        .map_or_else(HidingErrorBound::unavailable, |bits| {
            HidingErrorBound::available(bits + shared_pow_bits as f64, pow_classification)
        });
    let with_pow = weakest_before_pow_bits.map_or_else(HidingErrorBound::unavailable, |bits| {
        HidingErrorBound::available(bits + shared_pow_bits as f64, pow_classification)
    });
    let query_round = HidingQueryRoundReport {
        source_before_pow_bits,
        masks_before_pow_bits,
        weakest_before_pow_bits,
        before_pow,
        shared_pow_bits,
        source_with_pow,
        with_pow,
    };

    let mca_bounds = core::iter::once(&source)
        .chain(mask_groups.iter())
        .map(|component| component.mca.bits);
    let mca_classification = mca_classification(assumption);
    let mca_sum = sum_available_probability_bits(mca_bounds)
        .map_or_else(HidingErrorBound::unavailable, |bits| {
            HidingErrorBound::available(bits, mca_classification)
        });

    let list_classification = list_classification(assumption);
    let product_list_log2 = sum_available(
        core::iter::once(&source)
            .chain(mask_groups.iter())
            .map(|component| component.paired_list.log2_size),
    );
    let product_list = HidingListBound {
        log2_size: product_list_log2,
        classification: if product_list_log2.is_some() {
            list_classification
        } else {
            HidingBoundClassification::Unavailable
        },
    };
    let product_list_over_field =
        product_list_log2.map_or_else(HidingErrorBound::unavailable, |list_bits| {
            HidingErrorBound::available(
                field_order_log2_floor as f64 - list_bits,
                list_classification,
            )
        });
    let gamma_classification = combine_classifications(
        mca_sum.classification,
        product_list_over_field.classification,
    );
    let gamma_combined =
        sum_available_probability_bits([mca_sum.bits, product_list_over_field.bits])
            .map_or_else(HidingErrorBound::unavailable, |bits| {
                HidingErrorBound::available(bits, gamma_classification)
            });
    let gamma_round = HidingGammaRoundReport {
        mca_sum,
        product_list,
        product_list_over_field,
        combined: gamma_combined,
    };

    let round_by_round_classification = combine_classifications(
        gamma_round.combined.classification,
        query_round.before_pow.classification,
    );
    let round_by_round =
        minimum_available([gamma_round.combined.bits, query_round.before_pow.bits])
            .map_or_else(HidingErrorBound::unavailable, |bits| {
                HidingErrorBound::available(bits, round_by_round_classification)
            });

    HidingBaseCaseSecurityReport {
        target_bits,
        assumption,
        field_order_log2_estimate,
        field_order_log2_floor,
        source,
        mask_groups,
        query_round,
        gamma_round,
        round_by_round,
    }
}

fn code_report(
    geometry: HidingCodeGeometry,
    assumption: SecurityAssumption,
    field_order_log2_floor: usize,
) -> HidingCodeSecurityReport {
    let HidingCodeGeometry {
        role,
        message_len,
        randomness_len,
        domain_size,
        width,
        requested_queries,
    } = geometry;
    let randomized_dimension = message_len + randomness_len;
    let dyadic_dimension = randomized_dimension.next_power_of_two();
    let log_degree = dyadic_dimension.ilog2() as usize;
    let raw_log_inv_rate = domain_size.ilog2() as usize - log_degree;
    let analysis_log_inv_rate = (raw_log_inv_rate > 0).then_some(raw_log_inv_rate);
    let one_minus_delta =
        analysis_log_inv_rate.map(|rate| libm::pow(2., assumption.log_1_delta(rate)));
    let analysis_radius = one_minus_delta.map(|one_minus_delta| 1. - one_minus_delta);
    let effective_queries = requested_queries.min(domain_size);

    let (query_miss, mca, paired_list) = analysis_log_inv_rate.map_or_else(
        || {
            (
                HidingErrorBound::unavailable(),
                HidingErrorBound::unavailable(),
                HidingListBound {
                    log2_size: None,
                    classification: HidingBoundClassification::Unavailable,
                },
            )
        },
        |log_inv_rate| {
            let one_minus_delta = one_minus_delta.expect("a positive-rate envelope has a radius");
            let query_bits = assumption.queries_error(log_inv_rate, effective_queries);
            let mca_bits =
                assumption.prox_gaps_error(log_degree, log_inv_rate, field_order_log2_floor, 2)
                    - libm::log2(width as f64);
            let paired_interleaving = 2 * width;
            (
                HidingErrorBound::available(query_bits, HidingBoundClassification::Proven),
                HidingErrorBound::available(mca_bits, mca_classification(assumption)),
                HidingListBound {
                    log2_size: Some(interleaved_list_bits(
                        assumption,
                        log_degree,
                        log_inv_rate,
                        paired_interleaving,
                        one_minus_delta,
                    )),
                    classification: list_classification(assumption),
                },
            )
        },
    );

    HidingCodeSecurityReport {
        role,
        message_len,
        randomness_len,
        randomized_dimension,
        dyadic_dimension,
        domain_size,
        randomized_rate: randomized_dimension as f64 / domain_size as f64,
        analysis_log_inv_rate,
        analysis_radius,
        width,
        requested_queries,
        effective_queries,
        query_miss,
        mca,
        paired_list,
    }
}

fn interleaved_list_bits(
    assumption: SecurityAssumption,
    log_degree: usize,
    log_inv_rate: usize,
    interleaving: usize,
    one_minus_delta: f64,
) -> f64 {
    let base = assumption.list_size_bits(log_degree, log_inv_rate);
    let crude = interleaving as f64 * base;

    // Lemma 3.13 of "Zero-Knowledge IOPPs for Constrained Interleaved
    // Codes" (ePrint 2026/391), recalling [GGR11], gives the
    // interleaving-stable alternative binomial(b + r, r) * L^r. Taking the
    // better of this and L^K is conservative because both are upper bounds.
    let rho = libm::pow(2., -(log_inv_rate as f64));
    let distance = 1. - rho;
    let delta = 1. - one_minus_delta;
    let gap = one_minus_delta - rho;
    let b = libm::ceil(delta / gap) as usize;
    let r = libm::ceil(libm::log2(distance / gap)) as usize;
    let stable = b
        .checked_add(r)
        .map_or(f64::INFINITY, |n| log2_binomial(n, r) + r as f64 * base);
    crude.min(stable)
}

fn log2_binomial(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (1..=k)
        .map(|i| libm::log2((n - k + i) as f64) - libm::log2(i as f64))
        .sum()
}

const fn mca_classification(assumption: SecurityAssumption) -> HidingBoundClassification {
    match assumption {
        SecurityAssumption::UniqueDecoding => HidingBoundClassification::Proven,
        SecurityAssumption::JohnsonBound => HidingBoundClassification::Approximation,
        SecurityAssumption::CapacityBound => HidingBoundClassification::Conditional,
    }
}

const fn list_classification(assumption: SecurityAssumption) -> HidingBoundClassification {
    match assumption {
        SecurityAssumption::UniqueDecoding | SecurityAssumption::JohnsonBound => {
            HidingBoundClassification::Proven
        }
        SecurityAssumption::CapacityBound => HidingBoundClassification::Conditional,
    }
}

const fn combine_classifications(
    left: HidingBoundClassification,
    right: HidingBoundClassification,
) -> HidingBoundClassification {
    match (left, right) {
        (HidingBoundClassification::Unavailable, _)
        | (_, HidingBoundClassification::Unavailable) => HidingBoundClassification::Unavailable,
        (HidingBoundClassification::Conditional, _)
        | (_, HidingBoundClassification::Conditional) => HidingBoundClassification::Conditional,
        (HidingBoundClassification::Approximation, _)
        | (_, HidingBoundClassification::Approximation) => HidingBoundClassification::Approximation,
        (HidingBoundClassification::Proven, HidingBoundClassification::Proven) => {
            HidingBoundClassification::Proven
        }
    }
}

/// Returns `-log2(sum_i 2^-bits_i)` without underflowing large exponents.
fn sum_probability_bits(bits: impl IntoIterator<Item = f64>) -> f64 {
    let bits: Vec<f64> = bits.into_iter().collect();
    let minimum = bits.iter().copied().fold(f64::INFINITY, f64::min);
    if minimum == f64::INFINITY {
        return minimum;
    }
    let scaled_sum: f64 = bits
        .into_iter()
        .map(|value| libm::pow(2., -(value - minimum)))
        .sum();
    minimum - libm::log2(scaled_sum)
}

fn sum_available_probability_bits(bits: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
    let bits: Option<Vec<_>> = bits.into_iter().collect();
    bits.map(sum_probability_bits)
}

fn sum_available(bits: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
    bits.into_iter()
        .try_fold(0., |sum, value| value.map(|value| sum + value))
}

fn minimum_available(bits: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
    bits.into_iter().try_fold(f64::INFINITY, |minimum, value| {
        value.map(|value| minimum.min(value))
    })
}

/// A fractional `log2` estimate of the exact field cardinality.
///
/// The top 53 integer bits are retained before conversion. The report exposes
/// this for diagnostics only; theorem bounds use the exact integer floor.
fn log2_order_estimate<EF: Field>() -> f64 {
    const RETAINED_BITS: u64 = 53;

    let order = EF::order();
    let order_bits = order.bits();
    let shift = order_bits.saturating_sub(RETAINED_BITS) as usize;
    let high = (&order >> shift)
        .to_u64_digits()
        .first()
        .copied()
        .expect("a field has nonzero order");
    shift as f64 + libm::log2(high as f64)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    const FIELD_LOG2_ESTIMATE: f64 = 128.75;
    const FIELD_LOG2_FLOOR: usize = 128;

    fn generic_report(
        assumption: SecurityAssumption,
        source: HidingCodeGeometry,
        mask_groups: Vec<HidingCodeGeometry>,
        shared_pow_bits: usize,
    ) -> HidingBaseCaseSecurityReport {
        build_hiding_base_case_security_report(
            100,
            assumption,
            FIELD_LOG2_ESTIMATE,
            FIELD_LOG2_FLOOR,
            source,
            mask_groups,
            shared_pow_bits,
        )
    }

    #[test]
    fn heterogeneous_report_composes_query_and_gamma_terms() {
        let source = HidingCodeGeometry::new(HidingCodeRole::Source, 2, 1, 16, 1, 4);
        let mask =
            HidingCodeGeometry::new(HidingCodeRole::SumcheckMask { round: 0 }, 1, 1, 16, 2, 3);
        let report = generic_report(SecurityAssumption::UniqueDecoding, source, vec![mask], 5);

        assert_eq!(report.field_order_log2_estimate, FIELD_LOG2_ESTIMATE);
        assert_eq!(report.field_order_log2_floor, FIELD_LOG2_FLOOR);
        assert_eq!(report.source.randomized_dimension, 3);
        assert_eq!(report.source.dyadic_dimension, 4);
        assert_eq!(report.source.analysis_log_inv_rate, Some(2));
        assert_eq!(report.mask_groups[0].analysis_log_inv_rate, Some(3));
        assert_eq!(
            report.mask_groups[0].role,
            HidingCodeRole::SumcheckMask { round: 0 }
        );

        let source_query = SecurityAssumption::UniqueDecoding.queries_error(2, 4);
        let mask_query = SecurityAssumption::UniqueDecoding.queries_error(3, 3);
        let weakest_query = source_query.min(mask_query);
        assert_eq!(
            report.query_round.source_before_pow_bits,
            Some(source_query)
        );
        assert_eq!(
            report.query_round.masks_before_pow_bits,
            vec![Some(mask_query)]
        );
        assert_eq!(
            report.query_round.weakest_before_pow_bits,
            Some(weakest_query)
        );
        assert_eq!(report.query_round.before_pow.bits, Some(weakest_query));
        assert_eq!(report.query_round.with_pow.bits, Some(weakest_query + 5.));
        assert_eq!(
            report.query_round.with_pow.classification,
            HidingBoundClassification::Conditional
        );

        let source_mca = SecurityAssumption::UniqueDecoding.prox_gaps_error(2, 2, 128, 2);
        let mask_mca = SecurityAssumption::UniqueDecoding.prox_gaps_error(1, 3, 128, 2) - 1.;
        assert_eq!(
            report.gamma_round.mca_sum.bits,
            Some(sum_probability_bits([source_mca, mask_mca]))
        );
        assert_eq!(report.gamma_round.product_list.log2_size, Some(0.));
        assert_eq!(
            report.gamma_round.combined.classification,
            HidingBoundClassification::Proven
        );
        assert_eq!(
            report.round_by_round.classification,
            HidingBoundClassification::Proven
        );
        assert_eq!(report.round_by_round.bits, Some(weakest_query));
        assert_eq!(report.round_by_round.proven_bits(), Some(weakest_query));
    }

    #[test]
    fn regime_classifications_propagate_without_overstatement() {
        let source = HidingCodeGeometry::new(HidingCodeRole::Source, 2, 0, 16, 1, 1);

        for (assumption, expected) in [
            (
                SecurityAssumption::UniqueDecoding,
                HidingBoundClassification::Proven,
            ),
            (
                SecurityAssumption::JohnsonBound,
                HidingBoundClassification::Approximation,
            ),
            (
                SecurityAssumption::CapacityBound,
                HidingBoundClassification::Conditional,
            ),
        ] {
            let report = generic_report(assumption, source, vec![], 0);

            assert_eq!(
                report.query_round.before_pow.classification,
                HidingBoundClassification::Proven
            );
            assert_eq!(report.gamma_round.combined.classification, expected);
            assert_eq!(report.round_by_round.classification, expected);
            assert_eq!(
                report.round_by_round.proven_bits().is_some(),
                expected == HidingBoundClassification::Proven
            );
        }
    }

    #[test]
    fn aggregate_provenance_includes_a_nonbinding_coordinate() {
        for classification in [
            HidingBoundClassification::Conditional,
            HidingBoundClassification::Approximation,
        ] {
            let gamma = HidingErrorBound::available(120., classification);
            let query = HidingErrorBound::available(80., HidingBoundClassification::Proven);
            let combined = combine_classifications(gamma.classification, query.classification);
            let round_by_round =
                HidingErrorBound::available(gamma.bits.unwrap().min(query.bits.unwrap()), combined);

            assert_eq!(round_by_round.bits, Some(80.));
            assert_eq!(round_by_round.classification, classification);
            assert_eq!(round_by_round.proven_bits(), None);
        }
    }

    #[test]
    fn proven_bits_rejects_vacuous_proven_terms() {
        let vacuous = HidingErrorBound::available(-1., HidingBoundClassification::Proven);
        let positive = HidingErrorBound::available(1., HidingBoundClassification::Proven);

        assert_eq!(vacuous.bits, Some(0.));
        assert!(vacuous.vacuous);
        assert_eq!(vacuous.proven_bits(), None);
        assert_eq!(positive.proven_bits(), Some(1.));
    }

    #[test]
    fn analysis_radius_comes_from_log_one_minus_delta() {
        for log_inv_rate in 1..=16 {
            for assumption in [
                SecurityAssumption::UniqueDecoding,
                SecurityAssumption::JohnsonBound,
                SecurityAssumption::CapacityBound,
            ] {
                let report = generic_report(
                    assumption,
                    HidingCodeGeometry::new(HidingCodeRole::Source, 2, 0, 2 << log_inv_rate, 1, 1),
                    vec![],
                    0,
                );
                let radius = report.source.analysis_radius.unwrap();

                assert!(
                    (libm::log2(1. - radius) - assumption.log_1_delta(log_inv_rate)).abs() < 1e-10
                );
            }
        }
    }

    #[test]
    fn interleaved_list_uses_eprint_2026_391_lemma_3_13_bound() {
        let assumption = SecurityAssumption::JohnsonBound;
        let log_degree = 3;
        let log_inv_rate = 2;
        let interleaving = 4;
        let one_minus_delta = libm::pow(2., assumption.log_1_delta(log_inv_rate));

        let base = assumption.list_size_bits(log_degree, log_inv_rate);
        let crude = interleaving as f64 * base;
        let actual = interleaved_list_bits(
            assumption,
            log_degree,
            log_inv_rate,
            interleaving,
            one_minus_delta,
        );

        assert!((actual - 13.228_818_690_495_881).abs() < 1e-12);
        assert!((crude - 21.287_712_379_549_45).abs() < 1e-12);
        assert!(actual < crude);
    }

    #[test]
    fn saturated_queries_and_rate_one_envelopes_are_reported() {
        let saturated = generic_report(
            SecurityAssumption::UniqueDecoding,
            HidingCodeGeometry::new(HidingCodeRole::Source, 2, 0, 4, 1, 10),
            vec![],
            0,
        );
        assert_eq!(saturated.source.requested_queries, 10);
        assert_eq!(saturated.source.effective_queries, 4);
        assert_eq!(
            saturated.source.query_miss.bits,
            Some(SecurityAssumption::UniqueDecoding.queries_error(1, 4))
        );

        let rate_one = generic_report(
            SecurityAssumption::UniqueDecoding,
            HidingCodeGeometry::new(HidingCodeRole::Source, 4, 4, 8, 1, 1),
            vec![],
            0,
        );
        assert_eq!(rate_one.source.randomized_rate, 1.);
        assert_eq!(rate_one.source.analysis_log_inv_rate, None);
        assert_eq!(rate_one.source.analysis_radius, None);
        assert_eq!(rate_one.source.query_miss.bits, None);
        assert_eq!(
            rate_one.source.query_miss.classification,
            HidingBoundClassification::Unavailable
        );
        assert_eq!(rate_one.source.mca.bits, None);
        assert_eq!(rate_one.source.paired_list.log2_size, None);
        assert_eq!(rate_one.round_by_round.bits, None);
    }

    #[test]
    #[should_panic(expected = "the codeword domain size must be a power of two")]
    fn geometry_rejects_non_dyadic_domains() {
        let _ = HidingCodeGeometry::new(HidingCodeRole::Source, 1, 0, 3, 1, 1);
    }

    #[test]
    #[should_panic(expected = "the randomized-code dimension must fit")]
    fn geometry_rejects_dimensions_larger_than_the_domain() {
        let _ = HidingCodeGeometry::new(HidingCodeRole::Source, 5, 4, 8, 1, 1);
    }

    #[test]
    #[should_panic(expected = "the interleaving width must be nonzero")]
    fn geometry_rejects_zero_width() {
        let _ = HidingCodeGeometry::new(HidingCodeRole::Source, 1, 0, 8, 0, 1);
    }

    #[test]
    fn probability_sum_is_stable_for_large_security_bits() {
        let actual = sum_probability_bits([2_000., 2_001.]);
        let expected = 2_000. - libm::log2(1.5);

        assert!((actual - expected).abs() < 1e-12);
    }
}
