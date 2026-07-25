//! Diagnostic ideal-IOP soundness accounting for the hiding base case.
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

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use serde::Serialize;

use crate::parameters::SecurityAssumption;

use super::config::ZkWhirConfig;

/// How much theorem support a reported bound has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum HidingBoundClassification {
    /// The displayed ideal-IOP term follows from the selected coding theorem.
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
/// `bits` is `None` exactly when the term is unavailable.  Available values
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

    /// Returns bits only for an unconditional, proven ideal-IOP bound.
    #[must_use]
    pub const fn proven_bits(&self) -> Option<f64> {
        match (self.classification, self.bits) {
            (HidingBoundClassification::Proven, bits) => bits,
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
    /// exact zero miss probability.
    pub query_miss: HidingErrorBound,
    /// MCA error for this component.  Interleaved groups include the
    /// conservative factor equal to `width`.
    pub mca: HidingErrorBound,
    /// Candidate-list bound for the old/fresh pair.  Its interleaving is two
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
    /// User-requested security level.  The report does not assert it is met.
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
    /// notation. It does not include the compiler-conditional PoW estimate,
    /// sum the coordinates into a single protocol failure probability, or
    /// compose them with other WHIR or compiler errors.
    pub round_by_round: HidingErrorBound,
}

impl<EF, F, Challenger> ZkWhirConfig<EF, F, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Reports the heterogeneous hiding base case's ideal-IOP RBR terms.
    ///
    /// This is diagnostic only: it neither rejects the configuration nor
    /// certifies the compiled PCS or the preceding hiding reductions.
    #[must_use]
    pub fn hiding_base_case_security_report(&self) -> HidingBaseCaseSecurityReport {
        let field_order_log2_estimate = log2_order_estimate::<EF>();
        let field_order_log2_floor = EF::order().bits() as usize - 1;
        let final_config = self.final_round_config();
        let source = code_report(
            HidingCodeRole::Source,
            1 << final_config.num_variables,
            self.oracle_randomness[self.n_rounds()],
            final_config.domain_size >> final_config.folding_factor,
            1,
            self.final_queries,
            self.soundness_type,
            field_order_log2_floor as f64,
            EF::bits(),
        );

        let mut mask_groups = Vec::with_capacity(2 * self.n_rounds() + 1);
        let groups = self.mask_groups();
        for (index, group) in groups.into_iter().enumerate() {
            let role = if index % 2 == 0 {
                HidingCodeRole::SumcheckMask { round: index / 2 }
            } else {
                HidingCodeRole::CodeSwitchMask { round: index / 2 }
            };
            mask_groups.push(code_report(
                role,
                group.shape.message_len,
                group.shape.randomness_len,
                group.shape.domain_size,
                group.width,
                self.mask_queries,
                self.soundness_type,
                field_order_log2_floor as f64,
                EF::bits(),
            ));
        }

        let source_before_pow_bits = source.query_miss.bits;
        let masks_before_pow_bits: Vec<Option<f64>> = mask_groups
            .iter()
            .map(|group| group.query_miss.bits)
            .collect();
        let weakest_before_pow_bits = minimum_available(
            core::iter::once(source_before_pow_bits).chain(masks_before_pow_bits.iter().copied()),
        );
        let before_pow = weakest_before_pow_bits
            .map_or_else(HidingErrorBound::unavailable, |bits| {
                HidingErrorBound::available(bits, HidingBoundClassification::Proven)
            });
        let shared_pow_bits = self.final_pow_bits;
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
        let mca_classification = mca_classification(self.soundness_type);
        let mca_sum = sum_available_probability_bits(mca_bounds)
            .map_or_else(HidingErrorBound::unavailable, |bits| {
                HidingErrorBound::available(bits, mca_classification)
            });

        let list_classification = list_classification(self.soundness_type);
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
            target_bits: self.security_level,
            assumption: self.soundness_type,
            field_order_log2_estimate,
            field_order_log2_floor,
            source,
            mask_groups,
            query_round,
            gamma_round,
            round_by_round,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn code_report(
    role: HidingCodeRole,
    message_len: usize,
    randomness_len: usize,
    domain_size: usize,
    width: usize,
    num_queries: usize,
    assumption: SecurityAssumption,
    field_bits: f64,
    rounded_field_bits: usize,
) -> HidingCodeSecurityReport {
    let randomized_dimension = message_len + randomness_len;
    let dyadic_dimension = randomized_dimension.next_power_of_two();
    debug_assert!(domain_size.is_power_of_two());
    debug_assert!(dyadic_dimension <= domain_size);
    let log_degree = dyadic_dimension.ilog2() as usize;
    let raw_log_inv_rate = domain_size.ilog2() as usize - log_degree;
    let analysis_log_inv_rate = (raw_log_inv_rate > 0).then_some(raw_log_inv_rate);
    let analysis_radius = analysis_log_inv_rate.map(|rate| proximity_radius(assumption, rate));
    let effective_queries = num_queries.min(domain_size);

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
            let query_bits = assumption.queries_error(log_inv_rate, effective_queries);
            let mca_bits = scalar_mca_bits(
                assumption,
                log_degree,
                log_inv_rate,
                field_bits,
                rounded_field_bits,
            ) - libm::log2(width as f64);
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
        requested_queries: num_queries,
        effective_queries,
        query_miss,
        mca,
        paired_list,
    }
}

fn proximity_radius(assumption: SecurityAssumption, log_inv_rate: usize) -> f64 {
    let rho = libm::pow(2., -(log_inv_rate as f64));
    match assumption {
        SecurityAssumption::UniqueDecoding => 0.5 * (1. - rho),
        SecurityAssumption::JohnsonBound => {
            1. - libm::sqrt(rho) - libm::pow(2., assumption.log_eta(log_inv_rate))
        }
        SecurityAssumption::CapacityBound => {
            1. - rho - libm::pow(2., assumption.log_eta(log_inv_rate))
        }
    }
}

fn scalar_mca_bits(
    assumption: SecurityAssumption,
    log_degree: usize,
    log_inv_rate: usize,
    field_bits: f64,
    rounded_field_bits: usize,
) -> f64 {
    // Reuse the security crate's scalar formula, then remove the optimistic
    // upward rounding from `Field::bits()`.  The Johnson helper deliberately
    // keeps only the dominant BCSS25 term, hence its Approximation label.
    let rounded = assumption.prox_gaps_error(log_degree, log_inv_rate, rounded_field_bits, 2);
    rounded - (rounded_field_bits as f64 - field_bits)
}

fn interleaved_list_bits(
    assumption: SecurityAssumption,
    log_degree: usize,
    log_inv_rate: usize,
    interleaving: usize,
) -> f64 {
    let base = assumption.list_size_bits(log_degree, log_inv_rate);
    let crude = interleaving as f64 * base;

    // Lemma 3.13's interleaving-stable alternative:
    // binomial(b + r, r) * L^r.  Taking the better of this and L^K is
    // conservative because both are upper bounds.
    let rho = libm::pow(2., -(log_inv_rate as f64));
    let distance = 1. - rho;
    let delta = proximity_radius(assumption, log_inv_rate);
    let gap = distance - delta;
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
    use HidingBoundClassification::{Approximation, Conditional, Proven, Unavailable};

    match (left, right) {
        (Unavailable, _) | (_, Unavailable) => Unavailable,
        (Conditional, _) | (_, Conditional) => Conditional,
        (Approximation, _) | (_, Approximation) => Approximation,
        (Proven, Proven) => Proven,
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

    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeField32;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};

    use super::*;
    use crate::parameters::{FoldingFactor, ProtocolParameters};
    use crate::pcs::zk::config::ZkParameters;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2KoalaBear<16>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    fn stock_config(assumption: SecurityAssumption) -> ZkWhirConfig<EF, F, Challenger> {
        ZkWhirConfig::new(
            16,
            ProtocolParameters {
                security_level: 100,
                pow_bits: 16,
                round_log_inv_rates: vec![],
                folding_factor: FoldingFactor::Constant(4),
                soundness_type: assumption,
                starting_log_inv_rate: 2,
            },
            ZkParameters {
                ell_zk: 16,
                mask_log_inv_rate: 5,
            },
        )
        .unwrap()
    }

    fn classification_config(assumption: SecurityAssumption) -> ZkWhirConfig<EF, F, Challenger> {
        ZkWhirConfig::new(
            16,
            ProtocolParameters {
                security_level: 32,
                pow_bits: 0,
                round_log_inv_rates: vec![],
                folding_factor: FoldingFactor::Constant(4),
                soundness_type: assumption,
                starting_log_inv_rate: 2,
            },
            ZkParameters {
                ell_zk: 4,
                mask_log_inv_rate: 1,
            },
        )
        .unwrap()
    }

    #[test]
    fn field_report_separates_estimate_from_rigorous_floor() {
        let estimate = log2_order_estimate::<EF>();
        let exact = 4. * libm::log2(F::ORDER_U32 as f64);

        assert!((estimate - exact).abs() < 1e-12);
        assert_eq!(EF::order().bits() as usize - 1, 123);
        assert!(estimate < EF::bits() as f64);
    }

    #[test]
    fn stock_capacity_report_exposes_randomized_source_rate_loss() {
        let report =
            stock_config(SecurityAssumption::CapacityBound).hiding_base_case_security_report();

        assert_eq!(report.assumption, SecurityAssumption::CapacityBound);
        assert_eq!(report.field_order_log2_floor, 123);
        assert!((report.field_order_log2_estimate - 123.954_738_749_797).abs() < 1e-12);
        assert_eq!(report.source.message_len, 16);
        assert_eq!(report.source.randomness_len, 11);
        assert_eq!(report.source.randomized_dimension, 27);
        assert_eq!(report.source.dyadic_dimension, 32);
        assert_eq!(report.source.domain_size, 4096);
        assert_eq!(report.source.randomized_rate, 27. / 4096.);
        assert_eq!(report.source.analysis_log_inv_rate, Some(7));
        assert!(report.source.analysis_radius.is_some());
        assert_eq!(report.source.requested_queries, 11);
        assert_eq!(report.source.effective_queries, 11);
        assert_eq!(report.query_round.shared_pow_bits, 13);
        assert!(
            (report.query_round.source_with_pow.bits.unwrap() - 89.225_717_393_194_63).abs()
                < 1e-12
        );
        assert_eq!(
            report.query_round.with_pow.bits,
            report.query_round.source_with_pow.bits
        );
        assert_eq!(
            report.query_round.with_pow.classification,
            HidingBoundClassification::Conditional
        );
        assert!(report.query_round.before_pow.bits.unwrap() < report.target_bits as f64);
        assert!(report.query_round.with_pow.bits.unwrap() < report.target_bits as f64);
        assert_eq!(
            report.gamma_round.combined.classification,
            HidingBoundClassification::Conditional
        );
        assert!(report.gamma_round.product_list_over_field.vacuous);
        assert!(report.round_by_round.vacuous);
    }

    #[test]
    fn mask_groups_keep_chronological_roles() {
        let report =
            stock_config(SecurityAssumption::CapacityBound).hiding_base_case_security_report();
        let roles: Vec<_> = report.mask_groups.iter().map(|group| group.role).collect();

        assert_eq!(
            roles,
            vec![
                HidingCodeRole::SumcheckMask { round: 0 },
                HidingCodeRole::CodeSwitchMask { round: 0 },
                HidingCodeRole::SumcheckMask { round: 1 },
                HidingCodeRole::CodeSwitchMask { round: 1 },
                HidingCodeRole::SumcheckMask { round: 2 },
            ]
        );
        assert!(
            report
                .mask_groups
                .iter()
                .all(|group| group.analysis_log_inv_rate == Some(5))
        );
        assert!(
            report
                .mask_groups
                .iter()
                .all(|group| group.randomness_len == 21)
        );
    }

    #[test]
    fn regime_classifications_propagate_without_overstatement() {
        let unique = classification_config(SecurityAssumption::UniqueDecoding)
            .hiding_base_case_security_report();
        assert_eq!(
            unique.gamma_round.combined.classification,
            HidingBoundClassification::Proven
        );
        assert_eq!(
            unique.round_by_round.classification,
            HidingBoundClassification::Proven
        );
        assert!(!unique.round_by_round.vacuous);

        let johnson = classification_config(SecurityAssumption::JohnsonBound)
            .hiding_base_case_security_report();
        assert_eq!(
            johnson.source.mca.classification,
            HidingBoundClassification::Approximation
        );
        assert_eq!(
            johnson.source.paired_list.classification,
            HidingBoundClassification::Proven
        );
        assert_eq!(
            johnson.gamma_round.combined.classification,
            HidingBoundClassification::Approximation
        );

        let capacity =
            stock_config(SecurityAssumption::CapacityBound).hiding_base_case_security_report();
        assert_eq!(
            capacity.gamma_round.combined.classification,
            HidingBoundClassification::Conditional
        );
    }

    #[test]
    fn query_pow_is_added_once_after_the_max_error_composition() {
        let report =
            stock_config(SecurityAssumption::CapacityBound).hiding_base_case_security_report();
        let expected_before = report
            .query_round
            .masks_before_pow_bits
            .iter()
            .copied()
            .chain(core::iter::once(report.query_round.source_before_pow_bits));
        let expected_before = minimum_available(expected_before).unwrap();

        assert_eq!(
            report.query_round.weakest_before_pow_bits,
            Some(expected_before)
        );
        assert_eq!(report.query_round.before_pow.bits, Some(expected_before));
        assert_eq!(
            report.query_round.with_pow.bits,
            Some(expected_before + report.query_round.shared_pow_bits as f64)
        );
        assert_eq!(
            report.query_round.with_pow.classification,
            HidingBoundClassification::Conditional
        );
    }

    #[test]
    fn heterogeneous_mca_errors_add_as_probabilities() {
        let report =
            stock_config(SecurityAssumption::CapacityBound).hiding_base_case_security_report();
        let component_bits = core::iter::once(&report.source)
            .chain(&report.mask_groups)
            .map(|component| component.mca.bits.unwrap());

        assert_eq!(
            report.gamma_round.mca_sum.bits,
            Some(sum_probability_bits(component_bits))
        );
    }

    #[test]
    fn saturated_query_schedule_reports_effective_rows() {
        let report = code_report(
            HidingCodeRole::Source,
            2,
            0,
            4,
            1,
            10,
            SecurityAssumption::UniqueDecoding,
            100.,
            100,
        );

        assert_eq!(report.requested_queries, 10);
        assert_eq!(report.effective_queries, 4);
        assert_eq!(
            report.query_miss.bits,
            Some(SecurityAssumption::UniqueDecoding.queries_error(1, 4))
        );
    }

    #[test]
    fn rate_one_dyadic_envelope_is_unavailable() {
        let report = code_report(
            HidingCodeRole::Source,
            4,
            4,
            8,
            1,
            1,
            SecurityAssumption::UniqueDecoding,
            100.,
            100,
        );

        assert_eq!(report.randomized_rate, 1.);
        assert_eq!(report.analysis_log_inv_rate, None);
        assert_eq!(report.analysis_radius, None);
        assert_eq!(report.requested_queries, 1);
        assert_eq!(report.effective_queries, 1);
        assert_eq!(report.query_miss.bits, None);
        assert_eq!(
            report.query_miss.classification,
            HidingBoundClassification::Unavailable
        );
        assert_eq!(report.mca.bits, None);
        assert_eq!(report.paired_list.log2_size, None);
    }
}
