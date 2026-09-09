//! Caller-facing parameters, and the round schedule derived from them.
//!
//! [`BinaryPcsParams`] is what a caller picks: a code rate, a grinding budget, and a target
//! security level. [`BinaryPcsConfig::try_new`] turns those, together with the committed
//! polynomial's arity, into the schedule the prover and the verifier both read — how many
//! folds run, how long the final codeword is, and how many queries the unique-decoding regime
//! demands. Every way that combination can fail to describe a usable protocol is a variant of
//! [`BinaryPcsConfigError`] and is returned rather than asserted, since the parameters come
//! from the caller.

use p3_security::binary::BinaryPcsRegime;
use thiserror::Error;

/// Largest grinding request `BinaryChallenger<BinaryField128, _>::grind` accepts.
///
/// The challenger asserts `bits + 8 <= min(F::bits(), 64)`, so the counter width, not the
/// field width, is what binds at this level.
const MAX_POW_BITS: usize = 56;

/// Parameters chosen by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryPcsParams {
    /// Log of the inverse code rate. The codeword is `2^log_inv_rate` times the message.
    pub log_inv_rate: usize,
    /// Grinding bits demanded once, before the query phase.
    pub pow_bits: usize,
    /// Target for the union of opening-claim, fold, sumcheck, and query errors, in bits.
    pub security_level: usize,
}

/// The round schedule derived from [`BinaryPcsParams`] and the polynomial's arity.
///
/// The schedule uses `p3-security` to budget all algebraic opening errors and queries.
/// Opening protocols are checked against the remaining claim-batching capacity.
/// In particular it does not price the commitment scheme, which is supplied separately.
/// A digest narrower than `2 * security_level` bits leaves the reported level undeliverable.
/// Supplying one wide enough is the caller's obligation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryPcsConfig {
    params: BinaryPcsParams,
    num_variables: usize,
    num_queries: usize,
    log_folding_factor: usize,
}

/// Why a [`BinaryPcsConfig`] could not be derived.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum BinaryPcsConfigError {
    /// A batch must fold between one and all remaining message variables.
    #[error("folding factor log {requested} must be in 1..={num_variables}")]
    InvalidFoldingFactor {
        requested: usize,
        num_variables: usize,
    },
    /// The codeword length does not fit in a `usize`.
    ///
    /// A codeword is indexed by `usize`, so `log_len` must stay below `max_bits`
    /// (`usize::BITS`): at or past that width, computing the codeword's length as
    /// `1usize << log_len` is already invalid, regardless of how large the additive domain
    /// itself is.
    #[error("codeword length 2^{log_len} does not fit in a {max_bits}-bit usize index")]
    CodewordLengthExceedsUsize { log_len: usize, max_bits: usize },

    /// The committed polynomial has no variables, so there is no round to fold.
    #[error("polynomial has no variables to fold")]
    NoVariablesToFold,

    /// The challenger cannot produce a witness this wide.
    #[error("grinding requests {requested} bits; the witness type admits at most {max}")]
    PowBitsExceedWitnessCapacity { requested: usize, max: usize },

    /// Grinding would consume the whole budget, leaving nothing for the queries to buy.
    #[error("security level {security_level} does not exceed the grinding budget {pow_bits}")]
    SecurityLevelBelowPowBits {
        security_level: usize,
        pow_bits: usize,
    },

    /// The target exceeds what the alphabet can deliver at this domain size.
    ///
    /// Queries buy back only the proximity term. The fold and sumcheck rounds each lose bits
    /// to the field's width, and no query count recovers them, so a target above `max` would
    /// be reported as met while being unreachable at any query count.
    #[error(
        "security level {security_level} exceeds the {max} bits the field leaves at this domain size"
    )]
    SecurityLevelExceedsFieldCapacity { security_level: usize, max: usize },

    /// The derived query count was zero, which would accept any codeword.
    #[error("derived a query count of zero")]
    ZeroQueries,
}

impl BinaryPcsConfig {
    /// Derive the round schedule for a polynomial in `num_variables` variables.
    ///
    /// Every variable is folded: the commit phase lays the whole polynomial out as a single
    /// codeword column. By default each variable fold commits its own word; use
    /// [`Self::try_with_folding`] to batch several folds between commitments.
    ///
    /// # Errors
    ///
    /// Returns a [`BinaryPcsConfigError`] if the codeword length does not fit in a `usize`,
    /// if the polynomial has no variables, if grinding exceeds what the challenger can witness
    /// or the security budget, if the target exceeds what the field can deliver, or if the
    /// derived query count is zero.
    pub fn try_new(
        num_variables: usize,
        params: BinaryPcsParams,
    ) -> Result<Self, BinaryPcsConfigError> {
        Self::try_new_with_folding(num_variables, params, 1)
    }

    /// Derive a schedule with batching selected before validating its security target.
    ///
    /// Unlike starting with `try_new` and then changing the folding factor, this admits
    /// targets that need exhaustive batched queries to release the query-error reserve.
    /// Returns the same configuration errors as [`Self::try_new`], or an invalid fold factor.
    pub fn try_new_with_folding(
        num_variables: usize,
        params: BinaryPcsParams,
        log_folding_factor: usize,
    ) -> Result<Self, BinaryPcsConfigError> {
        // The codeword length is computed downstream as `1usize << log_len`, so `log_len`
        // must stay below `usize::BITS` or that shift is already invalid. `checked_add`
        // guards the sum itself: an adversarial `num_variables` and `log_inv_rate` must not
        // silently wrap into an in-range `log_len`.
        let max_bits = usize::BITS as usize;
        let log_len = num_variables.saturating_add(params.log_inv_rate);
        let in_range = matches!(
            num_variables.checked_add(params.log_inv_rate),
            Some(len) if len < max_bits
        );
        if !in_range {
            return Err(BinaryPcsConfigError::CodewordLengthExceedsUsize { log_len, max_bits });
        }

        if num_variables == 0 {
            return Err(BinaryPcsConfigError::NoVariablesToFold);
        }

        if log_folding_factor == 0 || log_folding_factor > num_variables {
            return Err(BinaryPcsConfigError::InvalidFoldingFactor {
                requested: log_folding_factor,
                num_variables,
            });
        }

        if params.pow_bits > MAX_POW_BITS {
            return Err(BinaryPcsConfigError::PowBitsExceedWitnessCapacity {
                requested: params.pow_bits,
                max: MAX_POW_BITS,
            });
        }

        if params.security_level <= params.pow_bits {
            return Err(BinaryPcsConfigError::SecurityLevelBelowPowBits {
                security_level: params.security_level,
                pow_bits: params.pow_bits,
            });
        }

        let num_queries = BinaryPcsRegime::queries_for_target(
            params.security_level,
            params.log_inv_rate,
            params.pow_bits,
        );
        if num_queries == 0 {
            return Err(BinaryPcsConfigError::ZeroQueries);
        }
        let config = Self {
            params,
            num_variables,
            num_queries,
            log_folding_factor,
        };
        config.validate_security()?;
        Ok(config)
    }

    const fn validate_security(&self) -> Result<(), BinaryPcsConfigError> {
        let max = self.security_regime().max_security_bits();
        if self.params.security_level > max {
            return Err(BinaryPcsConfigError::SecurityLevelExceedsFieldCapacity {
                security_level: self.params.security_level,
                max,
            });
        }
        Ok(())
    }

    /// Batch up to `log_folding_factor` sequential variable folds between commitments.
    /// The last batch folds the remaining variables, even when shorter.
    ///
    /// For batching, charge every deterministic virtual fold, including those without roots.
    /// The UDR mutual-agreement bound lifts agreement through each omitted fold: the input
    /// word is fixed before its challenge, and the query authenticates the whole coset.
    /// See Appendix A, Theorem 8 of <https://eprint.iacr.org/2024/1553>.
    /// Sequential independent challenges are used, not powers of one sampled challenge.
    ///
    /// Sum `(N / 2^r + 1) / |F|` over every variable fold, plus `2 / |F|` per
    /// sumcheck round. Reserve one bit each for that sum and the query error so their sum
    /// meets the requested budget. Query PoW is credited only to the query error.
    /// Single and batched schedules use the same summed field budget. Exhaustive queries
    /// have zero error and need no query reserve.
    ///
    /// # Errors
    /// Rejects an empty/oversized batch or a target exceeding the batched field budget.
    pub fn try_with_folding(
        mut self,
        log_folding_factor: usize,
    ) -> Result<Self, BinaryPcsConfigError> {
        if log_folding_factor == 0 || log_folding_factor > self.num_variables {
            return Err(BinaryPcsConfigError::InvalidFoldingFactor {
                requested: log_folding_factor,
                num_variables: self.num_variables,
            });
        }
        self.log_folding_factor = log_folding_factor;
        self.validate_security()?;
        Ok(self)
    }

    /// Maximum number of sequential variable challenges per committed fold batch.
    #[must_use]
    pub const fn log_folding_factor(&self) -> usize {
        self.log_folding_factor
    }

    /// Number of fold batches, including the final batch sent in the clear.
    #[must_use]
    pub const fn num_fold_batches(&self) -> usize {
        self.num_variables.div_ceil(self.log_folding_factor)
    }

    /// Each batch's starting variable and number of challenges, in transcript order.
    pub(crate) fn fold_batches(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.num_variables)
            .step_by(self.log_folding_factor)
            .map(|start| {
                (
                    start,
                    self.log_folding_factor.min(self.num_variables - start),
                )
            })
    }

    /// Number of variables of the committed polynomial.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Number of 2-to-1 codeword folds: one per residual sumcheck round.
    #[must_use]
    pub const fn num_fold_rounds(&self) -> usize {
        self.num_variables
    }

    /// Log length of the final codeword, which is sent in full.
    #[must_use]
    pub const fn log_final_len(&self) -> usize {
        self.params.log_inv_rate
    }

    /// Log length of the base codeword: the polynomial's arity blown up by the inverse rate.
    #[must_use]
    pub(crate) const fn log_domain_size(&self) -> usize {
        self.num_variables + self.params.log_inv_rate
    }

    /// Length of the base codeword committed at commit time.
    #[must_use]
    pub(crate) const fn domain_size(&self) -> usize {
        1usize << self.log_domain_size()
    }

    /// Number of query indices sampled in the query phase.
    #[must_use]
    pub const fn num_queries(&self) -> usize {
        self.num_queries
    }

    /// Log of the inverse code rate.
    #[must_use]
    pub const fn log_inv_rate(&self) -> usize {
        self.params.log_inv_rate
    }

    /// Grinding bits demanded before the query phase.
    #[must_use]
    pub const fn pow_bits(&self) -> usize {
        self.params.pow_bits
    }

    /// Validated security model for this exact fold and query schedule.
    #[must_use]
    pub const fn security_regime(&self) -> BinaryPcsRegime {
        match BinaryPcsRegime::new(
            self.num_variables,
            self.params.log_inv_rate,
            self.log_folding_factor,
            self.num_queries,
            self.params.pow_bits,
        ) {
            Some(regime) => regime,
            None => panic!("configuration invariants must describe a binary PCS regime"),
        }
    }

    /// Configured total algebraic opening-security target, excluding hash collisions.
    #[must_use]
    pub const fn security_level(&self) -> usize {
        self.params.security_level
    }

    /// Conservative capacity for scalar evaluations, counting current and successor columns.
    ///
    /// The validated configuration always leaves room for at least one claim. Query grinding
    /// does not increase this alpha budget: it happens after the batching challenge.
    #[must_use]
    pub fn max_opening_claims(&self) -> usize {
        self.security_regime()
            .max_opening_claims(self.params.security_level)
    }

    /// Query security before grinding; infinity when every base coset is queried.
    #[must_use]
    pub fn query_security_bits(&self) -> f64 {
        self.security_regime().query_security_bits()
    }

    /// Whole-bit lower bound for all fold and sumcheck errors, before opening batching.
    #[must_use]
    pub const fn field_security_bits(&self) -> usize {
        self.security_regime().field_security_bits()
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField128;
    use p3_field::Field;
    use p3_security::InstanceShape;
    use p3_security::binary::BINARY_PCS_FIELD_BITS;
    use p3_security::fri::{FriRegime, commit_phase_error_udr};

    use super::{BinaryPcsConfig, BinaryPcsConfigError, BinaryPcsParams};

    const fn params() -> BinaryPcsParams {
        BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 16,
            security_level: 100,
        }
    }

    #[test]
    fn the_alphabet_width_matches_the_field() {
        assert_eq!(BINARY_PCS_FIELD_BITS, BinaryField128::bits());
    }

    #[test]
    fn derives_the_fold_schedule_and_query_count() {
        let config = BinaryPcsConfig::try_new(10, params()).unwrap();
        assert_eq!(config.num_variables(), 10);
        // Every variable folds the codeword.
        assert_eq!(config.num_fold_rounds(), 10);
        // Folding stops when the message is a constant, leaving the rate expansion.
        assert_eq!(config.log_final_len(), 2);
        assert!(config.num_queries() > 0);
    }

    #[test]
    fn batched_schedule_keeps_all_variables_and_a_short_final_batch() {
        let config = BinaryPcsConfig::try_new(7, params())
            .unwrap()
            .try_with_folding(3)
            .unwrap();
        assert_eq!(
            config.fold_batches().collect::<alloc::vec::Vec<_>>(),
            [(0, 3), (3, 3), (6, 1)]
        );
        assert_eq!(config.num_fold_rounds(), 7);
        assert_eq!(config.num_fold_batches(), 3);
        assert_eq!(config.log_final_len(), 2);
        assert!(config.query_security_bits() >= 85.0);
        assert!(
            BinaryPcsConfig::try_new(7, params())
                .unwrap()
                .try_with_folding(0)
                .is_err()
        );
        assert!(
            BinaryPcsConfig::try_new(7, params())
                .unwrap()
                .try_with_folding(8)
                .is_err()
        );
    }

    #[test]
    fn batching_prices_every_virtual_fold_and_reserves_error_budget() {
        let config = BinaryPcsConfig::try_new(19, params())
            .unwrap()
            .try_with_folding(3)
            .unwrap();
        // Sum (2^(21-r)+1), r=0..18, plus 2*19 = 4_194_353; ceil(log2)=23.
        assert_eq!(config.field_security_bits(), 105);
        let mut p = params();
        p.security_level = 105;
        assert!(BinaryPcsConfig::try_new(19, p).is_err());
        p.security_level = 104;
        assert!(
            BinaryPcsConfig::try_new(19, p)
                .unwrap()
                .try_with_folding(3)
                .is_ok()
        );
    }

    #[test]
    fn exhaustive_batching_can_reclaim_the_query_reserve_at_construction() {
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 117,
        };
        assert!(BinaryPcsConfig::try_new(7, params).is_err());
        let batched = BinaryPcsConfig::try_new_with_folding(7, params, 2).unwrap();
        assert_eq!(batched.max_opening_claims(), 1012);
        assert!(batched.try_with_folding(1).is_err());
    }

    /// `query_security_bits` reports the achieved security the sampled queries buy back after
    /// grinding, which must at least meet the protocol's target: `security_level - pow_bits`.
    #[test]
    fn query_security_bits_meets_the_protocol_target() {
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 8,
            security_level: 100,
        };
        let config = BinaryPcsConfig::try_new(16, params).unwrap();
        let protocol_target = (params.security_level - params.pow_bits) as f64;
        assert!(
            config.query_security_bits() >= protocol_target,
            "{} does not meet the protocol target {protocol_target}",
            config.query_security_bits()
        );
    }

    #[test]
    fn query_count_grows_as_the_rate_approaches_one() {
        let mut low = params();
        low.log_inv_rate = 1;
        let mut high = params();
        high.log_inv_rate = 4;
        let few = BinaryPcsConfig::try_new(10, high).unwrap().num_queries();
        let many = BinaryPcsConfig::try_new(10, low).unwrap().num_queries();
        assert!(
            many > few,
            "a worse rate must demand more queries: {many} vs {few}"
        );
    }

    /// Independent comparison with the sum of the FRI per-round UDR bounds and quadratic
    /// sumcheck errors. Comparing only the largest FRI round would miss omitted folds.
    #[test]
    fn the_field_security_agrees_with_p3_security() {
        for num_variables in [10, 16, 20, 24] {
            let config = BinaryPcsConfig::try_new(
                num_variables,
                BinaryPcsParams {
                    security_level: 90,
                    ..params()
                },
            )
            .unwrap();
            let regime = FriRegime {
                log_blowup: config.log_inv_rate(),
                num_queries: config.num_queries(),
                log_final_poly_len: config.log_final_len(),
                max_log_arity: 1,
                commit_pow_bits: 0,
                query_pow_bits: config.pow_bits(),
            };
            let mut errors: alloc::vec::Vec<_> = (1..=num_variables)
                .map(|remaining| {
                    commit_phase_error_udr(
                        &regime,
                        &InstanceShape {
                            log_trace_length: remaining,
                            modulus_bits: BINARY_PCS_FIELD_BITS,
                            collision_resistance: BINARY_PCS_FIELD_BITS,
                            num_batched_functions: 1,
                        },
                    )
                    .unwrap()
                })
                .collect();
            errors.push(p3_security::ErrorBits::from_log2(
                128.0 - (2.0 * num_variables as f64).log2(),
            ));
            let reference = p3_security::ErrorBits::sum(&errors).bits();
            let ours = config.field_security_bits() as f64;
            assert!(ours <= reference && reference - ours < 1.0);
        }
    }

    #[test]
    fn rejects_a_polynomial_with_no_variables() {
        assert_eq!(
            BinaryPcsConfig::try_new(0, params()),
            Err(BinaryPcsConfigError::NoVariablesToFold)
        );
    }

    #[test]
    fn rejects_grinding_beyond_the_witness_capacity() {
        let mut p = params();
        p.pow_bits = 57;
        assert_eq!(
            BinaryPcsConfig::try_new(10, p),
            Err(BinaryPcsConfigError::PowBitsExceedWitnessCapacity {
                requested: 57,
                max: 56,
            })
        );
    }

    #[test]
    fn rejects_a_codeword_at_least_as_wide_as_the_index_type() {
        let max_bits = usize::BITS as usize;
        // `num_variables + log_inv_rate` lands exactly on `max_bits`, the smallest length
        // for which `1usize << log_len` is already invalid.
        let num_variables = max_bits - params().log_inv_rate;
        assert_eq!(
            BinaryPcsConfig::try_new(num_variables, params()),
            Err(BinaryPcsConfigError::CodewordLengthExceedsUsize {
                log_len: max_bits,
                max_bits,
            })
        );
    }

    #[test]
    fn rejects_a_rate_that_derives_zero_queries() {
        // At log_inv_rate = 0, log_1_delta(0) = log2(1 + 2^0) - 1 = 0.0 exactly, so the
        // derived query count saturates to zero regardless of the security level.
        let mut p = params();
        p.log_inv_rate = 0;
        assert_eq!(
            BinaryPcsConfig::try_new(10, p),
            Err(BinaryPcsConfigError::ZeroQueries)
        );
    }

    #[test]
    fn rejects_a_security_level_at_or_below_the_grinding_budget() {
        let mut p = params();
        p.security_level = 16;
        assert_eq!(
            BinaryPcsConfig::try_new(10, p),
            Err(BinaryPcsConfigError::SecurityLevelBelowPowBits {
                security_level: 16,
                pow_bits: 16,
            })
        );
    }

    /// A target the alphabet cannot deliver is rejected rather than met on paper by piling on
    /// queries: at 20 variables and rate `2^-2` the base codeword holds `2^22` symbols, so the
    /// summed field error plus the query reserve caps the configured target at 103 bits.
    #[test]
    fn rejects_a_security_level_the_field_cannot_deliver() {
        let mut p = params();
        p.security_level = 120;
        assert_eq!(
            BinaryPcsConfig::try_new(20, p),
            Err(BinaryPcsConfigError::SecurityLevelExceedsFieldCapacity {
                security_level: 120,
                max: 103,
            })
        );

        // The stated cap is accepted.
        p.security_level = 103;
        assert!(BinaryPcsConfig::try_new(20, p).is_ok());
    }

    /// The cap tightens as the committed polynomial grows: a longer codeword spends more of
    /// the field's width on the fold's proximity term.
    #[test]
    fn the_field_capacity_shrinks_as_the_domain_grows() {
        let small = BinaryPcsConfig::try_new(10, params())
            .unwrap()
            .field_security_bits();
        let large = BinaryPcsConfig::try_new(20, params())
            .unwrap()
            .field_security_bits();
        assert!(
            large < small,
            "a larger domain must leave fewer bits: {large} vs {small}"
        );
    }
}
