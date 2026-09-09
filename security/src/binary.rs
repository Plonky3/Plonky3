//! Additive-domain binary PCS soundness over GF(2^128).
//!
//! Only unique decoding is supported. For each independent binary fold at word length N,
//! charge (N + 1)/|F| for proximity and 2/|F| for its quadratic sumcheck. This includes
//! virtual folds between Merkle commitments: full-coset authentication lifts agreement
//! through each fold (Appendix A, Theorem 8 of <https://eprint.iacr.org/2024/1553>).
//! Opening k evaluations with powers of one alpha adds (k - 1)/|F|. All errors are summed.
//!
//! The sole grind precedes query sampling, after alpha and every fold challenge. It buys
//! back only query error. An alpha grind would require a different transcript and model.
//! Hash/transcript collision security is supplied separately by the caller.

use crate::{ErrorBits, SecurityAssumption};

/// Exact bit width of the binary PCS's codeword alphabet.
pub const BINARY_PCS_FIELD_BITS: usize = 128;

/// Validated mirror of the binary PCS schedule. Constructed by the protocol crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryPcsRegime {
    num_variables: usize,
    log_inv_rate: usize,
    log_folding_factor: usize,
    num_queries: usize,
    query_pow_bits: usize,
}

impl BinaryPcsRegime {
    /// Reject empty, overflowing, rate-one, or query-free schedules.
    pub const fn new(
        num_variables: usize,
        log_inv_rate: usize,
        log_folding_factor: usize,
        num_queries: usize,
        query_pow_bits: usize,
    ) -> Option<Self> {
        let log_domain = match num_variables.checked_add(log_inv_rate) {
            Some(value) => value,
            None => return None,
        };
        if num_variables == 0
            || log_inv_rate == 0
            || log_domain >= usize::BITS as usize
            || log_folding_factor == 0
            || log_folding_factor > num_variables
            || num_queries == 0
        {
            return None;
        }
        Some(Self {
            num_variables,
            log_inv_rate,
            log_folding_factor,
            num_queries,
            query_pow_bits,
        })
    }

    /// Query count reserving half the requested error for queries. Full enumeration has
    /// zero query error and releases that reserve in [`Self::max_opening_claims`].
    pub fn queries_for_target(
        security_level: usize,
        log_inv_rate: usize,
        query_pow_bits: usize,
    ) -> usize {
        SecurityAssumption::UniqueDecoding.queries(
            security_level
                .saturating_add(1)
                .saturating_sub(query_pow_bits)
                .max(1),
            log_inv_rate,
        )
    }

    /// The only credited grinding site: immediately before query sampling.
    pub const fn query_pow_bits(&self) -> usize {
        self.query_pow_bits
    }

    /// Whether every first-batch coset is authenticated, eliminating query error.
    pub const fn queries_are_exhaustive(&self) -> bool {
        self.num_queries
            >= 1usize << (self.num_variables + self.log_inv_rate - self.log_folding_factor)
    }

    /// Sum of every fold and sumcheck error, in units of 1/2^128.
    pub const fn field_error_numerator(&self) -> u128 {
        (2u128 << (self.num_variables + self.log_inv_rate)) - (2u128 << self.log_inv_rate)
            + 3 * self.num_variables as u128
    }

    /// Whole-bit lower bound before adding opening claims or query error.
    pub const fn field_security_bits(&self) -> usize {
        let numerator = self.field_error_numerator();
        BINARY_PCS_FIELD_BITS - (u128::BITS - (numerator - 1).leading_zeros()) as usize
    }

    /// Field ceiling after reserving query error, before opening-claim batching.
    pub const fn max_security_bits(&self) -> usize {
        self.field_security_bits()
            .saturating_sub(if self.queries_are_exhaustive() { 0 } else { 1 })
    }

    /// Query security before grinding; infinity when all cosets are checked.
    pub fn query_security_bits(&self) -> f64 {
        if self.queries_are_exhaustive() {
            f64::INFINITY
        } else {
            SecurityAssumption::UniqueDecoding.queries_error(self.log_inv_rate, self.num_queries)
        }
    }

    /// Conservative claim capacity for a positive total-error target.
    ///
    /// Returns zero if the base error already exceeds its budget, queries are too weak,
    /// or the target is unsupported. In particular, subtraction underflow does not grant
    /// a spurious one-claim allowance. Configurations must reject zero capacity themselves.
    pub fn max_opening_claims(&self, security_level: usize) -> usize {
        let reserve = usize::from(!self.queries_are_exhaustive());
        if security_level == 0
            || security_level > BINARY_PCS_FIELD_BITS - reserve
            || self.query_security_bits() + (self.query_pow_bits as f64)
                < (security_level + reserve) as f64
        {
            return 0;
        }
        let budget = 1u128 << (BINARY_PCS_FIELD_BITS - security_level - reserve);
        let Some(remaining) = budget.checked_sub(self.field_error_numerator()) else {
            return 0;
        };
        (remaining + 1).min(usize::MAX as u128) as usize
    }

    /// Union of the opening-claim, fold, sumcheck, and query errors.
    ///
    /// Under unique decoding the candidate polynomial is unique at commitment time.
    /// This error therefore needs no candidate-list multiplier when composed with an AIR.
    pub fn opening_error(&self, num_claims: usize) -> ErrorBits {
        let numerator = self.field_error_numerator() + num_claims.saturating_sub(1) as u128;
        let mut rounded = numerator as f64;
        if (rounded as u128) < numerator {
            rounded = rounded.next_up();
        }
        let field = ErrorBits::from_log2(BINARY_PCS_FIELD_BITS as f64 - libm::log2(rounded));
        let queries = ErrorBits::from_log2(self.query_security_bits() + self.query_pow_bits as f64);
        ErrorBits::sum(&[field, queries])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_claim_boundaries_and_exhaustive_queries() {
        let tiny = BinaryPcsRegime::new(1, 2, 1, 200, 0).unwrap();
        assert_eq!(tiny.field_error_numerator(), 11);
        assert_eq!(tiny.max_opening_claims(124), 6);
        assert_eq!(tiny.opening_error(6).bits(), 124.0);
        assert!(tiny.opening_error(7).bits() < 124.0);
        let batched = BinaryPcsRegime::new(4, 2, 2, 200, 0).unwrap();
        assert_eq!(batched.field_error_numerator(), 132);
        // All 16 cosets are checked: the unused query reserve admits 381 claims.
        assert_eq!(batched.max_opening_claims(119), 381);
        assert_eq!(batched.opening_error(381).bits(), 119.0);
        assert!(batched.opening_error(382).bits() < 119.0);
    }

    #[test]
    fn an_exhausted_or_invalid_budget_has_zero_capacity() {
        let tiny = BinaryPcsRegime::new(1, 2, 1, 200, 0).unwrap();
        for target in [0, 125, 128, 129, usize::MAX] {
            assert_eq!(tiny.max_opening_claims(target), 0);
        }
        let too_few_queries = BinaryPcsRegime::new(10, 2, 1, 1, 0).unwrap();
        assert_eq!(too_few_queries.max_opening_claims(100), 0);
        assert!(BinaryPcsRegime::new(0, 2, 1, 1, 0).is_none());
        assert!(BinaryPcsRegime::new(1, 0, 1, 1, 0).is_none());
        assert!(BinaryPcsRegime::new(1, 2, 2, 1, 0).is_none());
        assert!(BinaryPcsRegime::new(1, 2, 1, 0, 0).is_none());
        assert!(BinaryPcsRegime::new(usize::MAX, 2, 1, 1, 0).is_none());
    }

    #[test]
    fn query_grinding_cannot_subsidize_alpha_or_folds() {
        let plain = BinaryPcsRegime::new(10, 2, 1, 150, 0).unwrap();
        let ground = BinaryPcsRegime::new(10, 2, 1, 150, 16).unwrap();
        assert_eq!(plain.field_error_numerator(), 8214);
        assert_eq!(
            plain.max_opening_claims(100),
            ground.max_opening_claims(100)
        );
        assert_eq!(plain.max_opening_claims(100), 134_209_515);
        assert!(plain.opening_error(2).bits() >= 100.0);
        assert!(ground.opening_error(2).bits() > plain.opening_error(2).bits());
    }

    #[test]
    fn both_folding_schedules_charge_every_round() {
        let single = BinaryPcsRegime::new(20, 2, 1, 150, 0).unwrap();
        let batched = BinaryPcsRegime::new(20, 2, 3, 150, 0).unwrap();
        assert_eq!(single.field_error_numerator(), 8_388_660);
        assert_eq!(
            single.field_error_numerator(),
            batched.field_error_numerator()
        );
        assert_eq!(single.max_security_bits(), 103);
    }
}
