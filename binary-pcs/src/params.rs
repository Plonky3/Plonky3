//! Caller-facing parameters, and the round schedule derived from them.
//!
//! [`BinaryPcsParams`] is what a caller picks: a code rate, a grinding budget, and a target
//! security level. [`BinaryPcsConfig::try_new`] turns those, together with the committed
//! polynomial's arity, into the schedule the prover and the verifier both read — how many
//! folds run, how long the final codeword is, and how many queries the unique-decoding regime
//! demands. Every way that combination can fail to describe a usable protocol is a variant of
//! [`BinaryPcsConfigError`] and is returned rather than asserted, since the parameters come
//! from the caller.

use p3_security::SecurityAssumption;
use thiserror::Error;

/// Largest grinding request `BinaryChallenger<BinaryField128, _>::grind` accepts.
///
/// The challenger asserts `bits + 8 <= min(F::bits(), 64)`, so the counter width, not the
/// field width, is what binds at this level.
const MAX_POW_BITS: usize = 56;

/// Bit width of the codeword alphabet, `BinaryField128`.
///
/// Checked against the field itself by `the_alphabet_width_matches_the_field` below, so this
/// cannot drift from the type the crate actually commits over.
const ALPHABET_BITS: usize = 128;

/// The regime this scheme is analysed in.
///
/// Capacity-rate list decodability is refuted over characteristic 2 with `F_2`-subspace
/// domains, and the Cantor domain is one. The Johnson bound is not refuted by this: it is an
/// unconditional theorem whose radius those same counterexamples show to be tight, but
/// `p3-security` documents it as resting on a correlated-agreement conjecture, and it is
/// excluded here by choice, not by mathematics. It is fixed here rather than taken from the
/// caller, which is why this crate does not re-export `SecurityAssumption`.
const REGIME: SecurityAssumption = SecurityAssumption::UniqueDecoding;

// A change of regime must fail to compile, not fail at run time.
// The other regimes are refuted over `F_2`-subspace domains.
// No configuration should accept them, so there is nothing to report as an error.
const _: () = assert!(matches!(REGIME, SecurityAssumption::UniqueDecoding));

/// `ceil(log2(value))`.
///
/// For `value >= 2` the highest set bit of `value - 1` sits one below the ceiling at a power
/// of two and exactly at it otherwise, so the leading-zero count gives the ceiling in both
/// cases. `value <= 1` needs no bits.
const fn log2_ceil_u128(value: u128) -> usize {
    if value <= 1 {
        return 0;
    }
    (u128::BITS - (value - 1).leading_zeros()) as usize
}

/// Bits of security the rounds leave once the field's width is paid for, rounded down.
///
/// The queries are not the only place soundness is spent, and the two rounds below lose bits
/// that no query count buys back. Both are bounded by a multiple of `1/|F|`:
///
/// - **The commit phase.** At fold arity 2 in the unique-decoding regime the per-round
///   proximity error is `(arity - 1) * (n + 1) / |F|`, i.e. `(n + 1) / |F|` for a base codeword
///   of `n` symbols. That is the bound `p3_security::fri::commit_phase_error_udr` states, which
///   `the_field_security_agrees_with_p3_security` below checks this function against.
/// - **The sumcheck.** Each of the `num_fold_rounds` rounds sends a degree-2 polynomial, so a
///   forged round survives a uniform challenge with probability at most `2 / |F|`.
///
/// Their sum is `n + 1 + 2 * num_fold_rounds` over `|F|`. The sum cannot overflow: `try_new`
/// admits `log_domain_size` only below `usize::BITS`, so `n` stays below `2^64`.
///
/// The two terms compose under different conventions:
///
/// - The commit term is a per-round bound, charged once.
/// - The sumcheck term is union-bounded over every round.
///
/// Charging a per-round bound once is the round-by-round convention.
/// The security crate composes the same way, taking the worst term rather than summing errors.
/// The mixture here is conservative, never optimistic.
///
/// At arity 2 the choice is numerically inert:
///
/// ```text
///     commit    n + 1      = 2^22   at the largest shape benched here
///     sumcheck  2 * rounds = 40
/// ```
///
/// Raising the arity shrinks the round count without shrinking `n`, so the two stay far apart.
///
/// Taking the ceiling of the logarithm rounds the result **down**, so this understates the
/// achieved security by up to one bit rather than overstating it.
const fn field_security_bits_at(log_domain_size: usize, num_fold_rounds: usize) -> usize {
    let numerator = (1u128 << log_domain_size) + 1 + 2 * num_fold_rounds as u128;
    ALPHABET_BITS.saturating_sub(log2_ceil_u128(numerator))
}

/// Parameters chosen by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryPcsParams {
    /// Log of the inverse code rate. The codeword is `2^log_inv_rate` times the message.
    pub log_inv_rate: usize,
    /// Grinding bits demanded once, before the query phase.
    pub pow_bits: usize,
    /// Target security, in bits.
    pub security_level: usize,
}

/// The round schedule derived from [`BinaryPcsParams`] and the polynomial's arity.
///
/// The schedule prices the field's width and the query count, and nothing else.
/// In particular it does not price the commitment scheme, which is supplied separately.
/// A digest narrower than `2 * security_level` bits leaves the reported level undeliverable.
/// Supplying one wide enough is the caller's obligation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryPcsConfig {
    params: BinaryPcsParams,
    num_variables: usize,
    num_queries: usize,
}

/// Why a [`BinaryPcsConfig`] could not be derived.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum BinaryPcsConfigError {
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
    /// codeword column, so there is no head of preprocessing rounds to configure. Binding a
    /// prefix of the variables inside a committed row instead is the deferred head collapse,
    /// which would reintroduce a folding parameter alongside the machinery to honour it.
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

        // Every variable folds, so the fold-round count is the arity itself.
        let field_security_bits = field_security_bits_at(log_len, num_variables);
        if params.security_level > field_security_bits {
            return Err(BinaryPcsConfigError::SecurityLevelExceedsFieldCapacity {
                security_level: params.security_level,
                max: field_security_bits,
            });
        }

        // Grinding buys back `pow_bits`, so the queries only cover the remainder.
        let protocol_security_level = params.security_level - params.pow_bits;
        let num_queries = REGIME.queries(protocol_security_level, params.log_inv_rate);
        if num_queries == 0 {
            return Err(BinaryPcsConfigError::ZeroQueries);
        }

        Ok(Self {
            params,
            num_variables,
            num_queries,
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

    /// Bits of security the sampled queries deliver.
    #[must_use]
    pub fn query_security_bits(&self) -> f64 {
        REGIME.queries_error(self.params.log_inv_rate, self.num_queries)
    }

    /// Bits of security the fold and sumcheck rounds leave, rounded down.
    ///
    /// This is the ceiling on the target: [`Self::try_new`] rejects a `security_level` above
    /// it, because no query count buys those bits back.
    #[must_use]
    pub const fn field_security_bits(&self) -> usize {
        field_security_bits_at(self.log_domain_size(), self.num_fold_rounds())
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField128;
    use p3_field::Field;
    use p3_security::InstanceShape;
    use p3_security::fri::{FriRegime, commit_phase_error_udr};

    use super::{ALPHABET_BITS, BinaryPcsConfig, BinaryPcsConfigError, BinaryPcsParams};

    const fn params() -> BinaryPcsParams {
        BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 16,
            security_level: 100,
        }
    }

    #[test]
    fn the_alphabet_width_matches_the_field() {
        assert_eq!(ALPHABET_BITS, BinaryField128::bits());
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

    /// The fold term this crate charges must agree with the one `p3-security` states for the
    /// unique-decoding commit phase, so the two cannot drift apart silently. Arity 2 is
    /// `max_log_arity = 1`, and every fold round here runs at `pow_bits = 0`.
    ///
    /// This crate's value must never be the larger of the two — it is the one `try_new`
    /// enforces — and must stay within a rounding of it: the two differ only by this crate's
    /// extra sumcheck term, which the codeword term dwarfs, and by its rounding down to a
    /// whole bit.
    #[test]
    fn the_field_security_agrees_with_p3_security() {
        for &num_variables in &[10usize, 16, 20, 24] {
            let config = BinaryPcsConfig::try_new(num_variables, params()).unwrap();
            let regime = FriRegime {
                log_blowup: config.log_inv_rate(),
                num_queries: config.num_queries(),
                log_final_poly_len: config.log_final_len(),
                max_log_arity: 1,
                commit_pow_bits: 0,
                query_pow_bits: config.pow_bits(),
            };
            let shape = InstanceShape {
                log_trace_length: config.num_variables(),
                modulus_bits: ALPHABET_BITS,
                collision_resistance: ALPHABET_BITS,
                num_batched_functions: 1,
            };
            let reference = commit_phase_error_udr(&regime, &shape)
                .expect("arity 2 folds, so there is a commit-phase round")
                .bits();
            let ours = config.field_security_bits() as f64;
            assert!(
                ours <= reference && reference - ours < 1.5,
                "num_variables={num_variables}: {ours} bits does not track p3-security's {reference}"
            );
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
    /// fold and sumcheck rounds alone cap the achievable security at 105 bits.
    #[test]
    fn rejects_a_security_level_the_field_cannot_deliver() {
        let mut p = params();
        p.security_level = 120;
        assert_eq!(
            BinaryPcsConfig::try_new(20, p),
            Err(BinaryPcsConfigError::SecurityLevelExceedsFieldCapacity {
                security_level: 120,
                max: 105,
            })
        );

        // One bit below the cap is accepted, so the boundary is the stated one and not an
        // accident of a much looser bound.
        p.security_level = 105;
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
