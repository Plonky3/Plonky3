use p3_security::SecurityAssumption;
use thiserror::Error;

/// Largest grinding request `BinaryChallenger<BinaryField128, _>::grind` accepts.
///
/// The challenger asserts `bits + 8 <= min(F::bits(), 64)`, so the counter width, not the
/// field width, is what binds at this level.
const MAX_POW_BITS: usize = 56;

/// The regime this scheme is analysed in.
///
/// Capacity-rate list decodability is refuted over characteristic 2 with `F_2`-subspace
/// domains, and the Cantor domain is one. The Johnson bound is not refuted by this: it is an
/// unconditional theorem whose radius those same counterexamples show to be tight, but
/// `p3-security` documents it as resting on a correlated-agreement conjecture, and it is
/// excluded here by choice, not by mathematics. It is fixed here rather than taken from the
/// caller, which is why this crate does not re-export `SecurityAssumption`.
const REGIME: SecurityAssumption = SecurityAssumption::UniqueDecoding;

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

/// Parameters derived from [`BinaryPcsParams`], the polynomial's arity and the folding factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryPcsConfig {
    params: BinaryPcsParams,
    num_variables: usize,
    folding: usize,
    num_queries: usize,
}

/// Why a [`BinaryPcsConfig`] could not be derived.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum BinaryPcsConfigError {
    /// Folding factors above zero are not supported.
    ///
    /// The commit phase reads a width-1 codeword; a wider one would be fed to a fold that
    /// assumes adjacent pairs lie in one column.
    #[error("folding factor {requested} is not supported; only 0 is")]
    FoldingNotSupported { requested: usize },

    /// The codeword length does not fit in a `usize`.
    ///
    /// A codeword is indexed by `usize`, so `log_len` must stay below `max_bits`
    /// (`usize::BITS`): at or past that width, computing the codeword's length as
    /// `1usize << log_len` is already invalid, regardless of how large the additive domain
    /// itself is.
    #[error("codeword length 2^{log_len} does not fit in a {max_bits}-bit usize index")]
    CodewordLengthExceedsUsize { log_len: usize, max_bits: usize },

    /// Reached only when `num_variables == 0`, since folding is fixed at zero: the committed
    /// polynomial has no variables at all, so there is nothing for a folding round to fold.
    #[error(
        "polynomial has no variables to fold (num_variables = {num_variables}, folding = {folding})"
    )]
    FoldingLeavesNoRounds {
        num_variables: usize,
        folding: usize,
    },

    /// The challenger cannot produce a witness this wide.
    #[error("grinding requests {requested} bits; the witness type admits at most {max}")]
    PowBitsExceedWitnessCapacity { requested: usize, max: usize },

    /// Grinding would consume the whole budget, leaving nothing for the queries to buy.
    #[error("security level {security_level} does not exceed the grinding budget {pow_bits}")]
    SecurityLevelBelowPowBits {
        security_level: usize,
        pow_bits: usize,
    },

    /// The derived query count was zero, which would accept any codeword.
    #[error("derived a query count of zero")]
    ZeroQueries,

    /// Reached with a regime other than unique decoding.
    ///
    /// Unreachable through the public API, which exposes no regime knob; this guards the
    /// private core against a future caller.
    #[error("only the unique-decoding regime is supported over F_2-subspace domains")]
    UnsupportedSoundnessRegime,
}

impl BinaryPcsConfig {
    /// Derive the configuration for a polynomial in `num_variables` variables committed with
    /// the given folding factor.
    ///
    /// # Errors
    ///
    /// Returns a [`BinaryPcsConfigError`] if the codeword length does not fit in a `usize`,
    /// if folding leaves no rounds, if grinding exceeds what the challenger can witness or the
    /// security budget, or if the derived query count is zero.
    pub fn try_new(
        num_variables: usize,
        folding: usize,
        params: BinaryPcsParams,
    ) -> Result<Self, BinaryPcsConfigError> {
        if folding != 0 {
            return Err(BinaryPcsConfigError::FoldingNotSupported { requested: folding });
        }

        if !matches!(REGIME, SecurityAssumption::UniqueDecoding) {
            return Err(BinaryPcsConfigError::UnsupportedSoundnessRegime);
        }

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

        if folding >= num_variables {
            return Err(BinaryPcsConfigError::FoldingLeavesNoRounds {
                num_variables,
                folding,
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

        // Grinding buys back `pow_bits`, so the queries only cover the remainder.
        let protocol_security_level = params.security_level - params.pow_bits;
        let num_queries = REGIME.queries(protocol_security_level, params.log_inv_rate);
        if num_queries == 0 {
            return Err(BinaryPcsConfigError::ZeroQueries);
        }

        Ok(Self {
            params,
            num_variables,
            folding,
            num_queries,
        })
    }

    /// Number of variables of the committed polynomial.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Number of variables bound inside a committed row.
    #[must_use]
    pub const fn folding(&self) -> usize {
        self.folding
    }

    /// Number of 2-to-1 codeword folds: one per residual sumcheck round.
    #[must_use]
    pub const fn num_fold_rounds(&self) -> usize {
        self.num_variables - self.folding
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
}

#[cfg(test)]
mod tests {
    use super::{BinaryPcsConfig, BinaryPcsConfigError, BinaryPcsParams};

    const fn params() -> BinaryPcsParams {
        BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 16,
            security_level: 100,
        }
    }

    #[test]
    fn derives_the_fold_schedule_and_query_count() {
        let config = BinaryPcsConfig::try_new(10, 0, params()).unwrap();
        assert_eq!(config.num_variables(), 10);
        assert_eq!(config.folding(), 0);
        // Folding is fixed at zero, so every variable folds the codeword.
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
        let config = BinaryPcsConfig::try_new(16, 0, params).unwrap();
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
        let few = BinaryPcsConfig::try_new(10, 0, high).unwrap().num_queries();
        let many = BinaryPcsConfig::try_new(10, 0, low).unwrap().num_queries();
        assert!(
            many > few,
            "a worse rate must demand more queries: {many} vs {few}"
        );
    }

    #[test]
    fn rejects_a_polynomial_with_no_variables() {
        assert_eq!(
            BinaryPcsConfig::try_new(0, 0, params()),
            Err(BinaryPcsConfigError::FoldingLeavesNoRounds {
                num_variables: 0,
                folding: 0,
            })
        );
    }

    #[test]
    fn rejects_a_nonzero_folding_factor() {
        assert_eq!(
            BinaryPcsConfig::try_new(10, 1, params()),
            Err(BinaryPcsConfigError::FoldingNotSupported { requested: 1 })
        );
    }

    #[test]
    fn rejects_grinding_beyond_the_witness_capacity() {
        let mut p = params();
        p.pow_bits = 57;
        assert_eq!(
            BinaryPcsConfig::try_new(10, 0, p),
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
            BinaryPcsConfig::try_new(num_variables, 0, params()),
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
            BinaryPcsConfig::try_new(10, 0, p),
            Err(BinaryPcsConfigError::ZeroQueries)
        );
    }

    #[test]
    fn rejects_a_security_level_at_or_below_the_grinding_budget() {
        let mut p = params();
        p.security_level = 16;
        assert_eq!(
            BinaryPcsConfig::try_new(10, 0, p),
            Err(BinaryPcsConfigError::SecurityLevelBelowPowBits {
                security_level: 16,
                pow_bits: 16,
            })
        );
    }
}
