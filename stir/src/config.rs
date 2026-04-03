//! STIR protocol configuration: user-facing parameters and derived per-round configs.

use alloc::vec::Vec;
use core::marker::PhantomData;

use libm;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, SecurityAssumption};
use p3_field::{ExtensionField, TwoAdicField};

/// Maximum log-degree at which the prover sends the polynomial directly.
///
/// When the remaining polynomial degree is at most `2^MAX_LOG_DEGREE_DIRECT`,
/// the prover sends it in coefficient form instead of producing another STIR round.
pub const MAX_LOG_DEGREE_DIRECT: usize = 0;

/// User-facing STIR protocol parameters.
///
/// These are the inputs from which the full [`StirConfig`] is derived.
#[derive(Clone, Debug)]
pub struct StirParameters<M> {
    /// Log₂ of the inverse rate of the initial Reed-Solomon code.
    ///
    /// The initial evaluation domain has size `2^(log_starting_degree + log_blowup)`.
    /// The rate **improves** each round: degree drops by `k = 2^log_folding_factor`
    /// while the domain drops by only 2, so the effective inverse rate increases by
    /// `log_folding_factor - 1` per round.
    pub log_blowup: usize,

    /// Log₂ of the folding factor applied in every round.
    ///
    /// Each round folds `2^log_folding_factor` evaluation points into one,
    /// reducing the degree by that same factor, while the committed domain is halved
    /// (LDE step). This decoupling causes the code rate to improve each round.
    /// Must satisfy `log_folding_factor <= log_starting_degree - MAX_LOG_DEGREE_DIRECT`.
    ///
    /// **Note**: `log_folding_factor = 1` (arity k = 2) gives FRI-like behaviour with no rate
    /// improvement per round. Theorem 5.1 in the STIR paper requires k ≥ 4
    /// (`log_folding_factor ≥ 2`) for the stated regime. k = 2 is supported but operates
    /// outside the paper's theorem.
    pub log_folding_factor: usize,

    /// Which Reed-Solomon proximity bound to assume for soundness analysis.
    pub soundness_type: SecurityAssumption,

    /// Target security level in bits.
    pub security_level: usize,

    /// Maximum allowed proof-of-work difficulty in bits.
    ///
    /// If any derived round PoW exceeds this, the configuration is considered infeasible
    /// (check via [`StirConfig::check_pow_bits`]).
    pub max_pow_bits: usize,

    /// Merkle tree commitment scheme for codeword commitments.
    pub mmcs: M,
}

/// Derived configuration for a single STIR round.
///
/// All values are computed from [`StirParameters`] and the accumulated state
/// from prior rounds.
#[derive(Debug, Clone)]
pub struct StirRoundConfig<F> {
    /// Log₂ of the degree of the polynomial to be proximity-tested in this round.
    ///
    /// The polynomial has degree `< 2^log_degree` before folding.
    pub log_degree: usize,

    /// Log₂ of the committed codeword size in this round.
    ///
    /// Starts at `log_starting_degree + log_blowup` and decreases by 1 each round,
    /// since the committed oracle is an LDE of the fold onto a domain of half the
    /// current size (the rate improvement mechanism).
    pub log_domain_size: usize,

    /// Log₂ of the fold output size before LDE.
    ///
    /// `log_fold_domain_size = log_domain_size - log_folding_factor`.
    /// This is the size of the codeword produced by folding by `k`, before extending
    /// to the committed LDE domain of size `2^log_domain_size / 2`.
    pub log_fold_domain_size: usize,

    /// Shift of the evaluation domain (base field element).
    ///
    /// The domain is `shift * <g>` where `g` is the two-adic generator of order
    /// `2^log_domain_size`. Shifts stay in the base field to enable base-field FFTs.
    /// Advances as `shift^k` each round (where `k = 2^log_folding_factor`).
    pub domain_shift: F,

    /// Log₂ of the folding arity applied at the end of this round.
    ///
    /// The prover folds `2^log_folding_factor` coset points into one evaluation of
    /// the next round's polynomial.
    pub log_folding_factor: usize,

    /// Number of STIR proximity queries in this round.
    ///
    /// Decreases each round as the code rate improves (more inverse rate → fewer
    /// queries needed to achieve the same security level).
    pub num_queries: usize,

    /// Number of out-of-domain (OOD) evaluation samples in this round.
    ///
    /// Computed using the post-fold (improved) rate and next-round degree.
    pub num_ood_samples: usize,

    /// Proof-of-work difficulty (bits) for the STIR query phase in this round.
    ///
    /// Clamped to [`StirParameters::max_pow_bits`]. Use `required_pow_bits` to detect
    /// whether the cap prevents reaching the full security target.
    pub pow_bits: usize,

    /// Proof-of-work difficulty (bits) for the polynomial folding step in this round.
    ///
    /// Clamped to [`StirParameters::max_pow_bits`]. Use `required_folding_pow_bits` to detect
    /// whether the cap prevents reaching the full security target.
    pub folding_pow_bits: usize,

    /// Unclamped PoW difficulty required to reach the security target for the query phase.
    ///
    /// If this exceeds `max_pow_bits`, the round cannot achieve the full security target
    /// at the configured PoW cap. Detected by [`StirConfig::check_pow_bits`].
    pub required_pow_bits: usize,

    /// Unclamped PoW difficulty required to reach the security target for the folding step.
    pub required_folding_pow_bits: usize,
}

/// Fully derived STIR protocol configuration.
///
/// Built from [`StirParameters`] plus the starting polynomial degree.
/// Contains all precomputed values needed by the prover and verifier.
#[derive(Debug, Clone)]
pub struct StirConfig<F, EF, M, Challenger> {
    /// Log₂ of the degree of the initial polynomial.
    pub log_starting_degree: usize,

    /// Which Reed-Solomon proximity bound is assumed for soundness.
    pub soundness_type: SecurityAssumption,

    /// Target security level in bits.
    pub security_level: usize,

    /// Maximum allowed proof-of-work difficulty in bits.
    pub max_pow_bits: usize,

    /// Log₂ of the inverse rate of the initial RS code.
    ///
    /// The effective inverse rate increases by `log_folding_factor - 1` each round.
    pub log_blowup: usize,

    /// Log₂ of the folding arity. Constant across all rounds.
    pub log_folding_factor: usize,

    /// Per-round derived configurations for each intermediate STIR round.
    pub round_configs: Vec<StirRoundConfig<F>>,

    /// Log₂ of the degree of the final (directly-sent) polynomial.
    pub log_final_degree: usize,

    /// Number of STIR proximity queries in the final round.
    pub final_queries: usize,

    /// Proof-of-work difficulty (bits) for the final query phase.
    ///
    /// Clamped to `max_pow_bits`. See `required_final_pow_bits` for the unclamped value.
    pub final_pow_bits: usize,

    /// Proof-of-work difficulty (bits) for the final folding step.
    ///
    /// Clamped to `max_pow_bits`. See `required_final_folding_pow_bits` for the unclamped value.
    pub final_folding_pow_bits: usize,

    /// Unclamped PoW bits required to reach the security target for the final query phase.
    pub required_final_pow_bits: usize,

    /// Unclamped PoW bits required to reach the security target for the final folding step.
    pub required_final_folding_pow_bits: usize,

    /// Merkle tree commitment scheme.
    pub mmcs: M,

    _phantom: PhantomData<(F, EF, Challenger)>,
}

impl<F, EF, M, Challenger> StirConfig<F, EF, M, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    M: Mmcs<EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Derive a full STIR configuration from user-facing parameters.
    ///
    /// `log_starting_degree` is log₂ of the degree of the polynomial to commit to.
    ///
    /// # Panics
    ///
    /// Panics if `log_folding_factor == 0` (would divide by zero when computing total folds),
    /// or if `log_folding_factor > log_starting_degree`.
    pub fn new(log_starting_degree: usize, params: StirParameters<M>) -> Self {
        assert!(
            params.log_folding_factor >= 1,
            "log_folding_factor must be at least 1; got 0, which would divide by zero when \
             computing total_folds."
        );
        assert!(
            params.log_folding_factor <= log_starting_degree,
            "Folding factor ({}) must be <= starting degree ({}).",
            params.log_folding_factor,
            log_starting_degree
        );

        assert!(
            log_starting_degree + params.log_blowup <= F::TWO_ADICITY,
            "Initial domain size 2^{} exceeds the two-adicity of the base field ({}).",
            log_starting_degree + params.log_blowup,
            F::TWO_ADICITY,
        );

        // PoW contributes independently, so the algebraic protocol only needs
        // to cover the remaining security gap.
        let protocol_security_level = params.security_level.saturating_sub(params.max_pow_bits);

        let field_size_bits = EF::bits();
        let log_blowup = params.log_blowup;
        let log_folding_factor = params.log_folding_factor;

        // Determine number of intermediate rounds.
        // We fold until log_degree <= MAX_LOG_DEGREE_DIRECT, then send directly.
        // Each round reduces log_degree by log_folding_factor.
        let foldable_bits = log_starting_degree.saturating_sub(MAX_LOG_DEGREE_DIRECT);
        let total_folds = foldable_bits / log_folding_factor;

        // Last fold produces the final polynomial; intermediate rounds = total_folds - 1.
        let num_rounds = total_folds.saturating_sub(1);
        let log_final_degree = log_starting_degree - total_folds * log_folding_factor;

        // Initial domain shift: use the multiplicative generator so the
        // initial domain is disjoint from all subgroups of the base field.
        // Each round: new_shift = old_shift^(2^log_folding_factor) = old_shift^k.
        // This is the shift of the FOLD domain, which equals the LDE domain shift.
        let initial_shift = F::GENERATOR;

        let mut round_configs = Vec::with_capacity(num_rounds);
        let mut log_degree = log_starting_degree;
        // The committed codeword's domain starts at the full initial domain and halves
        // each round (rate improvement: degree drops by k, domain drops by 2).
        let mut log_domain_size = log_starting_degree + log_blowup;
        // The effective inverse rate starts at log_blowup and increases by
        // (log_folding_factor - 1) each round.
        let mut log_inv_rate = log_blowup;
        let mut domain_shift = initial_shift;

        for _round in 0..num_rounds {
            let log_fold_domain_size = log_domain_size - log_folding_factor;

            // After the fold, degree drops by k and rate improves.
            let log_degree_next = log_degree - log_folding_factor;
            let next_log_inv_rate = log_inv_rate + (log_folding_factor - 1);

            // Queries use the CURRENT (pre-fold) rate.
            let num_queries = params
                .soundness_type
                .queries(protocol_security_level, log_inv_rate);

            // OOD samples and combination errors use the post-fold (improved) rate and degree.
            let num_ood_samples = params.soundness_type.determine_ood_samples(
                params.security_level,
                log_degree_next,
                next_log_inv_rate,
                field_size_bits,
            );

            let query_error = params
                .soundness_type
                .queries_error(log_inv_rate, num_queries);

            let combination_error = params.soundness_type.queries_combination_error(
                field_size_bits,
                log_degree_next,
                next_log_inv_rate,
                num_ood_samples,
                num_queries,
            );

            let required_pow_bits = 0_f64
                .max(params.security_level as f64 - query_error.min(combination_error))
                as usize;
            let pow_bits = required_pow_bits.min(params.max_pow_bits);

            // Folding PoW uses the post-fold rate (proximity gap of the new code).
            let required_folding_pow_bits = libm::ceil(params.soundness_type.folding_pow_bits(
                params.security_level,
                field_size_bits,
                log_degree_next,
                next_log_inv_rate,
            )) as usize;
            let folding_pow_bits = required_folding_pow_bits.min(params.max_pow_bits);

            round_configs.push(StirRoundConfig {
                log_degree,
                log_domain_size,
                log_fold_domain_size,
                domain_shift,
                log_folding_factor,
                num_queries,
                num_ood_samples,
                pow_bits,
                folding_pow_bits,
                required_pow_bits,
                required_folding_pow_bits,
            });

            // Advance state for the next round.
            log_degree -= log_folding_factor;
            // Domain halves each round (LDE step), not shrinks by k.
            log_domain_size -= 1;
            log_inv_rate = next_log_inv_rate;
            // New shift = old_shift^k (fold-domain shift = LDE-domain shift).
            domain_shift = domain_shift.exp_power_of_2(log_folding_factor);
        }

        // Final round parameters use the accumulated (improved) inverse rate.
        let final_queries = params
            .soundness_type
            .queries(protocol_security_level, log_inv_rate);

        let required_final_pow_bits = libm::ceil(
            0_f64.max(
                params.security_level as f64
                    - params
                        .soundness_type
                        .queries_error(log_inv_rate, final_queries),
            ),
        ) as usize;
        let final_pow_bits = required_final_pow_bits.min(params.max_pow_bits);

        let required_final_folding_pow_bits =
            libm::ceil(0_f64.max(params.security_level as f64 - (field_size_bits - 1) as f64))
                as usize;
        let final_folding_pow_bits = required_final_folding_pow_bits.min(params.max_pow_bits);

        let config = Self {
            log_starting_degree,
            soundness_type: params.soundness_type,
            security_level: params.security_level,
            max_pow_bits: params.max_pow_bits,
            log_blowup,
            log_folding_factor: params.log_folding_factor,
            round_configs,
            log_final_degree,
            final_queries,
            final_pow_bits,
            final_folding_pow_bits,
            required_final_pow_bits,
            required_final_folding_pow_bits,
            mmcs: params.mmcs,
            _phantom: PhantomData,
        };

        assert!(
            config.check_pow_bits(),
            "max_pow_bits ({}) is insufficient to reach the {} -bit security target. \
             Increase max_pow_bits or lower the security_level.",
            config.max_pow_bits,
            config.security_level,
        );

        config
    }

    /// Log₂ of the initial evaluation domain size.
    pub const fn log_starting_domain_size(&self) -> usize {
        self.log_starting_degree + self.log_blowup
    }

    /// Number of intermediate STIR rounds (excluding the final send).
    pub const fn num_rounds(&self) -> usize {
        self.round_configs.len()
    }

    /// Number of codeword commitments produced (one per round + one for the input).
    pub const fn num_commitments(&self) -> usize {
        self.num_rounds() + 1
    }

    /// Size of the final polynomial (number of coefficients).
    pub const fn final_poly_len(&self) -> usize {
        1 << self.log_final_degree
    }

    /// Returns `true` if `max_pow_bits` is sufficient to reach the security target in every round.
    ///
    /// When this returns `false`, the algebraic soundness gap in some round exceeds
    /// `max_pow_bits`, meaning the actual security level is below `security_level`. Callers
    /// should either increase `max_pow_bits`, increase the blowup factor, or use more queries.
    pub fn check_pow_bits(&self) -> bool {
        if self.required_final_pow_bits > self.max_pow_bits
            || self.required_final_folding_pow_bits > self.max_pow_bits
        {
            return false;
        }
        self.round_configs.iter().all(|r| {
            r.required_pow_bits <= self.max_pow_bits
                && r.required_folding_pow_bits <= self.max_pow_bits
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stir_config_round_count() {
        use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
        use p3_challenger::DuplexChallenger;
        use p3_commit::ExtensionMmcs;
        use p3_field::Field;
        use p3_field::extension::BinomialExtensionField;
        use p3_merkle_tree::MerkleTreeMmcs;
        use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
        use rand::SeedableRng;

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;
        type Perm = Poseidon2BabyBear<16>;
        type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
        type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
        type PackedF = <F as Field>::Packing;
        type ValMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
        type MyMmcs = ExtensionMmcs<F, EF, ValMmcs>;
        type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut rng);
        let val_mmcs = ValMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);
        let mmcs = MyMmcs::new(val_mmcs);

        let params = StirParameters {
            log_blowup: 1,
            log_folding_factor: 1,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 80,
            max_pow_bits: 20,
            mmcs,
        };

        // log_starting_degree=8, fold by 2 each round -> 8 rounds total, 7 intermediate
        let config = StirConfig::<F, EF, MyMmcs, MyChallenger>::new(8, params);
        assert_eq!(config.log_final_degree, 0);
        // total_folds = 8, num_rounds = 7
        assert_eq!(config.num_rounds(), 7);

        // For k=2, rate is constant (log_folding_factor - 1 = 0), so queries are constant.
        let q0 = config.round_configs[0].num_queries;
        for rc in &config.round_configs {
            assert_eq!(rc.num_queries, q0, "k=2 should have constant query count");
        }
        // Domain halves each round (not shrinks by k=2 since k=2 means no rate improvement).
        let initial_log_domain = 8 + 1; // log_starting_degree + log_blowup
        for (i, rc) in config.round_configs.iter().enumerate() {
            assert_eq!(
                rc.log_domain_size,
                initial_log_domain - i,
                "log_domain_size should decrease by 1 per round"
            );
            assert_eq!(
                rc.log_fold_domain_size,
                rc.log_domain_size - rc.log_folding_factor,
                "log_fold_domain_size = log_domain_size - log_folding_factor"
            );
        }
    }
}
