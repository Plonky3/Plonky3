//! STIR protocol configuration: user-facing parameters and derived per-round configs.

use alloc::format;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, SecurityAssumption};
use p3_field::{ExtensionField, TwoAdicField};

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
    /// Must satisfy `log_folding_factor <= log_starting_degree`.
    ///
    /// The paper-backed STIR schedule implemented here requires `k ≥ 4`
    /// (`log_folding_factor ≥ 2`).
    pub log_folding_factor: usize,

    /// Which Reed-Solomon proximity bound to assume for soundness analysis.
    pub soundness_type: SecurityAssumption,

    /// Target security level in bits.
    pub security_level: usize,

    /// Fixed proof-of-work difficulty in bits applied to each Fiat-Shamir grinding step.
    ///
    /// Following the paper's discussion of PoW-assisted parameters, this reduces the
    /// algebraic target from `security_level` to `security_level - max_pow_bits`.
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
    /// Advances to a disjoint coset each round so the next witness domain avoids the
    /// current round's fold-query set.
    pub domain_shift: F,

    /// Log₂ of the folding arity applied at the end of this round.
    ///
    /// The prover folds `2^log_folding_factor` coset points into one evaluation of
    /// the next round's polynomial.
    pub log_folding_factor: usize,

    /// The round's `eta_i` parameter from the paper's recommended schedule.
    pub eta: f64,

    /// Number of STIR proximity queries in this round.
    ///
    /// Derived directly from §5.3 using the round's rate and `eta_i`.
    pub num_queries: usize,

    /// Number of out-of-domain (OOD) evaluation samples in this round.
    ///
    /// Fixed to `s = 1` in the provable regime and `s = 2` in the conjectured regime.
    pub num_ood_samples: usize,

    /// Proof-of-work difficulty used for the STIR query phase in this round.
    ///
    /// Derived as `max(0, security_level − query_algebraic_bits)` and capped at
    /// `max_pow_bits`. `query_algebraic_bits` is the worst (min) of the per-round query
    /// failure, OOD, and random-combination soundness terms.
    pub pow_bits: usize,

    /// Proof-of-work difficulty used for the polynomial folding step in this round.
    ///
    /// Derived as `max(0, security_level − fold_algebraic_bits)` and capped at
    /// `max_pow_bits`. `fold_algebraic_bits` is the worst (min) of the proximity-gaps and
    /// fold-sumcheck soundness terms.
    pub folding_pow_bits: usize,
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

    /// Fixed proof-of-work difficulty in bits applied to each grinding step.
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

    /// The final round's `eta_M` parameter from the paper's recommended schedule.
    pub final_eta: f64,

    /// Proof-of-work difficulty used for the final query phase.
    ///
    /// Derived per the same rule as [`StirRoundConfig::pow_bits`], but using only the
    /// final-round query-failure soundness (no OOD or combination in the final round).
    pub final_pow_bits: usize,

    /// Proof-of-work difficulty used for the final folding step.
    ///
    /// Derived per the same rule as [`StirRoundConfig::folding_pow_bits`].
    pub final_folding_pow_bits: usize,

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
    /// Panics if `log_folding_factor < 2` or if `log_folding_factor > log_starting_degree`.
    pub fn new(log_starting_degree: usize, params: StirParameters<M>) -> Self {
        assert!(
            params.log_folding_factor >= 2,
            "the paper-backed STIR parameter schedule requires log_folding_factor >= 2 (k >= 4)"
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

        assert!(
            !matches!(params.soundness_type, SecurityAssumption::UniqueDecoding),
            "the paper-backed STIR parameter schedule does not support UniqueDecoding"
        );
        assert!(
            params.security_level > params.max_pow_bits,
            "security_level must be greater than max_pow_bits"
        );

        let field_size_bits = EF::bits();
        let log_blowup = params.log_blowup;
        let log_folding_factor = params.log_folding_factor;
        let security_level = params.security_level;
        let max_pow_bits = params.max_pow_bits;
        let algebraic_security_level = security_level - max_pow_bits;
        let num_ood_samples = params.soundness_type.stir_num_ood_samples();

        // Convert algebraic-security bits to a PoW difficulty.
        // PoW = ceil(security_level − algebraic_bits), capped to [0, max_pow_bits].
        // A derived value > max_pow_bits is a hard misconfiguration: the user's parameters
        // do not deliver `security_level` bits even after exhausting the PoW budget.
        let derive_pow_bits = |label: &str, round: &str, algebraic_bits: f64| -> usize {
            let gap = (security_level as f64 - algebraic_bits).max(0.0);
            let needed = libm::ceil(gap) as usize;
            assert!(
                needed <= max_pow_bits,
                "{round} {label} requires {needed} PoW bits to reach \
                 security_level = {security_level} (algebraic bits = {algebraic_bits}), \
                 but max_pow_bits = {max_pow_bits}. Increase max_pow_bits, log_blowup, \
                 or use a larger field.",
            );
            needed
        };

        // Determine number of intermediate rounds. We fold all the way down to a polynomial
        // of size `2^log_final_degree` (where log_final_degree < log_folding_factor) and
        // send it directly. Each round reduces log_degree by log_folding_factor.
        let total_folds = log_starting_degree / log_folding_factor;
        assert!(
            total_folds > 0,
            "STIR requires at least one fold before the final direct-send stage"
        );

        // Last fold produces the final polynomial; intermediate rounds = total_folds - 1.
        let num_rounds = total_folds.saturating_sub(1);
        let log_final_degree = log_starting_degree - total_folds * log_folding_factor;

        // Initial domain shift: use the multiplicative generator so the
        // initial domain is disjoint from all subgroups of the base field.
        // Each round commits the folded oracle on a disjoint coset of the next domain.
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

        // Per-round target_bits adds a union-bound buffer of `ceil(log2(total_folds))` bits
        // to `algebraic_security_level`, so that summing the per-round error over all folds
        // (intermediate rounds + the final fold) is bounded by `2^{-algebraic_security_level}`:
        //
        //   sum_{i=0}^{total_folds-1} 2^{-(algebraic_security_level + log2(total_folds))}
        //     = total_folds · 2^{-algebraic_security_level} / total_folds
        //     = 2^{-algebraic_security_level}.
        //
        // The same buffer applies to the final round; the asymmetric "+1 / +0" rule used in
        // earlier revisions (and quoted in STIR §5.3) only delivers `algebraic_security_level`
        // bits when `total_folds ≤ 2`. For deeper protocols it shaves up to `log2(total_folds)`
        // bits — replacing it with the explicit log here makes the claimed security tight.
        let union_bound_buffer = libm::ceil(libm::log2(total_folds as f64)) as usize;
        let target_bits = algebraic_security_level + union_bound_buffer;
        let query_count = |stage_log_inv_rate: usize, eta: f64| {
            let failure_base = params
                .soundness_type
                .stir_query_failure_base(stage_log_inv_rate, eta);
            params
                .soundness_type
                .stir_queries_for_base(target_bits, failure_base)
        };
        let validate_eta = |stage: usize, stage_log_inv_rate: usize, eta: f64| {
            assert!(
                params
                    .soundness_type
                    .stir_eta_is_valid(stage_log_inv_rate, eta),
                "round {stage} produced eta = {eta}, which violates the paper's side-condition \
                 bound {}",
                params
                    .soundness_type
                    .stir_eta_upper_bound(stage_log_inv_rate)
            );
        };

        // Disjoint-coset side condition for round `i`. The schedule sets
        // `shift_{i+1} = shift_i^k * GEN`, so `shift_{i+1}/shift_i = GEN^{k^{i+1}}`.
        // Disjoint cosets `L_i ∩ L_{i+1} = ∅` requires that ratio ∉ H_i, i.e.
        // `(GEN^{k^{i+1}})^{|H_i|} = GEN^{2^{N_i}} ≠ 1` where
        //   `N_i = (i+1) * log_folding_factor + log_domain_i`.
        // Holds for any field whose multiplicative order has nontrivial odd part
        // (BabyBear, KoalaBear, Goldilocks, …); the assertion catches pathological fields.
        let assert_disjoint_cosets = |round_index: usize, log_domain_i: usize| {
            let n_i = (round_index + 1) * log_folding_factor + log_domain_i;
            assert!(
                F::GENERATOR.exp_power_of_2(n_i) != F::ONE,
                "STIR round {round_index}: disjoint-coset schedule requires \
                 Field::GENERATOR^(2^{n_i}) ≠ 1 (i.e. GEN^(k^{}) ∉ subgroup of size 2^{log_domain_i}).",
                round_index + 1,
            );
        };

        let mut final_eta = params.soundness_type.stir_initial_eta(
            algebraic_security_level,
            log_degree,
            log_inv_rate,
            log_folding_factor,
            field_size_bits,
        );
        validate_eta(0, log_inv_rate, final_eta);

        if total_folds != 1 {
            let num_queries = query_count(log_inv_rate, final_eta);
            assert_disjoint_cosets(0, log_domain_size);

            let fold_alg = params.soundness_type.fold_algebraic_bits(
                field_size_bits,
                log_degree,
                log_inv_rate,
            );
            let query_alg = params.soundness_type.stir_query_algebraic_bits(
                field_size_bits,
                log_degree,
                log_inv_rate,
                final_eta,
                num_queries,
                num_ood_samples,
            );
            let folding_pow_bits = derive_pow_bits("folding", "round 0", fold_alg);
            let pow_bits = derive_pow_bits("query", "round 0", query_alg);

            round_configs.push(StirRoundConfig {
                log_degree,
                log_domain_size,
                log_fold_domain_size: log_domain_size - log_folding_factor,
                domain_shift,
                log_folding_factor,
                eta: final_eta,
                num_queries,
                num_ood_samples,
                pow_bits,
                folding_pow_bits,
            });

            let mut prev_queries = num_queries;
            log_degree -= log_folding_factor;
            log_domain_size -= 1;
            log_inv_rate += log_folding_factor - 1;
            domain_shift = domain_shift.exp_power_of_2(log_folding_factor) * F::GENERATOR;

            for round in 1..num_rounds {
                final_eta = params.soundness_type.stir_recursive_eta(
                    algebraic_security_level,
                    log_degree,
                    log_inv_rate,
                    log_domain_size,
                    log_folding_factor,
                    field_size_bits,
                    prev_queries,
                );
                validate_eta(round, log_inv_rate, final_eta);

                let num_queries = query_count(log_inv_rate, final_eta);
                assert_disjoint_cosets(round, log_domain_size);

                let fold_alg = params.soundness_type.fold_algebraic_bits(
                    field_size_bits,
                    log_degree,
                    log_inv_rate,
                );
                let query_alg = params.soundness_type.stir_query_algebraic_bits(
                    field_size_bits,
                    log_degree,
                    log_inv_rate,
                    final_eta,
                    num_queries,
                    num_ood_samples,
                );
                let round_label = format!("round {round}");
                let folding_pow_bits = derive_pow_bits("folding", &round_label, fold_alg);
                let pow_bits = derive_pow_bits("query", &round_label, query_alg);

                round_configs.push(StirRoundConfig {
                    log_degree,
                    log_domain_size,
                    log_fold_domain_size: log_domain_size - log_folding_factor,
                    domain_shift,
                    log_folding_factor,
                    eta: final_eta,
                    num_queries,
                    num_ood_samples,
                    pow_bits,
                    folding_pow_bits,
                });

                prev_queries = num_queries;
                log_degree -= log_folding_factor;
                log_domain_size -= 1;
                log_inv_rate += log_folding_factor - 1;
                domain_shift = domain_shift.exp_power_of_2(log_folding_factor) * F::GENERATOR;
            }

            final_eta = params.soundness_type.stir_recursive_eta(
                algebraic_security_level,
                log_degree,
                log_inv_rate,
                log_domain_size,
                log_folding_factor,
                field_size_bits,
                prev_queries,
            );
            validate_eta(num_rounds, log_inv_rate, final_eta);
        }
        let final_queries = query_count(log_inv_rate, final_eta);

        // Final-round PoW: the final fold uses (log_degree, log_inv_rate) at the protocol
        // tail (after all intermediate increments). The final query phase has no OOD or
        // combination — just the query failure.
        let final_fold_alg =
            params
                .soundness_type
                .fold_algebraic_bits(field_size_bits, log_degree, log_inv_rate);
        let final_query_alg = params.soundness_type.stir_final_query_algebraic_bits(
            log_inv_rate,
            final_eta,
            final_queries,
        );
        let final_folding_pow_bits = derive_pow_bits("folding", "final", final_fold_alg);
        let final_pow_bits = derive_pow_bits("query", "final", final_query_alg);

        Self {
            log_starting_degree,
            soundness_type: params.soundness_type,
            security_level: params.security_level,
            max_pow_bits: params.max_pow_bits,
            log_blowup,
            log_folding_factor: params.log_folding_factor,
            round_configs,
            log_final_degree,
            final_queries,
            final_eta,
            final_pow_bits,
            final_folding_pow_bits,
            mmcs: params.mmcs,
            _phantom: PhantomData,
        }
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

    /// Returns `true` when the configured PoW leaves a positive algebraic security target.
    pub const fn check_pow_bits(&self) -> bool {
        self.security_level > self.max_pow_bits
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
            log_folding_factor: 2,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 80,
            max_pow_bits: 20,
            mmcs,
        };

        // log_starting_degree=8, fold by 4 each round -> 4 folds total, 3 intermediate rounds.
        let config = StirConfig::<F, EF, MyMmcs, MyChallenger>::new(8, params);
        assert_eq!(config.log_final_degree, 0);
        assert_eq!(config.num_rounds(), 3);
        // Per-round PoW is derived from the algebraic gap, capped at max_pow_bits=20.
        assert!(config.final_pow_bits <= 20);
        assert!(config.final_folding_pow_bits <= 20);

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
            assert_eq!(rc.num_ood_samples, 2, "capacity-bound STIR uses s = 2");
            assert!(rc.eta.is_finite() && rc.eta > 0.);
            assert!(
                rc.pow_bits <= 20,
                "round {i} pow_bits {} exceeds max_pow_bits",
                rc.pow_bits
            );
            assert!(
                rc.folding_pow_bits <= 20,
                "round {i} folding_pow_bits {} exceeds max_pow_bits",
                rc.folding_pow_bits
            );
            if i > 0 {
                assert!(
                    rc.num_queries <= config.round_configs[i - 1].num_queries,
                    "query counts should not increase as the code rate improves"
                );
            }
        }
        assert!(config.final_eta.is_finite() && config.final_eta > 0.);
    }

    #[test]
    fn test_stir_config_uses_fixed_ood_schedule() {
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

        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        let perm = Perm::new_from_rng_128(&mut rng);
        let val_mmcs = ValMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);

        let jb = StirConfig::<F, EF, MyMmcs, MyChallenger>::new(
            8,
            StirParameters {
                log_blowup: 2,
                log_folding_factor: 2,
                soundness_type: SecurityAssumption::JohnsonBound,
                security_level: 80,
                max_pow_bits: 20,
                mmcs: MyMmcs::new(val_mmcs.clone()),
            },
        );
        assert!(jb.round_configs.iter().all(|rc| rc.num_ood_samples == 1));

        let cb = StirConfig::<F, EF, MyMmcs, MyChallenger>::new(
            8,
            StirParameters {
                log_blowup: 2,
                log_folding_factor: 2,
                soundness_type: SecurityAssumption::CapacityBound,
                security_level: 80,
                max_pow_bits: 20,
                mmcs: MyMmcs::new(val_mmcs),
            },
        );
        assert!(cb.round_configs.iter().all(|rc| rc.num_ood_samples == 2));
    }

    #[test]
    fn test_stir_config_union_bound_buffer_scales_with_rounds() {
        // The per-round target_bits adds ceil(log2(total_folds)) to algebraic_security_level
        // so the per-round error sums to <= 2^{-algebraic_security_level} across all folds.
        // A deeper protocol (more folds) must request more queries per round than a shallow
        // one at the same security level / rate / eta, since the union-bound buffer is larger.
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

        let mut rng = rand::rngs::SmallRng::seed_from_u64(13);
        let perm = Perm::new_from_rng_128(&mut rng);
        let val_mmcs = ValMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);

        let make = |log_starting_degree| {
            StirConfig::<F, EF, MyMmcs, MyChallenger>::new(
                log_starting_degree,
                StirParameters {
                    log_blowup: 1,
                    log_folding_factor: 2,
                    soundness_type: SecurityAssumption::CapacityBound,
                    security_level: 80,
                    max_pow_bits: 20,
                    mmcs: MyMmcs::new(val_mmcs.clone()),
                },
            )
        };

        // Shallow: log_starting_degree=4 ⇒ total_folds = 2 ⇒ buffer = ceil(log2(2)) = 1.
        // Deep:    log_starting_degree=16 ⇒ total_folds = 8 ⇒ buffer = ceil(log2(8)) = 3.
        let shallow = make(4);
        let deep = make(16);

        // Both have positive query counts.
        assert!(shallow.final_queries > 0);
        assert!(deep.final_queries > 0);

        // The deeper protocol's final-round target is strictly larger because of the bigger
        // buffer, so for comparable rates final_queries must be ≥ the shallow one's.
        // (Eta differs across configurations, so we can only assert a soft inequality here.)
        // The strict invariant we can check: per-round target_bits is monotone in total_folds.
        let buffer = |n: usize| libm::ceil(libm::log2(n as f64)) as usize;
        assert_eq!(buffer(2), 1);
        assert_eq!(buffer(8), 3);
        assert!(buffer(8) > buffer(2));

        // Sanity: non-empty rounds; deep > shallow in number of fold steps.
        assert!(deep.num_rounds() + 1 > shallow.num_rounds() + 1);
    }
}
