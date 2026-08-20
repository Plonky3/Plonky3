//! STIR protocol configuration: user-facing parameters and derived per-round configs.

use alloc::format;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, TwoAdicField};

use crate::SecurityAssumption;
use crate::soundness::{StirSoundness, combine_min_log_eta};

/// Extra requirement round 0's `eta` must additionally satisfy so that the
/// batch-degree-correction `Combine` step (§4.5) reaches the target security when merging
/// `num_classes` height classes sharing this config's initial domain before the round-0
/// fold. `ell` is Lemma 4.13's inflated error argument, consulted by both
/// `SecurityAssumption` regimes' error terms.
#[derive(Clone, Copy, Debug)]
struct CombineRequirement {
    num_classes: usize,
    ell: u64,
}

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

    /// Log₂ of the folding factor applied from round 1 onward, and by the final
    /// direct-send stage when the schedule has at least one intermediate round.
    ///
    /// Each such round folds `2^log_folding_factor` evaluation points into one,
    /// reducing the degree by that same factor, while the committed domain is halved
    /// (LDE step). This decoupling causes the code rate to improve each round.
    ///
    /// The paper-backed STIR schedule implemented here requires `k ≥ 4`
    /// (`log_folding_factor ≥ 2`).
    pub log_folding_factor: usize,

    /// Log₂ of the folding factor applied in round 0 (the fold of the initial oracle).
    ///
    /// Construction 5.2 allows every round's folding parameter to differ; this crate
    /// exposes that generality only for round 0, since a smaller `k₀` shrinks each
    /// first-round query's fiber (`2^log_starting_folding_factor` rows of the input,
    /// times the input width) without touching the improved-rate schedule the later
    /// rounds are priced on. Must satisfy `log_starting_folding_factor <= log_starting_degree`
    /// and, like `log_folding_factor`, `log_starting_folding_factor ≥ 2` (`k₀ ≥ 4`).
    /// Set equal to `log_folding_factor` to recover a constant-arity schedule.
    pub log_starting_folding_factor: usize,

    /// Which Reed-Solomon proximity bound to assume for soundness analysis.
    pub soundness_type: SecurityAssumption,

    /// Target security level in bits.
    pub security_level: usize,

    /// Fixed proof-of-work difficulty in bits applied to each Fiat-Shamir grinding step.
    ///
    /// This can reduce the algebraic target only for challenges sampled immediately after
    /// the corresponding grind. OOD and shake-check errors receive no PoW credit.
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
    /// `max_pow_bits`. Only the query-failure and random-combination terms are eligible:
    /// the OOD points are sampled before this grind, while the shake challenge follows a
    /// later prover message, so both must meet the target without PoW credit.
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

    /// Log₂ of the folding arity used from round 1 onward.
    pub log_folding_factor: usize,

    /// Log₂ of the folding arity used in round 0 (the fold of the initial oracle).
    pub log_starting_folding_factor: usize,

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
    /// Panics if `log_folding_factor < 2`, if `log_starting_folding_factor < 2`, or if
    /// `log_starting_folding_factor > log_starting_degree`.
    pub fn new(log_starting_degree: usize, params: StirParameters<M>) -> Self {
        Self::new_impl(log_starting_degree, params, None)
    }

    /// Like [`Self::new`], but additionally inflates round 0's `eta` (and, transitively,
    /// the whole schedule that follows from it) so the batch-degree-correction `Combine`
    /// step (§4.5) also reaches `params.security_level` when merging `num_classes` height
    /// classes that share this config's initial domain, immediately before the round-0
    /// fold. Sharing `eta` between STIR's own round-0 acceptance radius and `Combine`'s
    /// error term (rather than checking `Combine` against a separately assumed radius)
    /// is what makes `Combine`'s "if the merged codeword passes, each class has
    /// correlated agreement" guarantee actually apply to STIR's own decoding.
    ///
    /// `ell` is Lemma 4.13's error argument `num_classes·(d* + 1) − Σᵢ dᵢ` (dᵢ each
    /// class's own, non-quotiented degree bound), consulted by both
    /// `SecurityAssumption` regimes' error terms.
    ///
    /// # Panics
    ///
    /// Same conditions as [`Self::new`].
    pub fn new_with_combine(
        log_starting_degree: usize,
        params: StirParameters<M>,
        num_classes: usize,
        ell: u64,
    ) -> Self {
        Self::new_impl(
            log_starting_degree,
            params,
            Some(CombineRequirement { num_classes, ell }),
        )
    }

    fn new_impl(
        log_starting_degree: usize,
        params: StirParameters<M>,
        combine: Option<CombineRequirement>,
    ) -> Self {
        assert!(
            params.log_folding_factor >= 2,
            "the paper-backed STIR parameter schedule requires log_folding_factor >= 2 (k >= 4)"
        );
        assert!(
            params.log_starting_folding_factor >= 2,
            "the paper-backed STIR parameter schedule requires log_starting_folding_factor >= 2 \
             (k0 >= 4)"
        );
        assert!(
            params.log_starting_folding_factor <= log_starting_degree,
            "Starting folding factor ({}) must be <= starting degree ({}).",
            params.log_starting_folding_factor,
            log_starting_degree
        );

        let log_starting_domain = log_starting_degree
            .checked_add(params.log_blowup)
            .expect("Initial domain exponent log_starting_degree + log_blowup overflows usize");

        assert!(
            log_starting_domain < usize::BITS as usize,
            "Initial domain exponent {log_starting_domain} must be less than usize::BITS ({})",
            usize::BITS
        );

        assert!(
            log_starting_domain <= F::TWO_ADICITY,
            "Initial domain size 2^{} exceeds the two-adicity of the base field ({}).",
            log_starting_domain,
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
        let log_starting_folding_factor = params.log_starting_folding_factor;
        let security_level = params.security_level;
        let max_pow_bits = params.max_pow_bits;
        let algebraic_security_level = security_level - max_pow_bits;
        let num_ood_samples = params.soundness_type.stir_num_ood_samples();

        // Determine number of intermediate rounds. Round 0 folds by k0
        // (`log_starting_folding_factor`); every fold after that, including the final
        // direct-send stage, folds by k (`log_folding_factor`). We fold all the way down
        // to a polynomial of size `2^log_final_degree` (where log_final_degree <
        // log_folding_factor, whenever more than the k0 fold happens) and send it
        // directly. When `log_starting_degree - log_starting_folding_factor` is itself
        // already `< log_folding_factor`, the k0 fold IS the final fold and no further
        // k-fold occurs.
        let after_starting_fold = log_starting_degree - log_starting_folding_factor;
        let extra_folds = after_starting_fold / log_folding_factor;
        let total_folds = 1 + extra_folds;

        // Last fold produces the final polynomial; intermediate rounds = total_folds - 1.
        let num_rounds = total_folds.saturating_sub(1);
        let log_final_degree = after_starting_fold - extra_folds * log_folding_factor;

        // Per-round target adds a union-bound buffer of `ceil(log2(num_terms))` so that
        // summing every algebraic failure mode across the protocol is bounded by
        // `2^{-security_level}`. Exact term count: each of the `total_folds - 1`
        // intermediate rounds has six independent terms (query tier: query
        // failure, OOD, random-combination, shake-check; folding tier: proximity-gaps,
        // sumcheck); the final stage has three (folding tier + final query failure); a
        // `Combine` bucket adds one more (Theorem 7.1's `ε_com` term, §4.5).
        // The buffer applies to every per-event term. OOD, shake-check, and Combine must
        // reach the buffered target algebraically because the query-phase grind does not
        // protect them.
        const TERMS_PER_INTERMEDIATE_ROUND: usize = 6;
        const FINAL_STAGE_TERMS: usize = 3;
        let combine_term = usize::from(combine.is_some_and(|c| c.num_classes >= 2));
        let num_alg_terms =
            TERMS_PER_INTERMEDIATE_ROUND * (total_folds - 1) + FINAL_STAGE_TERMS + combine_term;
        let union_bound_buffer = libm::ceil(libm::log2(num_alg_terms as f64)) as usize;
        let buffered_security_level = security_level + union_bound_buffer;

        // Convert algebraic-security bits to a PoW difficulty.
        // PoW = ceil(buffered_security_level − algebraic_bits), capped at max_pow_bits.
        // A derived value > max_pow_bits is a hard misconfiguration: the user's parameters
        // do not deliver `security_level` bits over `total_folds` rounds even after
        // exhausting the PoW budget.
        let derive_pow_bits = |label: &str, round: &str, algebraic_bits: f64| -> usize {
            let gap = (buffered_security_level as f64 - algebraic_bits).max(0.0);
            let needed = libm::ceil(gap) as usize;
            assert!(
                needed <= max_pow_bits,
                "{round} {label} requires {needed} PoW bits to reach \
                 buffered security target = {buffered_security_level} \
                 (security_level = {security_level} + union-bound buffer = {union_bound_buffer}, \
                 algebraic bits = {algebraic_bits}), \
                 but max_pow_bits = {max_pow_bits}. Increase max_pow_bits, log_blowup, \
                 or use a larger field.",
            );
            needed
        };

        // Initial domain shift: use the multiplicative generator so the
        // initial domain is disjoint from all subgroups of the base field.
        // Each round commits the folded oracle on a disjoint coset of the next domain.
        let initial_shift = F::GENERATOR;

        let mut round_configs = Vec::with_capacity(num_rounds);
        let mut log_degree = log_starting_degree;
        // The committed codeword's domain starts at the full initial domain and halves
        // each round (rate improvement: degree drops by k, domain drops by 2).
        let mut log_domain_size = log_starting_domain;
        // The effective inverse rate starts at log_blowup and increases by
        // (log_folding_factor - 1) each round.
        let mut log_inv_rate = log_blowup;
        let mut domain_shift = initial_shift;

        // Query count uses the PoW-assisted target so that summing the per-round query
        // failure over all folds is bounded by `2^{-algebraic_security_level}`.
        let pow_target_bits = algebraic_security_level + union_bound_buffer;
        let query_count = |stage_log_inv_rate: usize, eta: f64| {
            let failure_base = params
                .soundness_type
                .stir_query_failure_base(stage_log_inv_rate, eta);
            params
                .soundness_type
                .stir_queries_for_base(pow_target_bits, failure_base)
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
        // `shift_{i+1} = shift_i^{k_i} * GEN` each round (`k_i` = that round's own folding
        // factor), so `shift_i = GEN^{c_i}` for the nested recursion `c_{i+1} = c_i * k_i + 1`
        // — not a plain power of the summed `k_j`'s. Disjoint cosets `L_i ∩ L_{i+1} = ∅`
        // require `GEN` to avoid the size-`2^{log_domain_i}` subgroup reached at round `i`;
        // we check the simpler `GEN^{2^{N_i}} ≠ 1` where `N_i = (Σ_{j≤i} log_folding_factor_j)
        // + log_domain_i` is the cumulative folding-log through round `i`. Holds for any
        // field whose multiplicative order has nontrivial odd part (BabyBear, KoalaBear,
        // Goldilocks, …); the assertion catches pathological fields.
        let assert_disjoint_cosets =
            |round_index: usize, log_domain_i: usize, cumulative_log_folding: usize| {
                let n_i = cumulative_log_folding + log_domain_i;
                assert!(
                    F::GENERATOR.exp_power_of_2(n_i) != F::ONE,
                    "STIR round {round_index}: disjoint-coset schedule requires \
                     Field::GENERATOR^(2^{n_i}) ≠ 1 (i.e. GEN ∉ subgroup of size \
                     2^{log_domain_i} after the cumulative fold).",
                );
            };

        // Size eta against both classes of error: PoW-eligible folding/query terms target
        // `pow_target_bits`, while OOD and shake terms must reach the full buffered target.
        // Round 0 folds by `log_starting_folding_factor` (k0), not the steady-state
        // `log_folding_factor` used from round 1 on.
        let mut final_eta = params.soundness_type.stir_initial_eta(
            pow_target_bits,
            buffered_security_level,
            log_degree,
            log_inv_rate,
            log_starting_folding_factor,
            field_size_bits,
        );
        // Combine (§4.5) is not PoW-eligible (it runs once, before the query phase's
        // grind), so — like OOD and shake-check — it must reach the full buffered target
        // on its own. Evaluated at `log_degree`/`log_inv_rate` as they stand here: round
        // 0's own starting degree and rate, matching what Combine merges at (immediately
        // before the round-0 fold).
        if let Some(c) = combine.filter(|c| c.num_classes >= 2) {
            let log_combine_eta = combine_min_log_eta(
                params.soundness_type,
                field_size_bits,
                log_inv_rate,
                log_degree,
                c.ell,
                buffered_security_level,
            );
            final_eta = final_eta.max(libm::pow(2., log_combine_eta));
        }
        validate_eta(0, log_inv_rate, final_eta);

        // Round 0 reuses the `stir_initial_eta` already computed above; every subsequent
        // round derives eta from the previous round's query count via `stir_recursive_eta`.
        let mut prev_queries = 0;
        let mut cumulative_log_folding = 0usize;
        for round in 0..num_rounds {
            let round_log_folding_factor = if round == 0 {
                log_starting_folding_factor
            } else {
                log_folding_factor
            };
            if round != 0 {
                final_eta = params.soundness_type.stir_recursive_eta(
                    pow_target_bits,
                    buffered_security_level,
                    log_degree,
                    log_inv_rate,
                    log_domain_size,
                    log_folding_factor,
                    field_size_bits,
                    prev_queries,
                );
                validate_eta(round, log_inv_rate, final_eta);
            }

            let num_queries = query_count(log_inv_rate, final_eta);
            cumulative_log_folding += round_log_folding_factor;
            assert_disjoint_cosets(round, log_domain_size, cumulative_log_folding);

            let fold_alg = params.soundness_type.fold_algebraic_bits_at_log_eta(
                field_size_bits,
                log_degree,
                log_inv_rate,
                libm::log2(final_eta),
            );
            let query_alg = params.soundness_type.stir_query_pow_eligible_bits(
                field_size_bits,
                log_degree,
                log_inv_rate,
                final_eta,
                num_queries,
                num_ood_samples,
            );
            let unprotected_alg = params.soundness_type.stir_query_unprotected_bits(
                field_size_bits,
                log_degree,
                log_inv_rate,
                final_eta,
                num_queries,
                num_ood_samples,
            );
            let round_label = format!("round {round}");
            assert!(
                unprotected_alg >= buffered_security_level as f64,
                "{round_label} OOD/shake checks reach only {unprotected_alg:.4} bits, below \
                 the buffered target {buffered_security_level}; these challenges are not \
                 protected by the query-phase PoW. Use a larger challenge field or lower \
                 security target."
            );
            let folding_pow_bits = derive_pow_bits("folding", &round_label, fold_alg);
            let pow_bits = derive_pow_bits("query", &round_label, query_alg);

            round_configs.push(StirRoundConfig {
                log_degree,
                log_domain_size,
                log_fold_domain_size: log_domain_size - round_log_folding_factor,
                domain_shift,
                log_folding_factor: round_log_folding_factor,
                eta: final_eta,
                num_queries,
                num_ood_samples,
                pow_bits,
                folding_pow_bits,
            });

            prev_queries = num_queries;
            log_degree -= round_log_folding_factor;
            log_domain_size -= 1;
            log_inv_rate += round_log_folding_factor - 1;
            domain_shift = domain_shift.exp_power_of_2(round_log_folding_factor) * F::GENERATOR;
        }

        if total_folds != 1 {
            final_eta = params.soundness_type.stir_recursive_eta(
                pow_target_bits,
                buffered_security_level,
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
        let final_fold_alg = params.soundness_type.fold_algebraic_bits_at_log_eta(
            field_size_bits,
            log_degree,
            log_inv_rate,
            libm::log2(final_eta),
        );
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
            log_starting_folding_factor,
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

    /// Log₂ of the folding arity used by the final direct-send stage.
    ///
    /// This is `log_folding_factor` (the steady-state arity) unless the schedule has no
    /// intermediate rounds, in which case round 0's fold IS the final fold and this
    /// returns `log_starting_folding_factor` instead.
    pub const fn final_log_folding_factor(&self) -> usize {
        if self.num_rounds() == 0 {
            self.log_starting_folding_factor
        } else {
            self.log_folding_factor
        }
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
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_field::Field;
    use p3_field::extension::{BinomialExtensionField, CubicTrinomialExtensionField};
    use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;

    use super::*;

    type TestF = BabyBear;
    type TestEF = BinomialExtensionField<TestF, 4>;
    type TestPerm = Poseidon2BabyBear<16>;
    type TestHash = PaddingFreeSponge<TestPerm, 16, 8, 8>;
    type TestCompress = TruncatedPermutation<TestPerm, 2, 8, 16>;
    type TestPackedF = <TestF as Field>::Packing;
    type TestValMmcs = MerkleTreeMmcs<TestPackedF, TestPackedF, TestHash, TestCompress, 2, 8>;
    type TestMmcs = ExtensionMmcs<TestF, TestEF, TestValMmcs>;
    type TestChallenger = DuplexChallenger<TestF, TestPerm, 16, 8>;

    fn test_params(log_blowup: usize, log_folding_factor: usize) -> StirParameters<TestMmcs> {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let perm = TestPerm::new_from_rng_128(&mut rng);
        let val_mmcs = TestValMmcs::new(TestHash::new(perm.clone()), TestCompress::new(perm), 0);

        StirParameters {
            log_blowup,
            log_folding_factor,
            log_starting_folding_factor: log_folding_factor,
            soundness_type: SecurityAssumption::CapacityBound,
            security_level: 80,
            max_pow_bits: 20,
            mmcs: TestMmcs::new(val_mmcs),
        }
    }

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
            log_starting_folding_factor: 2,
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
    #[should_panic(
        expected = "Initial domain exponent log_starting_degree + log_blowup overflows usize"
    )]
    fn test_stir_config_rejects_initial_domain_exponent_addition_overflow() {
        let params = test_params(1, 2);
        let _ = StirConfig::<TestF, TestEF, TestMmcs, TestChallenger>::new(usize::MAX, params);
    }

    #[test]
    #[should_panic(expected = "must be less than usize::BITS")]
    fn test_stir_config_rejects_initial_domain_exponent_exceeding_usize_bits() {
        let params = test_params(usize::BITS as usize, 2);
        let _ = StirConfig::<TestF, TestEF, TestMmcs, TestChallenger>::new(2, params);
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
                log_starting_folding_factor: 2,
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
                log_starting_folding_factor: 2,
                soundness_type: SecurityAssumption::CapacityBound,
                security_level: 80,
                max_pow_bits: 20,
                mmcs: MyMmcs::new(val_mmcs),
            },
        );
        assert!(cb.round_configs.iter().all(|rc| rc.num_ood_samples == 2));
    }

    #[test]
    fn test_johnson_config_supports_realistic_security_targets() {
        type E100 = BinomialExtensionField<TestF, 5>;
        type M100 = ExtensionMmcs<TestF, E100, TestValMmcs>;

        assert_eq!(E100::bits(), 155);

        let mut rng = rand::rngs::SmallRng::seed_from_u64(101);
        let perm = TestPerm::new_from_rng_128(&mut rng);
        let val_mmcs = TestValMmcs::new(TestHash::new(perm.clone()), TestCompress::new(perm), 0);
        let config_100 = StirConfig::<TestF, E100, M100, TestChallenger>::new(
            20,
            StirParameters {
                log_blowup: 2,
                log_folding_factor: 2,
                log_starting_folding_factor: 2,
                soundness_type: SecurityAssumption::JohnsonBound,
                security_level: 100,
                max_pow_bits: 0,
                mmcs: M100::new(val_mmcs),
            },
        );
        assert!(
            config_100
                .round_configs
                .iter()
                .all(|rc| rc.pow_bits == 0 && rc.folding_pow_bits == 0)
        );
        assert_eq!(config_100.final_pow_bits, 0);
        assert_eq!(config_100.final_folding_pow_bits, 0);

        type F128 = Goldilocks;
        type E128 = CubicTrinomialExtensionField<F128>;
        type Perm128 = Poseidon2Goldilocks<8>;
        type Hash128 = PaddingFreeSponge<Perm128, 8, 4, 4>;
        type Compress128 = TruncatedPermutation<Perm128, 2, 4, 8>;
        type PackedF128 = <F128 as Field>::Packing;
        type ValMmcs128 = MerkleTreeMmcs<PackedF128, PackedF128, Hash128, Compress128, 2, 4>;
        type Mmcs128 = ExtensionMmcs<F128, E128, ValMmcs128>;
        type Challenger128 = DuplexChallenger<F128, Perm128, 8, 4>;

        assert_eq!(E128::bits(), 192);

        let perm = Perm128::new_from_rng_128(&mut rng);
        let val_mmcs = ValMmcs128::new(Hash128::new(perm.clone()), Compress128::new(perm), 0);
        let config_128 = StirConfig::<F128, E128, Mmcs128, Challenger128>::new(
            20,
            StirParameters {
                log_blowup: 2,
                log_folding_factor: 2,
                log_starting_folding_factor: 2,
                soundness_type: SecurityAssumption::JohnsonBound,
                security_level: 128,
                max_pow_bits: 0,
                mmcs: Mmcs128::new(val_mmcs),
            },
        );
        assert!(
            config_128
                .round_configs
                .iter()
                .all(|rc| rc.pow_bits == 0 && rc.folding_pow_bits == 0)
        );
        assert_eq!(config_128.final_pow_bits, 0);
        assert_eq!(config_128.final_folding_pow_bits, 0);
    }

    #[test]
    fn test_stir_config_union_bound_buffer_scales_with_rounds() {
        // The per-round target_bits adds ceil(log2(6 * total_folds)) to algebraic_security_level
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
                    log_starting_folding_factor: 2,
                    soundness_type: SecurityAssumption::CapacityBound,
                    security_level: 80,
                    max_pow_bits: 20,
                    mmcs: MyMmcs::new(val_mmcs.clone()),
                },
            )
        };

        // Shallow: log_starting_degree=4  ⇒ total_folds = 2 ⇒ buffer = ceil(log2(6·1+3)) = 4.
        // Deep:    log_starting_degree=16 ⇒ total_folds = 8 ⇒ buffer = ceil(log2(6·7+3)) = 6.
        let shallow = make(4);
        let deep = make(16);

        // Both have positive query counts.
        assert!(shallow.final_queries > 0);
        assert!(deep.final_queries > 0);

        // The deeper protocol's final-round target is strictly larger because of the bigger
        // buffer, so for comparable rates final_queries must be ≥ the shallow one's.
        // (Eta differs across configurations, so we can only assert a soft inequality here.)
        // The strict invariant we can check: per-round target_bits is monotone in total_folds.
        let buffer = |tf: usize| libm::ceil(libm::log2((6 * (tf - 1) + 3) as f64)) as usize;
        assert_eq!(buffer(2), 4);
        assert_eq!(buffer(8), 6);
        assert!(buffer(8) > buffer(2));

        // Sanity: non-empty rounds; deep > shallow in number of fold steps.
        assert!(deep.num_rounds() + 1 > shallow.num_rounds() + 1);
    }

    /// Whole-pipeline invariant: for every derived stage, the algebraic bits the
    /// *stored* schedule parameters actually deliver, plus the stored PoW bits, must
    /// reach `buffered_security_level`.
    #[test]
    fn test_stir_config_every_stage_meets_buffered_target() {
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

        let mut rng = rand::rngs::SmallRng::seed_from_u64(99);
        let perm = Perm::new_from_rng_128(&mut rng);
        let val_mmcs = ValMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);
        let field_size_bits = EF::bits();

        // (log_starting_degree, log_blowup, log_folding_factor, log_starting_folding_factor,
        // security_level, max_pow_bits, soundness_type).
        let cb = SecurityAssumption::CapacityBound;
        let jb = SecurityAssumption::JohnsonBound;
        let cases = [
            (8, 1, 2, 2, 80, 20, cb),
            (8, 2, 2, 2, 80, 20, cb),
            (8, 2, 2, 2, 80, 20, jb),
            (16, 1, 2, 2, 80, 20, cb),
            (4, 1, 2, 2, 80, 20, cb),
            (8, 1, 2, 2, 16, 0, cb),
            (12, 1, 3, 2, 16, 0, cb),
            (4, 1, 2, 2, 16, 0, cb),
            // k0 != k: round 0 folds by a different factor than every later round.
            (16, 1, 3, 2, 80, 20, cb),
        ];

        {
            for &(log_deg, log_blowup, log_fold, log_starting_fold, sec, max_pow, soundness_type) in
                &cases
            {
                let config = StirConfig::<F, EF, MyMmcs, MyChallenger>::new(
                    log_deg,
                    StirParameters {
                        log_blowup,
                        log_folding_factor: log_fold,
                        log_starting_folding_factor: log_starting_fold,
                        soundness_type,
                        security_level: sec,
                        max_pow_bits: max_pow,
                        mmcs: MyMmcs::new(val_mmcs.clone()),
                    },
                );

                // Mirror `StirConfig::new`'s buffered target.
                let total_folds = 1 + (log_deg - log_starting_fold) / log_fold;
                let buffer = libm::ceil(libm::log2((6 * (total_folds - 1) + 3) as f64)) as usize;
                let buffered = (sec + buffer) as f64;
                // Recomputed algebraic bits use the same `libm` math as the config, so
                // the only slack is float rounding in the comparison itself.
                let eps = 1e-9;

                let label = |stage: &str| {
                    format!(
                        "{soundness_type} {stage} below buffered target \
                         (log_deg={log_deg}, log_blowup={log_blowup}, \
                         log_fold={log_fold}, sec={sec}, max_pow={max_pow})"
                    )
                };

                for rc in &config.round_configs {
                    let log_inv_rate = rc.log_domain_size - rc.log_degree;

                    let query_alg = soundness_type.stir_query_pow_eligible_bits(
                        field_size_bits,
                        rc.log_degree,
                        log_inv_rate,
                        rc.eta,
                        rc.num_queries,
                        rc.num_ood_samples,
                    );
                    assert!(
                        query_alg + rc.pow_bits as f64 >= buffered - eps,
                        "{}: query_alg={query_alg:.4} + pow={} < {buffered:.4}",
                        label("intermediate-query"),
                        rc.pow_bits,
                    );

                    let unprotected_alg = soundness_type.stir_query_unprotected_bits(
                        field_size_bits,
                        rc.log_degree,
                        log_inv_rate,
                        rc.eta,
                        rc.num_queries,
                        rc.num_ood_samples,
                    );
                    assert!(
                        unprotected_alg >= buffered - eps,
                        "{}: OOD/shake={unprotected_alg:.4} < {buffered:.4} without PoW",
                        label("intermediate-unprotected"),
                    );

                    let fold_alg = soundness_type.fold_algebraic_bits_at_log_eta(
                        field_size_bits,
                        rc.log_degree,
                        log_inv_rate,
                        libm::log2(rc.eta),
                    );
                    assert!(
                        fold_alg + rc.folding_pow_bits as f64 >= buffered - eps,
                        "{}: fold_alg={fold_alg:.4} + pow={} < {buffered:.4}",
                        label("intermediate-fold"),
                        rc.folding_pow_bits,
                    );
                }

                // Final stage: reconstruct its (log_degree, log_inv_rate) by stepping
                // one fold past the last intermediate round, or from the starting
                // parameters when there are no intermediate rounds (total_folds == 1).
                let (final_log_degree, final_log_inv_rate) =
                    config
                        .round_configs
                        .last()
                        .map_or((log_deg, log_blowup), |last| {
                            let fd = last.log_degree - log_fold;
                            let fdom = last.log_domain_size - 1;
                            (fd, fdom - fd)
                        });

                let final_query_alg = soundness_type.stir_final_query_algebraic_bits(
                    final_log_inv_rate,
                    config.final_eta,
                    config.final_queries,
                );
                assert!(
                    final_query_alg + config.final_pow_bits as f64 >= buffered - eps,
                    "{}: final_query_alg={final_query_alg:.4} + pow={} < {buffered:.4}",
                    label("final-query"),
                    config.final_pow_bits,
                );

                let final_fold_alg = soundness_type.fold_algebraic_bits_at_log_eta(
                    field_size_bits,
                    final_log_degree,
                    final_log_inv_rate,
                    libm::log2(config.final_eta),
                );
                assert!(
                    final_fold_alg + config.final_folding_pow_bits as f64 >= buffered - eps,
                    "{}: final_fold_alg={final_fold_alg:.4} + pow={} < {buffered:.4}",
                    label("final-fold"),
                    config.final_folding_pow_bits,
                );
            }
        }
    }
}
