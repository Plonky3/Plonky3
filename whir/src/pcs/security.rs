//! Security evidence for the prescribed-point PCS adapter.

use alloc::vec::Vec;

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use p3_security::{ErrorBits, SecurityAssumption};
use p3_sumcheck::{OpeningProtocol, PrescribedOpeningSecurity};
use p3_util::log2_ceil_usize;

use crate::parameters::WhirConfig;

/// Union-composes the algebraic phases under the configured proximity assumption.
/// Hash security is supplied by the outer protocol, which owns that configuration.
pub(super) fn prescribed_security<EF, F, Challenger>(
    config: &WhirConfig<EF, F, Challenger>,
    protocol: &OpeningProtocol,
) -> Option<PrescribedOpeningSecurity>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // Public derived fields can be changed after construction. Only certify a
    // schedule which agrees with the validated parameter derivation.
    let canonical =
        WhirConfig::<EF, F, Challenger>::new(config.num_variables, config.params.clone()).ok()?;
    if config.commitment_ood_samples != canonical.commitment_ood_samples
        || config.folding_schedule != canonical.folding_schedule
        || config.starting_folding_pow_bits != canonical.starting_folding_pow_bits
        || config.final_queries != canonical.final_queries
        || config.final_pow_bits != canonical.final_pow_bits
        || config.final_sumcheck_rounds != canonical.final_sumcheck_rounds
        || config.final_folding_pow_bits != canonical.final_folding_pow_bits
        || config.round_parameters.len() != canonical.round_parameters.len()
        || config
            .round_parameters
            .iter()
            .zip(&canonical.round_parameters)
            .any(|(a, b)| {
                a.pow_bits != b.pow_bits
                    || a.folding_pow_bits != b.folding_pow_bits
                    || a.num_queries != b.num_queries
                    || a.ood_samples != b.ood_samples
                    || a.num_variables != b.num_variables
                    || a.folding_factor != b.folding_factor
                    || a.log_inv_rate != b.log_inv_rate
                    || a.domain_size != b.domain_size
                    || a.folded_domain_gen != b.folded_domain_gen
            })
    {
        return None;
    }

    let total_cells = protocol
        .table_shapes()
        .iter()
        .try_fold(0usize, |total, table| {
            let cells = (1usize.checked_shl(table.num_variables().try_into().ok()?)?)
                .checked_mul(table.width())?;
            total.checked_add(cells)
        })?;
    if total_cells == 0 || log2_ceil_usize(total_cells) != config.num_variables {
        return None;
    }
    let num_claims = protocol
        .iter_openings()
        .try_fold(config.commitment_ood_samples, |total, (_, batch)| {
            total.checked_add(batch.len())
        })?;
    config.validate_initial_claims(num_claims).ok()?;

    // Field::bits() rounds upward. A whole-bit lower bound avoids granting
    // nearly one nonexistent bit to small-prime extension fields.
    let field_bits = EF::bits().checked_sub(1)?;
    let assumption = config.soundness_type;
    let mut errors = Vec::new();
    errors.push(ErrorBits::from_log2(
        config.initial_claims_error(num_claims),
    ));

    let add_ood = |errors: &mut Vec<ErrorBits>, variables, rate, samples| {
        if assumption != SecurityAssumption::UniqueDecoding {
            errors.push(ErrorBits::from_log2(
                assumption.ood_error(variables, rate, field_bits, samples),
            ));
        }
    };
    let add_folds = |errors: &mut Vec<ErrorBits>, variables, rate, folds, pow| {
        // Use the largest degree/rate bound for every binary fold in a phase.
        // The Johnson helper retains the dominant term; one additional bit
        // covers its positive lower-order terms at the fixed m = 10.
        let gap = assumption.prox_gaps_error(variables, rate, field_bits, 2)
            - if assumption == SecurityAssumption::JohnsonBound {
                1.0
            } else {
                0.0
            };
        let sumcheck = assumption.fold_sumcheck_error(field_bits, variables, rate);
        for _ in 0..folds {
            errors.push(ErrorBits::from_log2(gap + pow as f64));
            errors.push(ErrorBits::from_log2(sumcheck + pow as f64));
        }
    };
    add_ood(
        &mut errors,
        config.num_variables,
        config.starting_log_inv_rate,
        config.commitment_ood_samples,
    );
    add_folds(
        &mut errors,
        config.num_variables,
        config.starting_log_inv_rate,
        config.folding_schedule[0],
        config.starting_folding_pow_bits,
    );
    let mut old_rate = config.starting_log_inv_rate;
    for (index, round) in config.round_parameters.iter().enumerate() {
        add_ood(
            &mut errors,
            round.num_variables,
            round.log_inv_rate,
            round.ood_samples,
        );
        errors.push(ErrorBits::from_log2(
            assumption.queries_error(old_rate, round.num_queries) + round.pow_bits as f64,
        ));
        errors.push(ErrorBits::from_log2(
            assumption.queries_combination_error(
                field_bits,
                round.num_variables,
                round.log_inv_rate,
                round.ood_samples,
                round.num_queries,
            ) + round.pow_bits as f64,
        ));
        add_folds(
            &mut errors,
            round.num_variables,
            round.log_inv_rate,
            config.folding_schedule[index + 1],
            round.folding_pow_bits,
        );
        old_rate = round.log_inv_rate;
    }
    errors.push(ErrorBits::from_log2(
        assumption.queries_error(old_rate, config.final_queries) + config.final_pow_bits as f64,
    ));
    for _ in 0..config.final_sumcheck_rounds {
        errors.push(ErrorBits::from_log2(
            field_bits as f64 - 1.0 + config.final_folding_pow_bits as f64,
        ));
    }
    let bits = ErrorBits::sum(&errors).bits();
    bits.is_finite().then_some(PrescribedOpeningSecurity {
        error: ErrorBits::from_log2(bits.max(0.0)),
        // Initial OOD samples are drawn only in open_at, after outer AIR/GKR
        // challenges. They cannot shrink the candidate set for those reductions.
        log2_max_candidates: assumption
            .list_size_bits(config.num_variables, config.starting_log_inv_rate),
    })
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::Field;
    use p3_field::extension::BinomialExtensionField;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_sumcheck::layout::PrefixProver;
    use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
    use crate::pcs::prover::WhirProver;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
    type Packed = <F as Field>::Packing;
    type Mmcs = MerkleTreeMmcs<Packed, Packed, Hash, Compress, 2, 8>;
    type Pcs = WhirProver<EF, F, Radix2DFTSmallBatch<F>, Mmcs, Challenger, PrefixProver<F, EF>>;

    fn pcs_with_assumption(soundness_type: SecurityAssumption) -> Pcs {
        let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(73));
        let mmcs = Mmcs::new(Hash::new(perm.clone()), Compress::new(perm), 0);
        let config = WhirConfig::new(
            12,
            ProtocolParameters {
                starting_log_inv_rate: 1,
                round_log_inv_rates: vec![],
                folding_factor: FoldingFactor::Constant(4),
                soundness_type,
                security_level: 32,
                pow_bits: 0,
            },
        )
        .unwrap();
        Pcs::new(config, Radix2DFTSmallBatch::default(), mmcs)
    }

    fn pcs() -> Pcs {
        pcs_with_assumption(SecurityAssumption::UniqueDecoding)
    }

    fn protocol(num_variables: usize) -> OpeningProtocol {
        OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(num_variables, 1),
            vec![OpeningBatch::new(vec![0], vec![])],
        )])
    }

    #[test]
    fn composed_security_charges_every_phase() {
        let bits = pcs()
            .prescribed_security(&protocol(12))
            .expect("WHIR supplies shape-checked algebraic security")
            .error
            .bits();
        // Each proximity phase targets 32 bits. Union-composing the phases must
        // lose bits, while remaining useful for a lower security target.
        assert!(bits > 20.0 && bits < 32.0, "composed security: {bits}");
    }

    #[test]
    fn candidate_list_is_fixed_before_opening_time_ood_checks() {
        let ud = pcs().prescribed_security(&protocol(12)).unwrap();
        assert_eq!(ud.log2_max_candidates, 0.0);
        for assumption in [
            SecurityAssumption::JohnsonBound,
            SecurityAssumption::CapacityBound,
        ] {
            let pcs = pcs_with_assumption(assumption);
            let evidence = pcs.prescribed_security(&protocol(12)).unwrap();
            assert!(pcs.config.commitment_ood_samples > 0);
            assert!(evidence.log2_max_candidates > 0.0);
            assert_eq!(
                evidence.log2_max_candidates,
                assumption.list_size_bits(12, 1)
            );
        }
    }

    #[test]
    fn mismatched_stacked_shape_has_no_security_evidence() {
        assert!(pcs().prescribed_security(&protocol(13)).is_none());
    }

    #[test]
    fn modified_derived_parameters_have_no_security_evidence() {
        let mut pcs = pcs();
        pcs.config.final_queries = 0;
        assert!(pcs.prescribed_security(&protocol(12)).is_none());
    }
}
