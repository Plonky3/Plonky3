use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, TwoRowMatrixView};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::UnivariatePcs;
use p3_field::{AbstractExtensionField, AbstractField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::reverse_slice_index_bits;

use crate::symbolic_builder::{get_log_quotient_degree, SymbolicAirBuilder};
use crate::{Proof, StarkConfig, VerifierConstraintFolder};

pub fn verify<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    proof: &Proof<SC>,
) -> Result<(), VerificationError>
where
    SC: StarkConfig,
    A: Air<SymbolicAirBuilder<SC::Val>> + for<'a> Air<VerifierConstraintFolder<'a, SC::Challenge>>,
{
    let log_quotient_degree = get_log_quotient_degree::<SC::Val, A>(air);

    let Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits,
    } = proof;

    let g_subgroup = SC::Val::two_adic_generator(*degree_bits);

    challenger.observe(commitments.trace.clone());
    let alpha: SC::Challenge = challenger.sample_ext_element();
    challenger.observe(commitments.quotient_chunks.clone());
    let zeta: SC::Challenge = challenger.sample_ext_element();

    let local_and_next = [vec![zeta, zeta * g_subgroup]];
    let commits_and_points = &[
        (commitments.trace.clone(), local_and_next.as_slice()),
        (
            commitments.quotient_chunks.clone(),
            &[vec![zeta.exp_power_of_2(log_quotient_degree)]],
        ),
    ];
    let values = vec![
        vec![vec![
            opened_values.trace_local.clone(),
            opened_values.trace_next.clone(),
        ]],
        vec![vec![opened_values.quotient_chunks.clone()]],
    ];
    // TODO
    let dims = &[
        vec![Dimensions {
            width: opened_values.trace_local.len(),
            height: 1 << degree_bits,
        }],
        vec![Dimensions {
            width: opened_values.quotient_chunks.len(),
            height: 1 << degree_bits,
        }],
    ];
    config
        .pcs()
        .verify_multi_batches(commits_and_points, dims, values, opening_proof, challenger)
        .map_err(|_| VerificationError::InvalidOpeningArgument)?;

    // Derive the opening of the quotient polynomial, which was split into degree n chunks, then
    // flattened into D base field polynomials. We first undo the flattening.
    let challenge_ext_degree = <SC::Challenge as AbstractExtensionField<SC::Val>>::D;
    let mut quotient_parts: Vec<SC::Challenge> = opened_values
        .quotient_chunks
        .chunks(challenge_ext_degree)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .map(|(i, &c)| <SC::Challenge as AbstractExtensionField<SC::Val>>::monomial(i) * c)
                .sum()
        })
        .collect();
    // Then we reconstruct the larger quotient polynomial from its degree-n parts.
    reverse_slice_index_bits(&mut quotient_parts);
    let quotient: SC::Challenge = zeta
        .powers()
        .zip(quotient_parts)
        .map(|(weight, part)| part * weight)
        .sum();

    let z_h = zeta.exp_power_of_2(*degree_bits) - SC::Challenge::one();
    let is_first_row = z_h / (zeta - SC::Val::one());
    let is_last_row = z_h / (zeta - g_subgroup.inverse());
    let is_transition = zeta - g_subgroup.inverse();
    let mut folder = VerifierConstraintFolder {
        main: TwoRowMatrixView {
            local: &opened_values.trace_local,
            next: &opened_values.trace_next,
        },
        is_first_row,
        is_last_row,
        is_transition,
        alpha,
        accumulator: SC::Challenge::zero(),
    };
    air.eval(&mut folder);
    let folded_constraints = folder.accumulator;

    // Finally, check that
    //     folded_constraints(zeta) = Z_H(zeta) * quotient(zeta)
    if folded_constraints != z_h * quotient {
        return Err(VerificationError::OodEvaluationMismatch);
    }

    Ok(())
}

#[derive(Debug)]
pub enum VerificationError {
    /// An error occurred while verifying the claimed openings.
    InvalidOpeningArgument,
    /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
    /// `quotient(zeta) Z_H(zeta)`.
    OodEvaluationMismatch,
}
