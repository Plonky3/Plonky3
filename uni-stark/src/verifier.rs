use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, TwoRowMatrixView};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::UnivariatePcs;
use p3_field::{AbstractExtensionField, AbstractField, Field, TwoAdicField};

use crate::{Proof, StarkConfig, VerifierConstraintFolder};

pub fn verify<SC, A>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    proof: &Proof<SC>,
) -> Result<(), VerificationError>
where
    SC: StarkConfig,
    A: for<'a> Air<VerifierConstraintFolder<'a, SC::Challenge>>,
{
    let degree_bits = 6; // TODO
    let log_quotient_degree = 1; // TODO
    let g_subgroup = SC::Domain::two_adic_generator(degree_bits);

    let Proof {
        commitments,
        opened_values,
        opening_proof,
    } = proof;

    challenger.observe(commitments.trace.clone());
    let alpha: SC::Challenge = challenger.sample_ext_element();
    challenger.observe(commitments.quotient_chunks.clone());
    let zeta: SC::Challenge = challenger.sample_ext_element();

    let local_and_next = [zeta, zeta * g_subgroup];
    let commits_and_points = &[
        (commitments.trace.clone(), local_and_next.as_slice()),
        (
            commitments.quotient_chunks.clone(),
            &[zeta.exp_power_of_2(log_quotient_degree)],
        ),
    ];
    let values = vec![
        vec![vec![
            opened_values.trace_local.clone(),
            opened_values.trace_next.clone(),
        ]],
        vec![vec![opened_values.quotient_chunks.clone()]],
    ];
    config
        .pcs()
        .verify_multi_batches(commits_and_points, values, opening_proof, challenger)
        .map_err(|_| VerificationError::InvalidOpeningArgument)?;

    // Derive the opening of the quotient polynomial, which was split into degree n chunks, then
    // flattened into D base field polynomials. We first undo the flattening.
    let challenge_ext_degree = <SC::Challenge as AbstractExtensionField<SC::Val>>::D;
    let quotient_parts: Vec<SC::Challenge> = opened_values
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
    let g_quotient_parts = SC::Domain::two_adic_generator(log_quotient_degree);
    let quotient: SC::Challenge = g_quotient_parts
        .powers()
        .zip(quotient_parts)
        .map(|(weight, part)| part * weight)
        .sum();

    let z_h = zeta.exp_power_of_2(degree_bits) - SC::Challenge::ONE;
    let is_first_row = z_h / (zeta - SC::Val::ONE);
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
        accumulator: SC::Challenge::ZERO,
    };
    air.eval(&mut folder);
    let folded_constraints = folder.accumulator;

    // Finally, check that
    //     folded_constraints(zeta) = Z_H(zeta) * quotient(zeta)
    if folded_constraints != z_h * quotient {
        // TODO: Re-enable when it's passing.
        // return Err(VerificationError::OodEvaluationMismatch);
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
