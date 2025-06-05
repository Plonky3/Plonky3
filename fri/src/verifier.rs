use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::reverse_bits_len;
use p3_util::zip_eq::zip_eq;

use crate::{CommitPhaseProofStep, FriFoldingStrategy, FriParameters, FriProof};

#[derive(Debug)]
pub enum FriError<CommitMmcsErr, InputError> {
    InvalidProofShape,
    CommitPhaseMmcsError(CommitMmcsErr),
    InputError(InputError),
    FinalPolyMismatch,
    InvalidPowWitness,
    MissingInput,
}

pub fn verify<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    proof: &FriProof<Challenge, M, Challenger::Witness, Folding::InputProof>,
    challenger: &mut Challenger,
    open_input: impl Fn(
        usize,
        &Folding::InputProof,
    ) -> Result<Vec<(usize, Challenge)>, FriError<M::Error, Folding::InputError>>,
) -> Result<(), FriError<M::Error, Folding::InputError>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample_algebra_element()
        })
        .collect();

    if proof.final_poly.len() != params.final_poly_len() {
        return Err(FriError::InvalidProofShape);
    }

    // Observe all coefficients of the final polynomial.
    proof
        .final_poly
        .iter()
        .for_each(|x| challenger.observe_algebra_element(*x));

    if proof.query_proofs.len() != params.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    // Check PoW.
    if !challenger.check_witness(params.proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    // The log of the maximum domain size.
    let log_max_height =
        proof.commit_phase_commits.len() + params.log_blowup + params.log_final_poly_len;

    // The log of the final domain size.
    let log_final_height = params.log_blowup + params.log_final_poly_len;

    for qp in &proof.query_proofs {
        let index = challenger.sample_bits(log_max_height + folding.extra_query_index_bits());
        let ro = open_input(index, &qp.input_proof)?;

        debug_assert!(
            ro.iter().tuple_windows().all(|((l, _), (r, _))| l > r),
            "reduced openings sorted by height descending"
        );

        let mut domain_index = index >> folding.extra_query_index_bits();

        // Starting at the evaluation at `index` of the initial domain,
        // perform fri folds until the domain size reaches the final domain size.
        // Check after each fold that the pair of sibling evaluations at the current
        // node match the commitment.
        let folded_eval = verify_query(
            folding,
            params,
            &mut domain_index,
            zip_eq(
                zip_eq(
                    &betas,
                    &proof.commit_phase_commits,
                    FriError::InvalidProofShape,
                )?,
                &qp.commit_phase_openings,
                FriError::InvalidProofShape,
            )?,
            ro,
            log_max_height,
            log_final_height,
        )?;

        // We open the final polynomial at index `domain_index`, which corresponds to evaluating
        // the polynomial at x^k, where x is the 2-adic generator of order `max_height` and k is
        // `reverse_bits_len(domain_index, log_max_height)`.
        let x = Val::two_adic_generator(log_max_height)
            .exp_u64(reverse_bits_len(domain_index, log_max_height) as u64);

        // Evaluate the final polynomial at x.
        let mut eval = Challenge::ZERO;
        for &coeff in proof.final_poly.iter().rev() {
            eval = eval * x + coeff;
        }

        if eval != folded_eval {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(())
}

type CommitStep<'a, F, M> = (
    (
        &'a F, // The challenge point beta used for the next fold of FRI evaluations.
        &'a <M as Mmcs<F>>::Commitment, // A commitment to the FRI evaluations on the current domain.
    ),
    &'a CommitPhaseProofStep<F, M>, // The sibling and opening proof for the current FRI node.
);

/// Verifies a single query chain in the FRI proof.
///
/// Given an initial `index` corresponding to a point in the initial domain
/// and a series of `reduced_openings` corresponding to evaluations of
/// polynomials to be added in at specific domain sizes, perform the standard
/// sequence of FRI folds, checking at each step that the pair of sibling evaluations
/// match the commitment.
fn verify_query<'a, Folding, F, EF, M>(
    folding: &Folding,
    params: &FriParameters<M>,
    index: &mut usize,
    steps: impl ExactSizeIterator<Item = CommitStep<'a, EF, M>>,
    reduced_openings: Vec<(usize, EF)>,
    log_max_height: usize,
    log_final_height: usize,
) -> Result<EF, FriError<M::Error, Folding::InputError>>
where
    F: Field,
    EF: ExtensionField<F>,
    M: Mmcs<EF> + 'a,
    Folding: FriFoldingStrategy<F, EF>,
{
    let mut ro_iter = reduced_openings.into_iter().peekable();
    let mut folded_eval = ro_iter
        .next_if(|(lh, _)| *lh == log_max_height)
        .map(|(_, ro)| ro)
        .ok_or(FriError::MissingInput)?;

    // We start with evaluations over a domain of size (1 << log_max_height). We fold
    // using FRI until the domain size reaches (1 << log_final_height).
    for (log_folded_height, ((&beta, comm), opening)) in zip_eq(
        (log_final_height..log_max_height).rev(),
        steps,
        FriError::InvalidProofShape,
    )? {
        // Get the index of the other sibling of the current fri node.
        let index_sibling = *index ^ 1;

        let mut evals = vec![folded_eval; 2];
        evals[index_sibling % 2] = opening.sibling_value;

        let dims = &[Dimensions {
            width: 2,
            height: 1 << log_folded_height,
        }];

        // Replace index with the index of the parent fri node.
        *index >>= 1;

        // Verify the commitment to the evaluations of the sibling nodes.
        params
            .mmcs
            .verify_batch(
                comm,
                dims,
                *index,
                BatchOpeningRef::new(&[evals.clone()], &opening.opening_proof), // It's possible to remove the clone here but unnecessary as evals is tiny.
            )
            .map_err(FriError::CommitPhaseMmcsError)?;

        // Fold the pair of evaluations of sibling nodes into the evaluation of the parent fri node.
        folded_eval = folding.fold_row(*index, log_folded_height, beta, evals.into_iter());

        // If there are new polynomials to roll in at the folded height, do so.
        //
        // Each element of `ro_iter` is the evaluation of a reduced opening polynomial, which is itself
        // a random linear combination `f_{i, 0}(x) + alpha f_{i, 1}(x) + ...`, but when we add it
        // to the current folded polynomial evaluation claim, we need to multiply by a new random factor
        // since `f_{i, 0}` has no leading coefficient.
        //
        // We use `beta^2` as the random factor since `beta` is already used in the folding.
        // This increases the query phase error probability by a negligible amount, and does not change
        // the required number of FRI queries.
        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height) {
            folded_eval += beta.square() * ro;
        }
    }

    // If ro_iter is not empty, we failed to fold in some polynomial evaluations.
    if ro_iter.next().is_some() {
        return Err(FriError::InvalidProofShape);
    }

    // If we reached this point, we have verified that, starting at the initial index,
    // the chain of folds has produced folded_eval.
    Ok(folded_eval)
}
