use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::{ExtensionField, Field};
use p3_fri::verifier::FriError;
use p3_fri::{FriFoldingStrategy, FriParameters};
use p3_matrix::Dimensions;
use p3_util::zip_eq::zip_eq;

use crate::{CircleCommitPhaseProofStep, CircleFriProof};

pub fn verify<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    proof: &CircleFriProof<Challenge, M, Challenger::Witness, Folding::InputProof>,
    challenger: &mut Challenger,
    open_input: impl Fn(
        usize,
        &Folding::InputProof,
    ) -> Result<Vec<(usize, Challenge)>, Folding::InputError>,
) -> Result<(), FriError<M::Error, Folding::InputError>>
where
    Val: Field,
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

    // Observe the claimed final polynomial.
    challenger.observe_algebra_element(proof.final_poly);

    if proof.query_proofs.len() != params.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    // Check PoW.
    if !challenger.check_witness(params.query_proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    // Validate that all query proofs have the same number of commit phase openings
    if !proof
        .query_proofs
        .iter()
        .all(|qp| qp.commit_phase_openings.len() == proof.commit_phase_commits.len())
    {
        return Err(FriError::InvalidProofShape);
    }

    // With variable arity, compute log_max_height by summing all log_arities
    let total_log_reduction: usize = proof
        .query_proofs
        .first()
        .map(|qp| {
            qp.commit_phase_openings
                .iter()
                .map(|o| o.log_arity as usize)
                .sum()
        })
        .unwrap_or(0);
    let log_max_height = total_log_reduction + params.log_blowup;

    for qp in &proof.query_proofs {
        let index = challenger.sample_bits(log_max_height + folding.extra_query_index_bits());
        let ro = open_input(index, &qp.input_proof).map_err(FriError::InputError)?;

        debug_assert!(
            ro.iter().tuple_windows().all(|((l, _), (r, _))| l > r),
            "reduced openings sorted by height descending"
        );

        // Starting at the evaluation at `index` of the initial domain,
        // perform fri folds until the domain size reaches the final domain size.
        // Check after each fold that the group of sibling evaluations at the current
        // node match the commitment.
        let folded_eval = verify_query(
            folding,
            params,
            index >> folding.extra_query_index_bits(),
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
        )?;

        // As we fold until the polynomial is constant, proof.final_poly should be a constant value and
        // we do not need to do any polynomial evaluations.
        if folded_eval != proof.final_poly {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(())
}

type CommitStep<'a, F, M> = (
    (
        &'a F, // The challenge point beta used for the next fold of Circle-FRI evaluations.
        &'a <M as Mmcs<F>>::Commitment, // A commitment to the Circle-FRI evaluations on the current domain.
    ),
    &'a CircleCommitPhaseProofStep<F, M>, // The siblings and opening proof for the current Circle-FRI node.
);

/// Verifies a single query chain in the Circle-FRI proof.
///
/// Given an initial `index` corresponding to a point in the initial domain
/// and a series of `reduced_openings` corresponding to evaluations of
/// polynomials to be added in at specific domain sizes, perform the standard
/// sequence of Circle-FRI folds, checking at each step that the group of sibling evaluations
/// match the commitment.
///
/// With variable arity, each round may fold by a different factor determined by the
/// `log_arity` field in the opening.
fn verify_query<'a, Folding, F, EF, M>(
    folding: &Folding,
    params: &FriParameters<M>,
    mut index: usize,
    steps: impl ExactSizeIterator<Item = CommitStep<'a, EF, M>>,
    reduced_openings: Vec<(usize, EF)>,
    log_max_height: usize,
) -> Result<EF, FriError<M::Error, Folding::InputError>>
where
    F: Field,
    EF: ExtensionField<F>,
    M: Mmcs<EF> + 'a,
    Folding: FriFoldingStrategy<F, EF>,
{
    let mut folded_eval = EF::ZERO;
    let mut ro_iter = reduced_openings.into_iter().peekable();

    // Track the current log_height as we fold down
    let mut log_current_height = log_max_height;

    // We start with evaluations over a domain of size (1 << log_max_height). We fold
    // using FRI until the domain size reaches (1 << log_blowup).
    for ((&beta, comm), opening) in steps {
        let log_arity = opening.log_arity as usize;
        let arity = 1 << log_arity;

        // Validate that sibling_values has the expected length (arity - 1)
        if opening.sibling_values.len() != arity - 1 {
            return Err(FriError::InvalidProofShape);
        }

        // If there are new polynomials to roll in at this height, do so.
        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_current_height) {
            folded_eval += ro;
        }

        // Reconstruct the full evaluation row from self + siblings
        let index_in_group = index % arity;
        let mut evals = vec![EF::ZERO; arity];
        evals[index_in_group] = folded_eval;

        let mut sibling_idx = 0;
        for j in 0..arity {
            if j != index_in_group {
                evals[j] = opening.sibling_values[sibling_idx];
                sibling_idx += 1;
            }
        }

        // Compute the new height after folding
        let log_folded_height = log_current_height - log_arity;

        let dims = &[Dimensions {
            width: arity,
            height: 1 << log_folded_height,
        }];

        // Replace index with the index of the parent fri node.
        index >>= log_arity;

        // Verify the commitment to the evaluations of the sibling nodes.
        params
            .mmcs
            .verify_batch(
                comm,
                dims,
                index,
                BatchOpeningRef::new(&[evals.clone()], &opening.opening_proof),
            )
            .map_err(FriError::CommitPhaseMmcsError)?;

        // Fold the group of evaluations of sibling nodes into the evaluation of the parent fri node.
        folded_eval =
            folding.fold_row(index, log_folded_height, log_arity, beta, evals.into_iter());

        // Update current height
        log_current_height = log_folded_height;
    }

    // Verify we reached the expected final height
    if log_current_height != params.log_blowup {
        return Err(FriError::InvalidProofShape);
    }

    // If ro_iter is not empty, we failed to fold in some polynomial evaluations.
    if ro_iter.next().is_some() {
        return Err(FriError::InvalidProofShape);
    }

    // If we reached this point, we have verified that, starting at the initial index,
    // the chain of folds has produced folded_eval.
    Ok(folded_eval)
}
