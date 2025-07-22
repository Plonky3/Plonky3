use alloc::collections::btree_map::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::zip_eq::zip_eq;
use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::{
    CommitPhaseProofStep, CommitmentWithOpeningPoints, FriFoldingStrategy, FriParameters, FriProof,
    QueryProof,
};

#[derive(Debug)]
pub enum FriError<CommitMmcsErr, InputError> {
    InvalidProofShape,
    CommitPhaseMmcsError(CommitMmcsErr),
    InputError(InputError),
    FinalPolyMismatch,
    InvalidPowWitness,
    MissingInput,
}

/// A chain of FRI input openings allowing a verifier to check a sequence of
/// FRI folds and rolls. The first element of each pair indicates the round of
/// fri in which the input should be rolled in. The second element is the opening.
pub type FriOpenings<F> = Vec<(usize, F)>;

/// Verifies a FRI proof.
///
/// Arguments:
/// - `folding`: The FRI folding scheme used by the prover.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `proof`: The proof to verify.
/// - `challenger`: The Fiat-Shamir challenger.
/// - `open_input`: A function that takes an index and opening proofs and returns a vector of reduced openings
///   used as FRI inputs. The opening proofs prove that the values `f(x)` are the ones committed
///   to and these are then combined into the FRI inputs.
pub fn verify_fri<Folding, Val, Challenge, InputMmcs, FriMmcs, Challenger>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    proof: &FriProof<Challenge, FriMmcs, Challenger::Witness, Folding::InputProof>,
    challenger: &mut Challenger,
    commitments_with_opening_points: &[CommitmentWithOpeningPoints<
        Challenge,
        InputMmcs::Commitment,
        TwoAdicMultiplicativeCoset<Val>,
    >],
    input_mmcs: &InputMmcs,
) -> Result<(), FriError<FriMmcs::Error, InputMmcs::Error>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<FriMmcs::Commitment>,
    Folding: FriFoldingStrategy<
            Val,
            Challenge,
            InputError = InputMmcs::Error,
            InputProof = Vec<BatchOpening<Val, InputMmcs>>,
        >,
{
    // Generate the Batch combination challenge
    // Soundness Error: `|f|/|EF|` where `|f|` is the number of different functions of the form
    // `(f(zeta) - fi(x))/(zeta - x)` which need to be checked.
    // Explicitly, `|f|` is `commitments_with_opening_points.flatten().flatten().len()`
    // (i.e counting the number (point, claimed_evaluation) pairs).
    let alpha: Challenge = challenger.sample_algebra_element();

    // `commit_phase_commits.len()` is the number of folding steps, so the maximum polynomial degree will be
    // `commit_phase_commits.len() + self.fri.log_final_poly_len` and so, as the same blow-up is used for all
    // polynomials, the maximum matrix height is:
    let log_global_max_height =
        proof.commit_phase_commits.len() + params.log_blowup + params.log_final_poly_len;

    // Generate all of the random challenges for the FRI rounds.
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            // To match with the prover (and for security purposes),
            // we observe the commitment before sampling the challenge.
            challenger.observe(comm.clone());
            challenger.sample_algebra_element()
        })
        .collect();

    // Ensure that the final polynomial has the expected degree.
    if proof.final_poly.len() != params.final_poly_len() {
        return Err(FriError::InvalidProofShape);
    }

    // Observe all coefficients of the final polynomial.
    proof
        .final_poly
        .iter()
        .for_each(|x| challenger.observe_algebra_element(*x));

    // Ensure that we have the expected number of FRI query proofs.
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

    for QueryProof {
        input_proof,
        commit_phase_openings,
    } in &proof.query_proofs
    {
        // For each query proof, we start by generating the random index.
        let index = challenger.sample_bits(log_max_height + folding.extra_query_index_bits());

        // Next we open all polynomials `f` at the relevant index and combine them into our FRI inputs.
        let ro = open_input(
            params,
            log_global_max_height,
            index,
            input_proof,
            alpha,
            input_mmcs,
            commitments_with_opening_points,
        )?;

        debug_assert!(
            ro.iter().tuple_windows().all(|((l, _), (r, _))| l > r),
            "reduced openings sorted by height descending"
        );

        // If we queried extra bits, shift them off now.
        let mut domain_index = index >> folding.extra_query_index_bits();

        // Starting at the evaluation at `index` of the initial domain,
        // perform FRI folds until the domain size reaches the final domain size.
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
                commit_phase_openings,
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

        // Assuming all the checks passed, the final check is to ensure that the folded evaluation
        // matches the evaluation of the final polynomial sent by the prover.

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

/// Verifies a single query chain in the FRI proof. This is the verifier complement
/// to the prover's [`answer_query`] function.
///
/// Given an initial `index` corresponding to a point in the initial domain
/// and a series of `reduced_openings` corresponding to evaluations of
/// polynomials to be added in at specific domain sizes, perform the standard
/// sequence of FRI folds, checking at each step that the pair of sibling evaluations
/// matches the commitment.
///
/// Arguments:
/// - `folding`: The FRI folding scheme used by the prover.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `start_index`: The opening index for the unfolded polynomial. For folded polynomials
///   we use this this index right shifted by the number of folds.
/// - `fold_data_iter`: An iterator containing, for each fold, the beta challenge, polynomial commitment
///   and commitment opening at the appropriate index.
/// - `reduced_openings`: A vector of pairs of a size and an opening. The opening is a linear combination
///   of all input polynomials of that size opened at the appropriate index. Each opening is added into the
///   the FRI folding chain once the domain size reaches the size specified in the pair.
/// - `log_max_height`: The log of the maximum domain size.
/// - `log_final_height`: The log of the final domain size.
fn verify_query<'a, Folding, F, EF, M>(
    folding: &Folding,
    params: &FriParameters<M>,
    start_index: &mut usize,
    fold_data_iter: impl ExactSizeIterator<Item = CommitStep<'a, EF, M>>,
    reduced_openings: FriOpenings<EF>,
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

    // These checks are not essential to security,
    // but they should be satisfied by any non malicious prover.
    // ro_iter being empty means that we have committed to no polynomials at all and
    // we need to roll in a polynomial initially otherwise we are just folding a zero polynomial.
    if ro_iter.peek().is_none() || ro_iter.peek().unwrap().0 != log_max_height {
        return Err(FriError::InvalidProofShape);
    }
    let mut folded_eval = ro_iter.next().unwrap().1;

    // We start with evaluations over a domain of size (1 << log_max_height). We fold
    // using FRI until the domain size reaches (1 << log_final_height).
    for (log_folded_height, ((&beta, comm), opening)) in zip_eq(
        // zip_eq ensures that we have the right number of steps.
        (log_final_height..log_max_height).rev(),
        fold_data_iter,
        FriError::InvalidProofShape,
    )? {
        // Get the index of the other sibling of the current FRI node.
        let index_sibling = *start_index ^ 1;

        let mut evals = vec![folded_eval; 2];
        evals[index_sibling % 2] = opening.sibling_value;

        let dims = &[Dimensions {
            width: 2,
            height: 1 << log_folded_height,
        }];

        // Replace index with the index of the parent FRI node.
        *start_index >>= 1;

        // Verify the commitment to the evaluations of the sibling nodes.
        params
            .mmcs
            .verify_batch(
                comm,
                dims,
                *start_index,
                BatchOpeningRef::new(&[evals.clone()], &opening.opening_proof), // It's possible to remove the clone here but unnecessary as evals is tiny.
            )
            .map_err(FriError::CommitPhaseMmcsError)?;

        // Fold the pair of sibling nodes to get the evaluation of the parent FRI node.
        folded_eval = folding.fold_row(*start_index, log_folded_height, beta, evals.into_iter());

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

/// index is the query position we are checking
/// input_proof is a vector of batch openings. Each batch opening contains a
/// list of opened values for a collection of matrices along with a batched opening proof.
/// We check the proofs and then combine the functions by mapping each function and opening point
/// pair to `(f(z) - f(x))/(z - x)` and then combining functions of the same height using
/// the challenge alpha.
fn open_input<Val, Challenge, InputMmcs, FriMmcs>(
    params: &FriParameters<FriMmcs>,
    log_global_max_height: usize,
    index: usize,
    input_proof: &[BatchOpening<Val, InputMmcs>],
    alpha: Challenge,
    input_mmcs: &InputMmcs,
    commitments_with_opening_points: &[CommitmentWithOpeningPoints<
        Challenge,
        InputMmcs::Commitment,
        TwoAdicMultiplicativeCoset<Val>,
    >],
) -> Result<FriOpenings<Challenge>, FriError<FriMmcs::Error, InputMmcs::Error>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
{
    // For each log_height, we store the alpha power and compute the reduced opening.
    // log_height -> (alpha_pow, reduced_opening)
    let mut reduced_openings = BTreeMap::<usize, (Challenge, Challenge)>::new();

    // For each batch commitment and opening proof
    for (batch_opening, (batch_commit, mats)) in zip_eq(
        input_proof,
        commitments_with_opening_points,
        FriError::InvalidProofShape,
    )? {
        // Find the height of each matrix in the batch.
        // Currently we only check domain.size() as the shift is
        // assumed to always be Val::GENERATOR.
        let batch_heights = mats
            .iter()
            .map(|(domain, _)| domain.size() << params.log_blowup)
            .collect_vec();
        let batch_dims = batch_heights
            .iter()
            // TODO: MMCS doesn't really need width; we put 0 for now.
            .map(|&height| Dimensions { width: 0, height })
            .collect_vec();

        if let Some(batch_max_height) = batch_heights.iter().max() {
            let log_batch_max_height = log2_strict_usize(*batch_max_height);
            let bits_reduced = log_global_max_height - log_batch_max_height;
            let reduced_index = index >> bits_reduced;

            // Verify that the opened values match the commitment.
            input_mmcs.verify_batch(
                batch_commit,
                &batch_dims,
                reduced_index,
                batch_opening.into(),
            )
        } else {
            // Empty batch?
            input_mmcs.verify_batch(batch_commit, &[], 0, batch_opening.into())
        }
        .map_err(FriError::InputError)?;

        // For each matrix in the commitment
        for (mat_opening, (mat_domain, mat_points_and_values)) in zip_eq(
            &batch_opening.opened_values,
            mats,
            FriError::InvalidProofShape,
        )? {
            let log_height = log2_strict_usize(mat_domain.size()) + params.log_blowup;

            let bits_reduced = log_global_max_height - log_height;
            let rev_reduced_index = reverse_bits_len(index >> bits_reduced, log_height);

            // TODO: this can be nicer with domain methods?

            // Compute gh^i
            let x = Val::GENERATOR
                * Val::two_adic_generator(log_height).exp_u64(rev_reduced_index as u64);

            let (alpha_pow, ro) = reduced_openings
                .entry(log_height) // Get a mutable reference to the entry.
                .or_insert((Challenge::ONE, Challenge::ZERO));

            // For each polynomial `f` in our matrix, compute `(f(z) - f(x))/(z - x)`,
            // scale by the appropriate alpha power and add to the reduced opening for this log_height.
            for (z, ps_at_z) in mat_points_and_values {
                let quotient = (*z - x).inverse();
                for (&p_at_x, &p_at_z) in zip_eq(mat_opening, ps_at_z, FriError::InvalidProofShape)?
                {
                    // Note we just checked batch proofs to ensure p_at_x is correct.
                    // x, z were sent by the verifier.
                    // ps_at_z was sent to the verifier and we are using fri to prove it is correct.
                    *ro += *alpha_pow * (p_at_z - p_at_x) * quotient;
                    *alpha_pow *= alpha;
                }
            }
        }

        // `reduced_openings` would have a log_height = log_blowup entry only if there was a
        // trace matrix of height 1. In this case `f` is constant, so `f(zeta) - f(x))/(zeta - x)`
        // must equal `0`.
        if let Some((_alpha_pow, ro)) = reduced_openings.get(&params.log_blowup)
            && !ro.is_zero()
        {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    // Return reduced openings descending by log_height.
    Ok(reduced_openings
        .into_iter()
        .rev()
        .map(|(log_height, (_, ro))| (log_height, ro))
        .collect())
}
