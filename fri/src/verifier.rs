use alloc::collections::btree_map::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::{log2_strict_usize, reverse_bits_len};
use thiserror::Error;

use crate::{
    CommitPhaseProofStep, CommitmentWithOpeningPoints, FriFoldingStrategy, FriParameters, FriProof,
    QueryProof,
};

#[derive(Debug, Error)]
pub enum FriError<CommitMmcsErr, InputError>
where
    CommitMmcsErr: core::fmt::Debug,
    InputError: core::fmt::Debug,
{
    #[error("invalid proof shape")]
    InvalidProofShape,
    #[error("query {query}: commit phase opening count mismatch: expected {expected}, got {got}")]
    QueryCommitPhaseOpeningsCountMismatch {
        query: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "query {query}: commit phase log-arity schedule mismatch: expected {expected:?}, got {got:?}"
    )]
    QueryLogAritiesMismatch {
        query: usize,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("commit PoW witness count mismatch: expected {expected}, got {got}")]
    CommitPowWitnessCountMismatch { expected: usize, got: usize },
    #[error("final polynomial length mismatch: expected {expected}, got {got}")]
    FinalPolyLengthMismatch { expected: usize, got: usize },
    #[error("query proof count mismatch: expected {expected}, got {got}")]
    QueryProofCountMismatch { expected: usize, got: usize },
    #[error("missing initial reduced opening at log height {expected}")]
    MissingInitialReducedOpening { expected: usize },
    #[error("initial reduced opening height mismatch: expected {expected}, got {got}")]
    InitialReducedOpeningHeightMismatch { expected: usize, got: usize },
    #[error("round {round}: sibling values length mismatch: expected {expected}, got {got}")]
    SiblingValuesLengthMismatch {
        round: usize,
        expected: usize,
        got: usize,
    },
    #[error("final folded height mismatch: expected {expected}, got {got}")]
    FinalFoldHeightMismatch { expected: usize, got: usize },
    #[error(
        "unconsumed reduced openings remain after folding: next log height {next_log_height}, remaining {remaining}"
    )]
    UnconsumedReducedOpenings {
        next_log_height: usize,
        remaining: usize,
    },
    #[error("input proof batch count mismatch: expected {expected}, got {got}")]
    InputProofBatchCountMismatch { expected: usize, got: usize },
    #[error("batch {batch}: opened-values matrix count mismatch: expected {expected}, got {got}")]
    BatchOpenedValuesCountMismatch {
        batch: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "batch {batch}, matrix {matrix}, point {point}: evaluation count mismatch: expected {expected}, got {got}"
    )]
    PointEvaluationCountMismatch {
        batch: usize,
        matrix: usize,
        point: usize,
        expected: usize,
        got: usize,
    },
    #[error("commit phase MMCS error: {0:?}")]
    CommitPhaseMmcsError(CommitMmcsErr),
    #[error("input error: {0:?}")]
    InputError(InputError),
    #[error("final polynomial mismatch: evaluation does not match expected value")]
    FinalPolyMismatch,
    #[error("invalid proof-of-work witness")]
    InvalidPowWitness,
}

/// A chain of FRI input openings allowing a verifier to check a sequence of
/// FRI folds and rolls. The first element of each pair indicates the round of
/// fri in which the input should be rolled in. The second element is the opening.
type FriOpenings<F> = Vec<(usize, F)>;

/// Verifies a FRI proof.
///
/// Arguments:
/// - `folding`: The FRI folding scheme used by the prover.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `proof`: The proof to verify.
/// - `challenger`: The Fiat-Shamir challenger.
/// - `commitments_with_opening_points`: A vector of joint commitments to collections of matrices
///   and openings of those matrices at a collection of points.
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

    // Validate that all query proofs have the same number of commit phase openings
    let expected_rounds = proof.commit_phase_commits.len();
    for (query, qp) in proof.query_proofs.iter().enumerate() {
        let got_rounds = qp.commit_phase_openings.len();
        if got_rounds != expected_rounds {
            return Err(FriError::QueryCommitPhaseOpeningsCountMismatch {
                query,
                expected: expected_rounds,
                got: got_rounds,
            });
        }
    }

    // Extract the per-round folding arities from the proof and ensure they are consistent.
    let log_arities: Vec<usize> = proof
        .query_proofs
        .first()
        .map(|qp| {
            qp.commit_phase_openings
                .iter()
                .map(|o| o.log_arity as usize)
                .collect()
        })
        .unwrap_or_default();

    for (query, qp) in proof.query_proofs.iter().enumerate().skip(1) {
        let got_log_arities = qp
            .commit_phase_openings
            .iter()
            .map(|o| o.log_arity as usize)
            .collect::<Vec<_>>();
        if got_log_arities != log_arities {
            return Err(FriError::QueryLogAritiesMismatch {
                query,
                expected: log_arities,
                got: got_log_arities,
            });
        }
    }

    // With variable arity, we compute log_global_max_height by summing all log_arities.
    // Each round reduces the domain size by its log_arity.
    let total_log_reduction: usize = log_arities.iter().sum();
    let log_global_max_height = total_log_reduction + params.log_blowup + params.log_final_poly_len;

    if proof.commit_pow_witnesses.len() != proof.commit_phase_commits.len() {
        return Err(FriError::CommitPowWitnessCountMismatch {
            expected: proof.commit_phase_commits.len(),
            got: proof.commit_pow_witnesses.len(),
        });
    }

    // Generate all of the random challenges for the FRI rounds, checking PoW per round.
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .zip(&proof.commit_pow_witnesses)
        .map(|(comm, witness)| {
            // Observe the commitment, check the PoW witness, then sample the
            // folding challenge.
            challenger.observe(comm.clone());
            if !challenger.check_witness(params.commit_proof_of_work_bits, *witness) {
                return Err(FriError::InvalidPowWitness);
            }
            Ok(challenger.sample_algebra_element())
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Ensure that the final polynomial has the expected degree.
    if proof.final_poly.len() != params.final_poly_len() {
        return Err(FriError::FinalPolyLengthMismatch {
            expected: params.final_poly_len(),
            got: proof.final_poly.len(),
        });
    }

    // Observe all coefficients of the final polynomial.
    challenger.observe_algebra_slice(&proof.final_poly);

    // Ensure that we have the expected number of FRI query proofs.
    if proof.query_proofs.len() != params.num_queries {
        return Err(FriError::QueryProofCountMismatch {
            expected: params.num_queries,
            got: proof.query_proofs.len(),
        });
    }

    // Bind the variable-arity schedule into the transcript before query grinding.
    for &log_arity in &log_arities {
        challenger.observe(Val::from_usize(log_arity));
    }

    // Check PoW.
    if !challenger.check_witness(params.query_proof_of_work_bits, proof.query_pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    // The log of the final domain size.
    let log_final_height = params.log_blowup + params.log_final_poly_len;

    for QueryProof {
        input_proof,
        commit_phase_openings,
    } in proof.query_proofs.iter()
    {
        // For each query proof, we start by generating the random index.
        let index =
            challenger.sample_bits(log_global_max_height + folding.extra_query_index_bits());

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

        let fold_data_iter = betas
            .iter()
            .zip(proof.commit_phase_commits.iter())
            .zip(commit_phase_openings.iter());

        // Starting at the evaluation at `index` of the initial domain,
        // perform FRI folds until the domain size reaches the final domain size.
        // Check after each fold that the pair of sibling evaluations at the current
        // node match the commitment.
        let folded_eval = verify_query(
            folding,
            params,
            &mut domain_index,
            fold_data_iter,
            ro,
            log_global_max_height,
            log_final_height,
        )?;

        // We open the final polynomial at index `domain_index`, which corresponds to evaluating
        // the polynomial at x^k, where x is the 2-adic generator of order `max_height` and k is
        // `reverse_bits_len(domain_index, log_global_max_height)`.
        let x = Val::two_adic_generator(log_global_max_height)
            .exp_u64(reverse_bits_len(domain_index, log_global_max_height) as u64);

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
/// sequence of FRI folds, checking at each step that the group of sibling evaluations
/// matches the commitment.
///
/// With variable arity, each round may fold by a different factor determined by the
/// `log_arity` field in the opening.
///
/// Arguments:
/// - `folding`: The FRI folding scheme used by the prover.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `start_index`: The opening index for the unfolded polynomial.
/// - `fold_data_iter`: An iterator containing, for each fold, the beta challenge, polynomial commitment
///   and commitment opening at the appropriate index.
/// - `reduced_openings`: A vector of pairs of a size and an opening. The opening is a linear combination
///   of all input polynomials of that size opened at the appropriate index. Each opening is added into the
///   the FRI folding chain once the domain size reaches the size specified in the pair.
/// - `log_global_max_height`: The log of the maximum domain size.
/// - `log_final_height`: The log of the final domain size.
#[inline]
fn verify_query<'a, Folding, F, EF, M>(
    folding: &Folding,
    params: &FriParameters<M>,
    start_index: &mut usize,
    fold_data_iter: impl ExactSizeIterator<Item = CommitStep<'a, EF, M>>,
    reduced_openings: FriOpenings<EF>,
    log_global_max_height: usize,
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
    let Some((first_log_height, _)) = ro_iter.peek() else {
        return Err(FriError::MissingInitialReducedOpening {
            expected: log_global_max_height,
        });
    };
    if *first_log_height != log_global_max_height {
        return Err(FriError::InitialReducedOpeningHeightMismatch {
            expected: log_global_max_height,
            got: *first_log_height,
        });
    }
    let mut folded_eval = ro_iter.next().unwrap().1;

    // Track the current log_height as we fold down
    let mut log_current_height = log_global_max_height;

    // We start with evaluations over a domain of size (1 << log_global_max_height). We fold
    // using FRI until the domain size reaches (1 << log_final_height).
    for (round, ((&beta, comm), opening)) in fold_data_iter.enumerate() {
        let log_arity = opening.log_arity as usize;
        let arity = 1 << log_arity;

        // Validate that sibling_values has the expected length (arity - 1)
        if opening.sibling_values.len() != arity - 1 {
            return Err(FriError::SiblingValuesLengthMismatch {
                round,
                expected: arity - 1,
                got: opening.sibling_values.len(),
            });
        }

        // Reconstruct the full evaluation row from self + siblings
        let index_in_group = *start_index % arity;
        let mut evals = vec![EF::ZERO; arity];
        evals[index_in_group] = folded_eval;

        let mut sibling_idx = 0;
        #[allow(clippy::needless_range_loop)]
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

        // Replace index with the index of the parent FRI node.
        *start_index >>= log_arity;

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

        // Fold the group of sibling nodes to get the evaluation of the parent FRI node.
        folded_eval = folding.fold_row(
            *start_index,
            log_folded_height,
            log_arity,
            beta,
            evals.into_iter(),
        );

        // Update current height
        log_current_height = log_folded_height;

        // If there are new polynomials to roll in at the folded height, do so.
        //
        // Each element of `ro_iter` is the evaluation of a reduced opening polynomial, which is itself
        // a random linear combination `f_{i, 0}(x) + alpha f_{i, 1}(x) + ...`, but when we add it
        // to the current folded polynomial evaluation claim, we need to multiply by a new random factor
        // since `f_{i, 0}` has no leading coefficient.
        //
        // We use `beta^arity` as the random factor to maintain independence.
        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height) {
            let beta_pow = beta.exp_power_of_2(log_arity);
            folded_eval += beta_pow * ro;
        }
    }

    // Verify we reached the expected final height
    if log_current_height != log_final_height {
        return Err(FriError::FinalFoldHeightMismatch {
            expected: log_final_height,
            got: log_current_height,
        });
    }

    // If ro_iter is not empty, we failed to fold in some polynomial evaluations.
    if let Some((next_log_height, _)) = ro_iter.next() {
        return Err(FriError::UnconsumedReducedOpenings {
            next_log_height,
            remaining: 1 + ro_iter.count(),
        });
    }

    // If we reached this point, we have verified that, starting at the initial index,
    // the chain of folds has produced folded_eval.
    Ok(folded_eval)
}

/// Given an index and a collection of opening proofs, check all opening proofs and combine
/// the opened values into the FRI inputs along the path specified by the index.
///
/// In cases where the maximum height of a batch of matrices is smaller than the
/// global max height, shift the index down to compensate.
///
/// We combine the functions by mapping each function and opening point pair to `(f(z) - f(x))/(z - x)`
/// and then combining functions of the same degree using the challenge alpha.
///
/// ## Arguments:
/// - `params`: The FRI parameters.
/// - `log_global_max_height`: The log of the maximum height of the input matrices.
/// - `index`: The index at which to open the functions.
/// - `input_proof`: A vector of batch openings with each opening containing a
///   list of opened values for a collection of matrices along with a batched opening proof.
/// - `alpha`: The challenge used to combine the functions.
/// - `input_mmcs`: The input multi-matrix commitment scheme.
/// - `commitments_with_opening_points`: A vector of joint commitments to collections of matrices
///   and openings of those matrices at a collection of points.
#[inline]
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

    if input_proof.len() != commitments_with_opening_points.len() {
        return Err(FriError::InputProofBatchCountMismatch {
            expected: commitments_with_opening_points.len(),
            got: input_proof.len(),
        });
    }

    // For each batch commitment and opening proof
    for (batch, (batch_opening, (batch_commit, mats))) in input_proof
        .iter()
        .zip(commitments_with_opening_points.iter())
        .enumerate()
    {
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

        // If the maximum height of the batch is smaller than the global max height,
        // we need to correct the index by right shifting it.
        // If the batch is empty, we set the index to 0.
        let reduced_index = batch_heights
            .iter()
            .max()
            .map(|&h| index >> (log_global_max_height - log2_strict_usize(h)))
            .unwrap_or(0);

        if batch_opening.opened_values.len() != mats.len() {
            return Err(FriError::BatchOpenedValuesCountMismatch {
                batch,
                expected: mats.len(),
                got: batch_opening.opened_values.len(),
            });
        }

        input_mmcs
            .verify_batch(
                batch_commit,
                &batch_dims,
                reduced_index,
                batch_opening.into(),
            )
            .map_err(FriError::InputError)?;

        // For each matrix in the commitment
        for (matrix, (mat_opening, (mat_domain, mat_points_and_values))) in batch_opening
            .opened_values
            .iter()
            .zip(mats.iter())
            .enumerate()
        {
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
            for (point, (z, ps_at_z)) in mat_points_and_values.iter().enumerate() {
                let quotient = (*z - x).inverse();
                if mat_opening.len() != ps_at_z.len() {
                    return Err(FriError::PointEvaluationCountMismatch {
                        batch,
                        matrix,
                        point,
                        expected: mat_opening.len(),
                        got: ps_at_z.len(),
                    });
                }
                for (&p_at_x, &p_at_z) in mat_opening.iter().zip(ps_at_z.iter()) {
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
        if let Some((_, ro)) = reduced_openings.get(&params.log_blowup)
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

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{CanSampleBits, DuplexChallenger};
    use p3_commit::{ExtensionMmcs, Pcs};
    use p3_dft::Radix2Dit;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::{TwoAdicFriFolding, TwoAdicFriPcs};

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type TestChallenger = DuplexChallenger<Val, Perm, 16, 8>;
    type MyPcs = TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs>;
    type Proof = FriProof<Challenge, ChallengeMmcs, Val, Vec<BatchOpening<Val, ValMmcs>>>;
    type Folding =
        TwoAdicFriFolding<Vec<BatchOpening<Val, ValMmcs>>, <ValMmcs as Mmcs<Val>>::Error>;

    struct TestFixture {
        proof: Proof,
        fri_params: FriParameters<ChallengeMmcs>,
        challenger: TestChallenger,
        cwop: Vec<
            CommitmentWithOpeningPoints<
                Challenge,
                <ValMmcs as Mmcs<Val>>::Commitment,
                TwoAdicMultiplicativeCoset<Val>,
            >,
        >,
        input_mmcs: ValMmcs,
        alpha: Challenge,
        betas: Vec<Challenge>,
        log_global_max_height: usize,
        log_final_height: usize,
        first_query_index: usize,
        first_query_ro: FriOpenings<Challenge>,
    }

    type VerifyResult = Result<
        (),
        FriError<<ChallengeMmcs as Mmcs<Challenge>>::Error, <ValMmcs as Mmcs<Val>>::Error>,
    >;

    impl TestFixture {
        fn folding(&self) -> Folding {
            TwoAdicFriFolding(core::marker::PhantomData)
        }

        fn verify(&self) -> VerifyResult {
            let mut ch = self.challenger.clone();
            verify_fri(
                &self.folding(),
                &self.fri_params,
                &self.proof,
                &mut ch,
                &self.cwop,
                &self.input_mmcs,
            )
        }

        fn verify_with_proof(&self, proof: &Proof) -> VerifyResult {
            let mut ch = self.challenger.clone();
            verify_fri(
                &self.folding(),
                &self.fri_params,
                proof,
                &mut ch,
                &self.cwop,
                &self.input_mmcs,
            )
        }
    }

    fn make_test_fixture() -> TestFixture {
        let mut rng = SmallRng::seed_from_u64(12345);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let input_mmcs = ValMmcs::new(hash.clone(), compress.clone(), 0);
        let fri_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress, 0));

        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: fri_mmcs,
        };

        let dft = Radix2Dit::default();
        let pcs = MyPcs::new(dft, input_mmcs.clone(), fri_params.clone());

        // Two polynomials of different sizes in one batch.
        let poly_log_sizes: [u8; 2] = [3, 4];
        let val_sizes: Vec<Val> = poly_log_sizes.iter().map(|&i| Val::from_u8(i)).collect();

        // -- prover --
        let mut p_challenger = TestChallenger::new(perm.clone());
        p_challenger.observe_slice(&val_sizes);

        let evaluations: Vec<_> = poly_log_sizes
            .iter()
            .map(|&deg_bits| {
                let deg = 1usize << deg_bits;
                (
                    <MyPcs as Pcs<Challenge, TestChallenger>>::natural_domain_for_degree(&pcs, deg),
                    RowMajorMatrix::<Val>::rand_nonzero(&mut rng, deg, 2),
                )
            })
            .collect();

        let (commitment, prover_data) =
            <MyPcs as Pcs<Challenge, TestChallenger>>::commit(&pcs, evaluations);
        p_challenger.observe(&commitment);
        let zeta: Challenge = p_challenger.sample_algebra_element();

        let open_data = vec![(&prover_data, vec![vec![zeta]; poly_log_sizes.len()])];
        let (opened_values, proof) = pcs.open(open_data, &mut p_challenger);

        // -- verifier challenger up to the verify_fri entry point --
        let mut v_ch = TestChallenger::new(perm);
        v_ch.observe_slice(&val_sizes);
        v_ch.observe(&commitment);
        let v_zeta: Challenge = v_ch.sample_algebra_element();
        debug_assert_eq!(zeta, v_zeta);

        let cwop: Vec<
            CommitmentWithOpeningPoints<
                Challenge,
                <ValMmcs as Mmcs<Val>>::Commitment,
                TwoAdicMultiplicativeCoset<Val>,
            >,
        > = vec![(
            commitment,
            poly_log_sizes
                .iter()
                .map(|&s| {
                    <MyPcs as Pcs<Challenge, TestChallenger>>::natural_domain_for_degree(
                        &pcs,
                        1 << s,
                    )
                })
                .zip(opened_values.into_iter().flatten().flatten())
                .map(|(domain, value)| (domain, vec![(v_zeta, value)]))
                .collect(),
        )];

        // Observe evaluations, matching what TwoAdicFriPcs::verify does.
        for cwop_entry in cwop.iter() {
            let mats = &cwop_entry.1;
            for mat_entry in mats.iter() {
                let points_and_values = &mat_entry.1;
                for (_, point) in points_and_values.iter() {
                    v_ch.observe_algebra_slice(point);
                }
            }
        }

        // Save the challenger state that verify_fri will receive.
        let saved_ch = v_ch.clone();

        // Replay the verify_fri preamble to extract intermediate values.
        let alpha: Challenge = v_ch.sample_algebra_element();

        let betas: Vec<Challenge> = proof
            .commit_phase_commits
            .iter()
            .zip(&proof.commit_pow_witnesses)
            .map(|(comm, witness)| {
                v_ch.observe(comm.clone());
                assert!(v_ch.check_witness(0, *witness));
                v_ch.sample_algebra_element()
            })
            .collect();

        v_ch.observe_algebra_slice(&proof.final_poly);

        let log_arities: Vec<usize> = proof.query_proofs[0]
            .commit_phase_openings
            .iter()
            .map(|o| o.log_arity as usize)
            .collect();
        for &la in &log_arities {
            v_ch.observe(Val::from_usize(la));
        }
        assert!(v_ch.check_witness(0, proof.query_pow_witness));

        let total_log_reduction: usize = log_arities.iter().sum();
        let log_global_max_height =
            total_log_reduction + fri_params.log_blowup + fri_params.log_final_poly_len;
        let log_final_height = fri_params.log_blowup + fri_params.log_final_poly_len;

        let first_query_index = v_ch.sample_bits(log_global_max_height);

        let first_query_ro = open_input(
            &fri_params,
            log_global_max_height,
            first_query_index,
            &proof.query_proofs[0].input_proof,
            alpha,
            &input_mmcs,
            &cwop,
        )
        .expect("open_input must succeed on a valid proof");

        TestFixture {
            proof,
            fri_params,
            challenger: saved_ch,
            cwop,
            input_mmcs,
            alpha,
            betas,
            log_global_max_height,
            log_final_height,
            first_query_index,
            first_query_ro,
        }
    }

    // ---------------------------------------------------------------
    // verify_fri: top-level proof shape checks
    // ---------------------------------------------------------------

    #[test]
    fn query_commit_phase_openings_count_mismatch() {
        let fix = make_test_fixture();
        let mut proof = fix.proof.clone();

        // Drop one commit-phase opening from the first query so its
        // count diverges from the number of commit-phase commitments.
        let original_len = proof.query_proofs[0].commit_phase_openings.len();
        proof.query_proofs[0].commit_phase_openings.pop();

        let err = fix.verify_with_proof(&proof).unwrap_err();
        match err {
            FriError::QueryCommitPhaseOpeningsCountMismatch {
                query,
                expected,
                got,
            } => {
                assert_eq!(query, 0);
                assert_eq!(expected, original_len);
                assert_eq!(got, original_len - 1);
            }
            other => panic!("expected QueryCommitPhaseOpeningsCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn query_log_arities_mismatch() {
        let fix = make_test_fixture();
        let mut proof = fix.proof.clone();

        // Flip the log_arity of the first opening in the second query
        // so the arity schedule diverges from query 0.
        let original = proof.query_proofs[1].commit_phase_openings[0].log_arity;
        proof.query_proofs[1].commit_phase_openings[0].log_arity = original + 1;

        let err = fix.verify_with_proof(&proof).unwrap_err();
        match err {
            FriError::QueryLogAritiesMismatch {
                query, expected, ..
            } => {
                assert_eq!(query, 1);
                assert_eq!(expected[0], original as usize);
            }
            other => panic!("expected QueryLogAritiesMismatch, got {other:?}"),
        }
    }

    #[test]
    fn commit_pow_witness_count_mismatch() {
        let fix = make_test_fixture();
        let mut proof = fix.proof.clone();

        // Add an extra PoW witness so the count exceeds the number
        // of commit-phase commitments.
        let expected_len = proof.commit_phase_commits.len();
        proof.commit_pow_witnesses.push(Val::ZERO);

        let err = fix.verify_with_proof(&proof).unwrap_err();
        match err {
            FriError::CommitPowWitnessCountMismatch { expected, got } => {
                assert_eq!(expected, expected_len);
                assert_eq!(got, expected_len + 1);
            }
            other => panic!("expected CommitPowWitnessCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn final_poly_length_mismatch() {
        let fix = make_test_fixture();
        let mut proof = fix.proof.clone();

        // Append a spurious coefficient so the final polynomial
        // exceeds the length dictated by the parameters.
        let expected_len = fix.fri_params.final_poly_len();
        proof.final_poly.push(Challenge::ZERO);

        let err = fix.verify_with_proof(&proof).unwrap_err();
        match err {
            FriError::FinalPolyLengthMismatch { expected, got } => {
                assert_eq!(expected, expected_len);
                assert_eq!(got, expected_len + 1);
            }
            other => panic!("expected FinalPolyLengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn query_proof_count_mismatch() {
        let fix = make_test_fixture();
        let mut proof = fix.proof.clone();

        // Remove one query proof so the count falls below num_queries.
        proof.query_proofs.pop();

        let err = fix.verify_with_proof(&proof).unwrap_err();
        match err {
            FriError::QueryProofCountMismatch { expected, got } => {
                assert_eq!(expected, fix.fri_params.num_queries);
                assert_eq!(got, fix.fri_params.num_queries - 1);
            }
            other => panic!("expected QueryProofCountMismatch, got {other:?}"),
        }
    }

    // ---------------------------------------------------------------
    // verify_query: fold-chain checks (called directly)
    // ---------------------------------------------------------------

    #[test]
    fn missing_initial_reduced_opening() {
        let fix = make_test_fixture();

        // An empty reduced_openings vector means no polynomial was
        // committed; the fold chain cannot start.
        let betas: Vec<Challenge> = vec![];
        let commits: Vec<<ChallengeMmcs as Mmcs<Challenge>>::Commitment> = vec![];
        let openings: Vec<CommitPhaseProofStep<Challenge, ChallengeMmcs>> = vec![];
        let fold_data = betas.iter().zip(commits.iter()).zip(openings.iter());

        let mut idx = 0usize;
        let err = verify_query::<Folding, Val, Challenge, ChallengeMmcs>(
            &fix.folding(),
            &fix.fri_params,
            &mut idx,
            fold_data,
            vec![],
            fix.log_global_max_height,
            fix.log_final_height,
        )
        .unwrap_err();

        match err {
            FriError::MissingInitialReducedOpening { expected } => {
                assert_eq!(expected, fix.log_global_max_height);
            }
            other => panic!("expected MissingInitialReducedOpening, got {other:?}"),
        }
    }

    #[test]
    fn initial_reduced_opening_height_mismatch() {
        let fix = make_test_fixture();

        // Provide a reduced opening whose log-height does not match
        // the global maximum, so the fold chain rejects it.
        let wrong_height = fix.log_global_max_height - 1;
        let ro = vec![(wrong_height, Challenge::ONE)];

        let betas: Vec<Challenge> = vec![];
        let commits: Vec<<ChallengeMmcs as Mmcs<Challenge>>::Commitment> = vec![];
        let openings: Vec<CommitPhaseProofStep<Challenge, ChallengeMmcs>> = vec![];
        let fold_data = betas.iter().zip(commits.iter()).zip(openings.iter());

        let mut idx = 0usize;
        let err = verify_query::<Folding, Val, Challenge, ChallengeMmcs>(
            &fix.folding(),
            &fix.fri_params,
            &mut idx,
            fold_data,
            ro,
            fix.log_global_max_height,
            fix.log_final_height,
        )
        .unwrap_err();

        match err {
            FriError::InitialReducedOpeningHeightMismatch { expected, got } => {
                assert_eq!(expected, fix.log_global_max_height);
                assert_eq!(got, wrong_height);
            }
            other => panic!("expected InitialReducedOpeningHeightMismatch, got {other:?}"),
        }
    }

    #[test]
    fn sibling_values_length_mismatch() {
        let fix = make_test_fixture();
        let mut proof = fix.proof.clone();

        // Add an extra sibling value to the first commit-phase
        // opening of every query so the arity check fails.
        for qp in &mut proof.query_proofs {
            qp.commit_phase_openings[0]
                .sibling_values
                .push(Challenge::ZERO);
        }

        let err = fix.verify_with_proof(&proof).unwrap_err();
        match err {
            FriError::SiblingValuesLengthMismatch {
                round,
                expected,
                got,
            } => {
                assert_eq!(round, 0);
                // Binary folding: arity 2, expected siblings = 1.
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            other => panic!("expected SiblingValuesLengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn final_fold_height_mismatch() {
        let fix = make_test_fixture();

        // Use the real fold data for the first query but drop the
        // last round so folding stops one step early.
        let n = fix.proof.commit_phase_commits.len();
        assert!(n >= 2, "need at least two fold rounds");

        let truncated_betas = &fix.betas[..n - 1];
        let truncated_commits = &fix.proof.commit_phase_commits[..n - 1];
        let truncated_openings = &fix.proof.query_proofs[0].commit_phase_openings[..n - 1];

        let fold_data = truncated_betas
            .iter()
            .zip(truncated_commits.iter())
            .zip(truncated_openings.iter());

        let mut idx = fix.first_query_index;
        let err = verify_query::<Folding, Val, Challenge, ChallengeMmcs>(
            &fix.folding(),
            &fix.fri_params,
            &mut idx,
            fold_data,
            fix.first_query_ro.clone(),
            fix.log_global_max_height,
            fix.log_final_height,
        )
        .unwrap_err();

        match err {
            FriError::FinalFoldHeightMismatch { expected, got } => {
                assert_eq!(expected, fix.log_final_height);
                // Stopped one binary fold early.
                assert_eq!(got, fix.log_final_height + 1);
            }
            other => panic!("expected FinalFoldHeightMismatch, got {other:?}"),
        }
    }

    #[test]
    fn unconsumed_reduced_openings() {
        let fix = make_test_fixture();

        // Append an extra reduced opening at a height below the
        // final fold height so no fold round ever consumes it.
        let mut ro = fix.first_query_ro.clone();
        ro.push((0, Challenge::ONE));

        let fold_data = fix
            .betas
            .iter()
            .zip(fix.proof.commit_phase_commits.iter())
            .zip(fix.proof.query_proofs[0].commit_phase_openings.iter());

        let mut idx = fix.first_query_index;
        let err = verify_query::<Folding, Val, Challenge, ChallengeMmcs>(
            &fix.folding(),
            &fix.fri_params,
            &mut idx,
            fold_data,
            ro,
            fix.log_global_max_height,
            fix.log_final_height,
        )
        .unwrap_err();

        match err {
            FriError::UnconsumedReducedOpenings {
                next_log_height,
                remaining,
            } => {
                assert_eq!(next_log_height, 0);
                assert_eq!(remaining, 1);
            }
            other => panic!("expected UnconsumedReducedOpenings, got {other:?}"),
        }
    }

    // ---------------------------------------------------------------
    // open_input: input opening shape checks (called directly)
    // ---------------------------------------------------------------

    #[test]
    fn input_proof_batch_count_mismatch() {
        let fix = make_test_fixture();

        // Pass an empty input proof vector while the verifier
        // expects one batch, creating a count mismatch.
        let empty_input: Vec<BatchOpening<Val, ValMmcs>> = vec![];
        let err = open_input(
            &fix.fri_params,
            fix.log_global_max_height,
            0,
            &empty_input,
            fix.alpha,
            &fix.input_mmcs,
            &fix.cwop,
        )
        .unwrap_err();

        match err {
            FriError::InputProofBatchCountMismatch { expected, got } => {
                assert_eq!(expected, fix.cwop.len());
                assert_eq!(got, 0);
            }
            other => panic!("expected InputProofBatchCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn batch_opened_values_count_mismatch() {
        let fix = make_test_fixture();
        let mut input_proof = fix.proof.query_proofs[0].input_proof.clone();

        // Remove one matrix opening from the first batch so the
        // count diverges from the number of committed matrices.
        let original_len = input_proof[0].opened_values.len();
        input_proof[0].opened_values.pop();

        let err = open_input(
            &fix.fri_params,
            fix.log_global_max_height,
            fix.first_query_index,
            &input_proof,
            fix.alpha,
            &fix.input_mmcs,
            &fix.cwop,
        )
        .unwrap_err();

        match err {
            FriError::BatchOpenedValuesCountMismatch {
                batch,
                expected,
                got,
            } => {
                assert_eq!(batch, 0);
                assert_eq!(expected, original_len);
                assert_eq!(got, original_len - 1);
            }
            other => panic!("expected BatchOpenedValuesCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn point_evaluation_count_mismatch() {
        let fix = make_test_fixture();
        let mut cwop = fix.cwop.clone();

        // Add a spurious evaluation value to the first point of
        // the first matrix so the count exceeds the opened columns.
        cwop[0].1[0].1[0].1.push(Challenge::ZERO);

        let err = open_input(
            &fix.fri_params,
            fix.log_global_max_height,
            fix.first_query_index,
            &fix.proof.query_proofs[0].input_proof,
            fix.alpha,
            &fix.input_mmcs,
            &cwop,
        )
        .unwrap_err();

        match err {
            FriError::PointEvaluationCountMismatch {
                batch,
                matrix,
                point,
                ..
            } => {
                assert_eq!(batch, 0);
                assert_eq!(matrix, 0);
                assert_eq!(point, 0);
            }
            other => panic!("expected PointEvaluationCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn valid_proof_passes() {
        let fix = make_test_fixture();
        fix.verify().expect("valid proof should pass verification");
    }
}
