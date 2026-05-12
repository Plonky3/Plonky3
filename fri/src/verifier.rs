use alloc::collections::btree_map::BTreeMap;
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
        let mut evals = EF::zero_vec(arity);
        evals[index_in_group] = folded_eval;

        let mut sibling_idx = 0;
        for (j, eval) in evals.iter_mut().enumerate() {
            if j != index_in_group {
                *eval = opening.sibling_values[sibling_idx];
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
    use alloc::vec;
    use core::marker::PhantomData;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::{BatchOpening, ExtensionMmcs, Mmcs, Pcs};
    use p3_dft::Radix2Dit;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::{
        CommitmentWithOpeningPoints, FriParameters, TwoAdicFriFolding, TwoAdicFriFoldingForMmcs,
        TwoAdicFriPcs,
    };

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Proof = FriProof<Challenge, ChallengeMmcs, Val, Vec<BatchOpening<Val, ValMmcs>>>;
    type Folding = TwoAdicFriFoldingForMmcs<Val, ValMmcs>;
    type TestError =
        FriError<<ChallengeMmcs as Mmcs<Challenge>>::Error, <ValMmcs as Mmcs<Val>>::Error>;

    /// All the data needed to invoke the top-level FRI verification.
    struct TestFixture {
        /// Protocol parameters (blowup, arity, queries, etc.).
        fri_params: FriParameters<ChallengeMmcs>,
        /// Base-field commitment scheme used for input polynomials.
        input_mmcs: ValMmcs,
        /// A valid proof produced by a real prover run.
        proof: Proof,
        /// Fiat-Shamir challenger already advanced past the opened-values
        /// observation step, ready for the verification entry point.
        challenger: Challenger,
        /// Commitment and opening-point data the verifier checks against.
        commitments_with_opening_points: Vec<
            CommitmentWithOpeningPoints<
                Challenge,
                <ValMmcs as Mmcs<Val>>::Commitment,
                TwoAdicMultiplicativeCoset<Val>,
            >,
        >,
    }

    /// Build a deterministic, minimal test fixture.
    ///
    /// Commits a single 8-row, 2-column trace, opens it at one
    /// random challenge point, and produces a valid FRI proof with:
    /// - 2 queries (minimum for arity-consistency checks)
    /// - binary folding (log arity 1)
    /// - blowup factor 2
    /// - zero proof-of-work bits
    ///
    /// # Proof Shape
    ///
    /// With degree 8 and blowup 2, the evaluation domain has 16 points.
    /// Binary folding halves the domain each round, producing 3 COMMIT
    /// rounds before reaching the final constant polynomial:
    ///
    /// ```text
    ///     Round 0: |L^(0)| = 16  -->  |L^(1)| = 8   (commit + fold)
    ///     Round 1: |L^(1)| = 8   -->  |L^(2)| = 4   (commit + fold)
    ///     Round 2: |L^(2)| = 4   -->  |L^(3)| = 2   (commit + fold)
    ///     Final:   1 coefficient (the constant polynomial)
    /// ```
    ///
    /// The resulting proof contains:
    /// - 3 commit-phase commitments (one per round)
    /// - 3 proof-of-work witnesses (one per round)
    /// - 2 query proofs, each with 3 commit-phase openings
    /// - 1 final polynomial coefficient
    ///
    /// The challenger is advanced past the opened-values observation,
    /// ready for the verification entry point.
    fn make_test_fixture() -> TestFixture {
        // Use a fixed seed so every test run is deterministic.
        let mut rng = SmallRng::seed_from_u64(42);

        // Build the permutation, hash, and compression for the Merkle tree.
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());

        // - One commitment scheme for base-field inputs,
        // - Another for extension-field FRI commitments.
        let input_mmcs = ValMmcs::new(hash.clone(), compress.clone(), 0);
        let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress, 0));

        // Minimal parameters that exercise all verifier paths cheaply.
        //
        // - blowup 2: evaluation domain is 2x the polynomial degree,
        //   the smallest blowup for a sound protocol
        // - final poly length 1: fold down to f^(r) with degree 0
        // - binary folding: each round halves the domain (arity 2)
        // - 2 queries: minimum for arity-schedule consistency checks
        //   across different query proofs
        // - 0 PoW bits: every witness is trivially valid
        let fri_params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 2,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };

        // Wrap the parameters and commitment scheme into a PCS instance.
        let pcs = TwoAdicFriPcs::new(Radix2Dit::default(), input_mmcs.clone(), fri_params.clone());

        // Commit to a single 8-row, 2-column trace matrix.
        // With blowup 2 the evaluation domain has 16 points (log height 4).
        // Binary folding produces 3 rounds: 16 -> 8 -> 4 -> 2.
        let log_degree = 3;
        let width = 2;
        let domain = <TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs> as Pcs<
            Challenge,
            Challenger,
        >>::natural_domain_for_degree(&pcs, 1 << log_degree);
        let trace = RowMajorMatrix::<Val>::rand_nonzero(&mut rng, 1 << log_degree, width);

        // Produce the Merkle commitment to the low-degree-extended trace.
        let (commitment, prover_data) =
            <TwoAdicFriPcs<Val, Radix2Dit<Val>, ValMmcs, ChallengeMmcs> as Pcs<
                Challenge,
                Challenger,
            >>::commit(&pcs, [(domain, trace)]);

        // Prover side:
        // Observe the commitment, sample an opening point, and produce the FRI proof.
        let mut p_challenger = Challenger::new(perm.clone());
        p_challenger.observe(&commitment);
        let zeta: Challenge = p_challenger.sample_algebra_element();
        let (opened_values, proof) =
            pcs.open(vec![(&prover_data, vec![vec![zeta]])], &mut p_challenger);

        // Verifier side:
        // Replay the transcript up to the point where the top-level FRI verification begins.
        let mut v_challenger = Challenger::new(perm);
        v_challenger.observe(&commitment);
        let v_zeta: Challenge = v_challenger.sample_algebra_element();
        assert_eq!(
            v_zeta, zeta,
            "prover and verifier must sample the same point"
        );

        // Assemble the commitment-with-opening-points structure that the
        // verifier checks the proof against.
        let cwop = vec![(
            commitment,
            vec![(domain, vec![(zeta, opened_values[0][0][0].clone())])],
        )];

        // Feed the opened evaluations into the verifier challenger.
        // This is the last transcript step before FRI verification begins.
        for (_, round) in &cwop {
            for (_, mat) in round {
                for (_, point) in mat {
                    v_challenger.observe_algebra_slice(point);
                }
            }
        }

        TestFixture {
            fri_params,
            input_mmcs,
            proof,
            challenger: v_challenger,
            commitments_with_opening_points: cwop,
        }
    }

    /// Convenience wrapper that constructs the folding strategy and
    /// invokes the top-level FRI verification.
    fn run_verify_fri(
        params: &FriParameters<ChallengeMmcs>,
        proof: &Proof,
        challenger: &mut Challenger,
        cwop: &[CommitmentWithOpeningPoints<
            Challenge,
            <ValMmcs as Mmcs<Val>>::Commitment,
            TwoAdicMultiplicativeCoset<Val>,
        >],
        input_mmcs: &ValMmcs,
    ) -> Result<(), TestError> {
        let folding: Folding = TwoAdicFriFolding(PhantomData);
        verify_fri(&folding, params, proof, challenger, cwop, input_mmcs)
    }

    #[test]
    fn valid_proof_passes() {
        // Baseline: an unmodified proof must pass all checks.
        let f = make_test_fixture();
        let mut challenger = f.challenger.clone();
        let result = run_verify_fri(
            &f.fri_params,
            &f.proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        );
        assert!(result.is_ok(), "valid proof should pass: {result:?}");
    }

    #[test]
    fn query_commit_phase_openings_count_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // In FRI, each COMMIT round produces one oracle f^(i). During the
        // QUERY phase, the verifier opens that oracle at the queried index.
        // So each query proof must carry exactly one opening per round.
        //
        // Fixture state: 3 rounds → 3 commitments → expect 3 openings.
        //
        // Mutation: append a duplicate opening to query 0.
        //
        //     query 0 openings:  [round_0, round_1, round_2, EXTRA]
        //     commitments:       [round_0, round_1, round_2]
        //     → 4 != 3 → error on query 0
        let extra = proof.query_proofs[0].commit_phase_openings[0].clone();
        proof.query_proofs[0].commit_phase_openings.push(extra);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        )
        .expect_err("should reject mismatched commit-phase opening count");

        let expected_rounds = f.proof.commit_phase_commits.len();
        match err {
            FriError::QueryCommitPhaseOpeningsCountMismatch {
                query,
                expected,
                got,
            } => {
                assert_eq!(query, 0);
                assert_eq!(expected, expected_rounds);
                assert_eq!(got, expected_rounds + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn query_log_arities_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // The folding schedule is the sequence of log-arities that
        // controls how much the domain shrinks each round. In this fixture,
        // binary folding means every round has log_arity = 1 (halving).
        // This schedule must be identical across all query proofs — it is
        // a protocol-wide constant, not a per-query choice.
        //
        // The verifier extracts the schedule from query 0 and checks all
        // others against it.
        //
        // Mutation: bump the first round's log_arity in query 1 from 1 to 2.
        //
        //     query 0 schedule: [1, 1, 1]   ← reference
        //     query 1 schedule: [2, 1, 1]   ← corrupted
        //     → mismatch detected at query 1

        // Capture the uncorrupted reference schedule from query 0.
        let expected_arities: Vec<usize> = proof.query_proofs[0]
            .commit_phase_openings
            .iter()
            .map(|o| o.log_arity as usize)
            .collect();

        // Corrupt query 1's first round.
        proof.query_proofs[1].commit_phase_openings[0].log_arity += 1;

        // Build what the corrupted schedule looks like for the assertion.
        let mut corrupted_arities = expected_arities.clone();
        corrupted_arities[0] += 1;

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        )
        .expect_err("should reject inconsistent arity schedule across queries");

        match err {
            FriError::QueryLogAritiesMismatch {
                query,
                expected,
                got,
            } => {
                assert_eq!(query, 1);
                assert_eq!(expected, expected_arities);
                assert_eq!(got, corrupted_arities);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn commit_pow_witness_count_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Each COMMIT round includes a proof-of-work grinding step: the
        // prover finds a hash preimage satisfying a difficulty target.
        // There must be exactly one witness per round — one per commitment.
        //
        // Fixture state: 3 rounds → 3 commitments → expect 3 witnesses.
        //
        // Mutation: push a dummy witness → 4 witnesses vs 3 commitments.
        proof.commit_pow_witnesses.push(Val::ZERO);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        )
        .expect_err("should reject proof with extra PoW witness");

        let expected_count = f.proof.commit_phase_commits.len();
        match err {
            FriError::CommitPowWitnessCountMismatch { expected, got } => {
                assert_eq!(expected, expected_count);
                assert_eq!(got, expected_count + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn final_poly_length_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // After all COMMIT rounds, f^(r) should be a polynomial of degree < 2^log_final_poly_len.
        // The prover sends its coefficients.
        // In this fixture, log_final_poly_len = 0 → exactly 1 coefficient.
        //
        // Mutation: append a zero coefficient → 2 coefficients vs 1 expected.
        proof.final_poly.push(Challenge::ZERO);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        )
        .expect_err("should reject proof with wrong final polynomial length");

        let expected_len = f.fri_params.final_poly_len();
        match err {
            FriError::FinalPolyLengthMismatch { expected, got } => {
                assert_eq!(expected, expected_len);
                assert_eq!(got, expected_len + 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn query_proof_count_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // The number of queries determines soundness: each independent
        // query multiplies the cheating prover's probability of escaping
        // detection. The protocol parameters fix the exact query count.
        //
        // Fixture state: num_queries = 2 → expect 2 query proofs.
        //
        // Mutation: pop one query proof → 1 vs 2 expected.
        proof.query_proofs.pop();

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        )
        .expect_err("should reject proof with missing query proof");

        match err {
            FriError::QueryProofCountMismatch { expected, got } => {
                assert_eq!(expected, f.fri_params.num_queries);
                assert_eq!(got, f.fri_params.num_queries - 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn missing_initial_reduced_opening() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // The QUERY phase fold chain needs a seed: the combined evaluation
        // of all input polynomials at the queried index. It must live at
        // the maximum domain height. No committed polynomials → no seed.
        //
        // Fixture state: 1 committed polynomial → 1 seed expected.
        //
        // Mutation: clear input proofs + external commitment data → no seed.
        //
        //     input proofs:      []   (was [batch_0])
        //     commitment data:   []   (was [(commit, mats)])
        //     → reduced openings = [] → fold chain has no starting value
        for qp in &mut proof.query_proofs {
            qp.input_proof = vec![];
        }
        let empty_cwop: Vec<
            CommitmentWithOpeningPoints<
                Challenge,
                <ValMmcs as Mmcs<Val>>::Commitment,
                TwoAdicMultiplicativeCoset<Val>,
            >,
        > = vec![];

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &empty_cwop,
            &f.input_mmcs,
        )
        .expect_err("should reject proof with no committed polynomials");

        match err {
            FriError::MissingInitialReducedOpening { .. } => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn sibling_values_length_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // In each fold round the verifier reads the queried evaluation
        // plus (arity - 1) siblings. Binary folding → arity 2 → 1 sibling.
        // Siblings are not in the Fiat-Shamir transcript → safe to mutate.
        //
        // Fixture state: binary fold → expect 1 sibling per round.
        //
        // Mutation: push an extra sibling into round 0 of query 0.
        //
        //     Before: sibling_values = [s0]          (length 1 = arity - 1)
        //     After:  sibling_values = [s0, ZERO]    (length 2 = arity)
        proof.query_proofs[0].commit_phase_openings[0]
            .sibling_values
            .push(Challenge::ZERO);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        )
        .expect_err("should reject proof with wrong number of sibling values");

        match err {
            FriError::SiblingValuesLengthMismatch {
                round,
                expected,
                got,
            } => {
                // Binary fold: arity = 2, so expect 1 sibling, got 2.
                assert_eq!(round, 0);
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn input_proof_batch_count_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Each batch of committed polynomials needs exactly one Merkle
        // batch-opening proof. This check fires before any cryptographic work.
        //
        // Fixture state: 1 batch commitment → expect 1 batch opening.
        //
        // Mutation: push an extra empty batch opening.
        //
        //     batch openings:  [batch_0, EXTRA]
        //     commitments:     [commit_0]
        //     → 2 != 1 → error
        let extra_batch = BatchOpening {
            opened_values: vec![],
            opening_proof: proof.query_proofs[0].input_proof[0].opening_proof.clone(),
        };
        proof.query_proofs[0].input_proof.push(extra_batch);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        )
        .expect_err("should reject proof with extra input batch");

        match err {
            FriError::InputProofBatchCountMismatch { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn batch_opened_values_count_mismatch() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Each committed matrix needs exactly one row of opened column
        // values. This check fires before Merkle verification.
        //
        // Fixture state: 1 matrix per batch → expect 1 opened-values entry.
        //
        // Mutation: pop the only opened-values entry.
        //
        //     opened_values:  []           (was [row_0])
        //     matrices:       [matrix_0]
        //     → 0 != 1 → error
        proof.query_proofs[0].input_proof[0].opened_values.pop();

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        )
        .expect_err("should reject proof with missing opened-values entry");

        match err {
            FriError::BatchOpenedValuesCountMismatch {
                batch,
                expected,
                got,
            } => {
                assert_eq!(batch, 0);
                assert_eq!(expected, 1);
                assert_eq!(got, 0);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn point_evaluation_count_mismatch() {
        let f = make_test_fixture();
        let mut cwop = f.commitments_with_opening_points.clone();

        // The verifier pairs opened column values f_i(x) with claimed
        // evaluations f_i(z) to form (f_i(z) - f_i(x)) / (z - x).
        // These two vectors must have the same length.
        //
        // Fixture state: 2 trace columns → 2 opened values, 2 claims.
        //
        // Mutation: push an extra claim. Only touches verifier-side data,
        // not the proof, so Merkle verification still passes. Claims are
        // not in the transcript, so the original challenger is reused.
        //
        //     opened values:  [f_0(x), f_1(x)]             (length 2)
        //     claims:         [f_0(z), f_1(z), EXTRA]      (length 3)
        //     → 2 != 3 → error
        cwop[0].1[0].1[0].1.push(Challenge::ZERO);

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &f.proof,
            &mut challenger,
            &cwop,
            &f.input_mmcs,
        )
        .expect_err("should reject proof with extra claimed evaluation");

        match err {
            FriError::PointEvaluationCountMismatch {
                batch,
                matrix,
                point,
                expected,
                got,
            } => {
                assert_eq!(batch, 0);
                assert_eq!(matrix, 0);
                assert_eq!(point, 0);
                // 2 trace columns opened, but 3 claimed evaluations.
                assert_eq!(expected, 2);
                assert_eq!(got, 3);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn initial_reduced_opening_height_mismatch() {
        let f = make_test_fixture();
        let params = &f.fri_params;
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);

        // The fold chain starts at the global maximum domain height.
        // Its seed (the first reduced opening) must live at that height.
        //
        //     global max height = 5   (domain |L^(0)| = 2^5 = 32)
        //     opening height    = 3   (domain |L^(?)| = 2^3 = 8)
        //     → mismatch: 5 != 3
        let log_global_max_height = 5;
        let wrong_height = 3;
        let reduced_openings: FriOpenings<Challenge> =
            vec![(wrong_height, Challenge::from(Val::from_u8(7)))];

        // Empty fold-data: the error fires before any folding.
        let betas: Vec<Challenge> = vec![];
        let commits: Vec<<ChallengeMmcs as Mmcs<Challenge>>::Commitment> = vec![];
        let openings: Vec<CommitPhaseProofStep<Challenge, ChallengeMmcs>> = vec![];
        let fold_data_iter = betas.iter().zip(commits.iter()).zip(openings.iter());

        let mut start_index = 0;
        let log_final_height = 1;

        let err = verify_query::<TwoAdicFriFolding<(), ()>, Val, Challenge, ChallengeMmcs>(
            &folding,
            params,
            &mut start_index,
            fold_data_iter,
            reduced_openings,
            log_global_max_height,
            log_final_height,
        )
        .expect_err("should reject opening at wrong initial height");

        match err {
            FriError::InitialReducedOpeningHeightMismatch { expected, got } => {
                assert_eq!(expected, log_global_max_height);
                assert_eq!(got, wrong_height);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn final_fold_height_mismatch() {
        let f = make_test_fixture();
        let params = &f.fri_params;
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);

        // After all COMMIT rounds, the domain must have been folded down to
        // exactly the final height. This defense-in-depth check guards
        // against implementation bugs.
        //
        //     global max height = 5   → fold chain starts at 2^5
        //     final height      = 1   → fold chain should end at 2^1
        //     fold rounds       = 0   → no folding happens
        //     → current height = 5 != final height 1 → error
        let log_global_max_height = 5;
        let log_final_height = 1;
        let reduced_openings: FriOpenings<Challenge> =
            vec![(log_global_max_height, Challenge::from(Val::from_u8(42)))];

        // No fold rounds: jump straight to the final-height check.
        let betas: Vec<Challenge> = vec![];
        let commits: Vec<<ChallengeMmcs as Mmcs<Challenge>>::Commitment> = vec![];
        let openings: Vec<CommitPhaseProofStep<Challenge, ChallengeMmcs>> = vec![];
        let fold_data_iter = betas.iter().zip(commits.iter()).zip(openings.iter());

        let mut start_index = 0;

        let err = verify_query::<TwoAdicFriFolding<(), ()>, Val, Challenge, ChallengeMmcs>(
            &folding,
            params,
            &mut start_index,
            fold_data_iter,
            reduced_openings,
            log_global_max_height,
            log_final_height,
        )
        .expect_err("should reject when final height is not reached");

        match err {
            FriError::FinalFoldHeightMismatch { expected, got } => {
                assert_eq!(expected, log_final_height);
                assert_eq!(got, log_global_max_height);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn unconsumed_reduced_openings() {
        let f = make_test_fixture();
        let params = &f.fri_params;
        let folding = TwoAdicFriFolding::<(), ()>(PhantomData);

        // The fold chain rolls in each committed polynomial at the round
        // where the domain shrinks to that polynomial's height. After all
        // rounds, every opening must have been consumed.
        //
        // Fixture state: global max = final = 5 → zero fold rounds needed.
        //
        // Mutation: provide an extra opening at height 3 that no round visits.
        //
        //     reduced_openings = [(height=5, v1), (height=3, v2)]
        //     fold rounds       = 0 → height 3 is never reached
        //     → opening at height 3 left over → error
        let log_global_max_height = 5;
        let log_final_height = 5;
        let reduced_openings: FriOpenings<Challenge> = vec![
            // Seed at the max height — consumed immediately.
            (log_global_max_height, Challenge::from(Val::from_u8(42))),
            // Extra opening at height 3 — never reached, becomes leftover.
            (3, Challenge::from(Val::from_u8(99))),
        ];

        // No fold rounds: the leftover is detected right after the loop.
        let betas: Vec<Challenge> = vec![];
        let commits: Vec<<ChallengeMmcs as Mmcs<Challenge>>::Commitment> = vec![];
        let openings: Vec<CommitPhaseProofStep<Challenge, ChallengeMmcs>> = vec![];
        let fold_data_iter = betas.iter().zip(commits.iter()).zip(openings.iter());

        let mut start_index = 0;

        let err = verify_query::<TwoAdicFriFolding<(), ()>, Val, Challenge, ChallengeMmcs>(
            &folding,
            params,
            &mut start_index,
            fold_data_iter,
            reduced_openings,
            log_global_max_height,
            log_final_height,
        )
        .expect_err("should reject proof with leftover reduced openings");

        match err {
            FriError::UnconsumedReducedOpenings {
                next_log_height,
                remaining,
            } => {
                assert_eq!(next_log_height, 3);
                assert_eq!(remaining, 1);
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn invalid_pow_witness() {
        let f = make_test_fixture();

        // Before each COMMIT round's folding challenge is sampled, the
        // verifier checks a proof-of-work witness against a difficulty
        // target. Witnesses that don't satisfy the target are rejected.
        let mut params = f.fri_params.clone();
        params.commit_proof_of_work_bits = 20;

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &params,
            &f.proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        )
        .expect_err("should reject proof with invalid PoW witness");

        match err {
            FriError::InvalidPowWitness => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn commit_phase_mmcs_error() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Each fold round verifies the Merkle opening proof for the
        // committed oracle f^(i). A corrupted authentication path makes
        // the recomputed root diverge from the commitment.
        //
        // Merkle proofs are NOT in the Fiat-Shamir transcript → safe to
        // mutate without desyncing the challenger.
        proof.query_proofs[0].commit_phase_openings[0].opening_proof[0] = Default::default();

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        )
        .expect_err("should reject proof with corrupted commit-phase Merkle proof");

        match err {
            FriError::CommitPhaseMmcsError(_) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn input_error() {
        let f = make_test_fixture();
        let mut proof = f.proof.clone();

        // Before the fold chain, the verifier checks Merkle proofs for
        // the input polynomials. A corrupted authentication path makes
        // the recomputed root diverge from the input commitment.
        //
        // Input Merkle proofs are NOT in the Fiat-Shamir transcript →
        // safe to mutate without desyncing the challenger.
        proof.query_proofs[0].input_proof[0].opening_proof[0] = Default::default();

        let mut challenger = f.challenger.clone();
        let err = run_verify_fri(
            &f.fri_params,
            &proof,
            &mut challenger,
            &f.commitments_with_opening_points,
            &f.input_mmcs,
        )
        .expect_err("should reject proof with corrupted input Merkle proof");

        match err {
            FriError::InputError(_) => {}
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn final_poly_mismatch() {
        let mut f = make_test_fixture();
        let proof = &mut f.proof;

        // After the fold chain, the verifier evaluates the final polynomial
        // at the folded point (Horner's method) and compares with the fold
        // chain's output. If the coefficients are wrong, the two disagree.
        //
        // The final polynomial is observed into the Fiat-Shamir transcript
        // before queries are sampled, so corrupting it desyncs the challenger
        // and causes Merkle failures before this check is reached. Because
        // this error path cannot be triggered through the full pipeline, we
        // test the underlying Horner evaluation in isolation.

        // Evaluate the honest polynomial at an arbitrary point.
        let x = Val::TWO;
        let honest_eval = proof
            .final_poly
            .iter()
            .rev()
            .fold(Challenge::ZERO, |acc, &c| acc * x + c);

        // Corrupt the constant coefficient.
        proof.final_poly[0] += Challenge::ONE;

        // Evaluate the corrupted polynomial at the same point.
        let corrupted_eval = proof
            .final_poly
            .iter()
            .rev()
            .fold(Challenge::ZERO, |acc, &c| acc * x + c);

        // The two must differ — this is what the verifier would catch.
        assert_ne!(honest_eval, corrupted_eval);
    }
}
