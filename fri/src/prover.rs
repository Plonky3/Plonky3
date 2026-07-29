use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};
use tracing::{debug_span, info_span, instrument};

use crate::{
    BatchMultiOpening, CommitPhaseMultiStep, FriFoldingStrategy, FriParameters, FriProof,
    ProverDataWithOpeningPoints, compute_log_arity_for_round,
};

/// Create a proof that an opening `f(zeta)` is correct by proving that the
/// function `(f(x) - f(zeta))/(x - zeta)` is low degree.
///
/// This further supports proving a batch of these claims for a collection of polynomials of shrinking degrees.
/// Polynomials of equal degree can be combined using randomness before calling this function.
///
/// The Soundness error from prove_fri comes from the paper:
/// Proximity Gaps for Reed-Solomon Codes (<https://eprint.iacr.org/2020/654>)
/// and is either `rate^{num_queries}` or `rate^{num_queries/2}` depending on if you rely on conjectured or
/// proven soundness. Particularly safety conscious users may want to set `num_queries` slightly higher than
/// this to account for the fact that most implementations batch inputs using a single random challenge
/// instead of one challenge for each polynomial and due to the birthday paradox,
/// there is a non trivial chance that two queried indices will be equal.
///
/// Arguments:
/// - `folding`: The FRI folding scheme to use.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `inputs`: The evaluation vectors of all polynomials we are applying FRI to. The function assumes that
///   commitments to these vectors have been produced and observed by the challenger earlier in the protocol.
/// - `challenger`: The Fiat-Shamir challenger to use for sampling challenges.
/// - `log_global_max_height`: The log of the maximum height of the input matrices.
/// - `prover_data_with_opening_points`: A list of pairs of a batch commitment to a collection
///   of matrices and a list of points to open those matrices at.
#[instrument(name = "FRI prover", skip_all)]
pub fn prove_fri<Folding, Val, Challenge, InputMmcs, FriMmcs, Challenger>(
    folding: &Folding,
    params: &FriParameters<FriMmcs>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
    log_global_max_height: usize,
    prover_data_with_opening_points: &[ProverDataWithOpeningPoints<
        '_,
        Challenge,
        InputMmcs::ProverData<RowMajorMatrix<Val>>,
    >],
    input_mmcs: &InputMmcs,
) -> FriProof<Challenge, FriMmcs, Challenger::Witness, Folding::InputProof>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<FriMmcs::Commitment>,
    Folding:
        FriFoldingStrategy<Val, Challenge, InputProof = Vec<BatchMultiOpening<Val, InputMmcs>>>,
{
    assert!(!inputs.is_empty());
    assert!(
        params.num_queries > 0,
        "num_queries must be at least 1 for FRI soundness"
    );
    assert!(
        params.max_log_arity > 0,
        "max_log_arity must be at least 1 to guarantee folding progress"
    );
    assert!(
        inputs
            .iter()
            .tuple_windows()
            .all(|(l, r)| l.len() >= r.len()),
        "Inputs are not sorted in descending order of length."
    );

    // Index sampling and `open_input` must agree on the height; the caller's value is canonical.
    assert_eq!(
        log_global_max_height,
        log2_strict_usize(inputs[0].len()),
        "log_global_max_height must match the largest input length"
    );
    let log_min_height = log2_strict_usize(inputs.last().unwrap().len());
    if params.log_final_poly_len > 0 {
        // Final_poly_degree must be less than or equal to the degree of the smallest polynomial.
        assert!(log_min_height > params.log_final_poly_len + params.log_blowup);
    }

    // Continually fold the inputs down until the polynomial degree reaches final_poly_degree.
    // Returns a vector of commitments to the intermediate stage polynomials, the intermediate stage polynomials
    // themselves and the final polynomial.
    // Note that the challenger observes the commitments and the final polynomial inside this function so we don't
    // need to observe the output of this function here.
    let commit_phase_result = commit_phase(folding, params, inputs, challenger);

    // Bind the chosen folding arities into the transcript.
    for &log_arity in &commit_phase_result.log_arities {
        challenger.observe(Val::from_usize(log_arity));
    }

    // Produce a proof of work witness before receiving any query challenges.
    // This helps to prevent grinding attacks.
    let pow_witness = challenger.grind(params.query_proof_of_work_bits);

    // Sample num_queries indexes to check.
    // The probability that no two FRI indices are equal (ignoring extra query index bits) is:
    // (Grabbed this from wikipedia page on the birthday problem)
    // N!/(N^{num_queries} * (N - num_queries)!) ~ (1 - 1/N)^{num_queries * (num_queries - 1)/2}
    //                                           ~ (1 - num_queries^2/2N)
    // Here N = 2^log_global_max_height.
    // With num_queries = 100, N = 2^20, this is 0.995 so there is a .5% chance of a collision.
    // Due to this, security conscious users may want to set num_queries a little higher than the
    // theoretical minimum.
    //
    // Sampling all indices in one block leaves the transcript identical to sampling
    // them one query at a time: nothing is observed between samples.
    let indices: Vec<usize> = iter::repeat_with(|| {
        challenger.sample_bits(log_global_max_height + folding.extra_query_index_bits())
    })
    .take(params.num_queries)
    .collect();

    let (input_openings, commit_phase_openings) = info_span!("query phase").in_scope(|| {
        // Openings of the inputs and of every commit-phase codeword at all
        // queried locations. Queries into one tree share a single proof, so
        // overlapping authentication paths are deduplicated instead of being
        // shipped once per query.
        let input_openings = open_inputs(
            log_global_max_height,
            &indices,
            prover_data_with_opening_points,
            input_mmcs,
        );
        let domain_indices: Vec<usize> = indices
            .iter()
            .map(|&index| index >> folding.extra_query_index_bits())
            .collect();
        let commit_phase_openings = answer_queries(
            params,
            &commit_phase_result.log_arities,
            &commit_phase_result.data,
            &domain_indices,
        );
        (input_openings, commit_phase_openings)
    });

    FriProof {
        commit_phase_commits: commit_phase_result.commits,
        commit_pow_witnesses: commit_phase_result.pow_witnesses,
        input_openings,
        commit_phase_openings,
        final_poly: commit_phase_result.final_poly,
        query_pow_witness: pow_witness,
    }
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>, Witness> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    log_arities: Vec<usize>,
    pow_witnesses: Vec<Witness>,
    final_poly: Vec<F>,
}

/// Perform the commit phase of the FRI protocol.
///
/// In each round we reduce our evaluations over `H` to evaluations over `H^k` (where k is the arity)
/// by folding with a random challenge. For instance, for arity 2, we have:
/// ```text
///     f_{i + 1}(x^2) = (f_i(x) + f_i(-x))/2 + beta_i (f_i(x) - f_i(-x))/2x
/// ```
/// We then commit to the evaluation vector over the smaller domain, i.e. `f_{i + 1}` over `H^2` for arity 2.
///
/// The arity for each round is dynamically computed to ensure we always commit at each input height level.
///
/// Once the degree of our polynomial falls below `final_poly_degree`, we compute the coefficients of our
/// polynomial and return them along with all intermediate evaluations and corresponding commitments.
///
/// Arguments:
/// - `folding`: The FRI folding scheme used by the prover.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `inputs`: The evaluation vectors of the polynomials. These must be sorted in descending order of length and each
///   evaluation vector must be in bit reversed order. This function assumes that commitments to these vectors
///   have already been produced and observed by the challenger.
/// - `challenger`: The Fiat-Shamir challenger to use for sampling challenges.
#[instrument(name = "commit phase", skip_all)]
fn commit_phase<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
) -> CommitPhaseResult<Challenge, M, <Challenger as GrindingChallenger>::Witness>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    assert!(
        params.max_log_arity > 0,
        "max_log_arity must be at least 1 to guarantee folding progress"
    );

    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = vec![];
    let mut data = vec![];
    let mut log_arities = vec![];
    let mut pow_witnesses = vec![];

    let log_final_height = params.log_blowup + params.log_final_poly_len;

    while folded.len() > params.blowup() * params.final_poly_len() {
        let log_current_height = log2_strict_usize(folded.len());
        let next_input_log_height = inputs_iter.peek().map(|v| log2_strict_usize(v.len()));

        //Compute the arity for this round
        let log_arity = compute_log_arity_for_round(
            log_current_height,
            next_input_log_height,
            log_final_height,
            params.max_log_arity,
        );
        let arity = 1 << log_arity;
        log_arities.push(log_arity);

        // As folded is in bit reversed order, the evaluations at conjugate points are adjacent.
        // We reinterpret the vector as a matrix of width `arity`.
        let leaves = RowMajorMatrix::new(folded, arity);

        // Commit to these evaluations and observe the commitment.
        let (commit, prover_data) = params.mmcs.commit_matrix(leaves);
        challenger.observe(commit.clone());
        commits.push(commit);

        // Produce a proof of work witness after observing the commitment and
        // before the Fiat-Shamir batching challenge.
        let pow_witness = challenger.grind(params.commit_proof_of_work_bits);
        pow_witnesses.push(pow_witness);

        // Get the Fiat-Shamir challenge for this round.
        let beta: Challenge = challenger.sample_algebra_element();

        // We passed ownership of `leaves` to the MMCS, so get a reference to it
        let leaves = params.mmcs.get_matrices(&prover_data).pop().unwrap();
        // Do the folding operation with the computed arity
        folded = folding.fold_matrix(beta, log_arity, leaves.as_view());

        data.push(prover_data);

        // If we have reached the size of the next input vector, we can add it to the current vector.
        if let Some(v) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            // Each element of `inputs_iter` is a reduced opening polynomial, which is itself a
            // random linear combination `f_{i, 0} + alpha f_{i, 1} + ...`, when we add it
            // to the current folded polynomial, we need to multiply by a random factor.
            // We use beta^arity as the random factor to maintain independence.
            let beta_pow = beta.exp_power_of_2(log_arity);
            izip!(&mut folded, v).for_each(|(c, x)| *c += beta_pow * x);
        }
    }

    // Now we need to get the coefficients of the final polynomial. As we know that the degree
    // is `<= params.final_poly_len()` and the evaluations are stored in bit-reversed order,
    // we can just truncate the folded vector, bit-reverse again and run an IDFT.
    folded.truncate(params.final_poly_len());
    reverse_slice_index_bits(&mut folded);
    let final_poly = debug_span!("idft final poly")
        .in_scope(|| Radix2DFTSmallBatch::default().idft_algebra(folded));

    // Observe all coefficients of the final polynomial.
    challenger.observe_algebra_slice(&final_poly);

    CommitPhaseResult {
        commits,
        data,
        log_arities,
        pow_witnesses,
        final_poly,
    }
}

/// Given the query indices, produce a proof that every chain of folds is correct.
/// This is the prover's complement to the verifier's fold-and-verify pass.
///
/// In addition to the output of this function, the prover must also supply the verifier with the input values
/// (with associated opening proofs). These are produced by the `open_inputs` function passed into `prove_fri`.
///
/// For each round, this returns the sibling values of every query (all values in the group
/// except the queried one) along with one shared opening proof covering all queries.
/// The verifier can then reconstruct each full group, verify the round's openings together,
/// and fold to get the value at the parent index.
///
/// With variable arity, the index shifts by `log_arities[i]` each round instead of always by 1
/// (i.e. when arity is fixed to 2).
///
/// Arguments:
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `log_arities`: The log2 of the arity used for each round.
/// - `folded_polynomial_commits`: A slice of commitments to the intermediate stage polynomials.
/// - `start_indices`: The opening indices for the unfolded polynomial, one per query.
#[inline]
fn answer_queries<F, M>(
    config: &FriParameters<M>,
    log_arities: &[usize],
    folded_polynomial_commits: &[M::ProverData<RowMajorMatrix<F>>],
    start_indices: &[usize],
) -> Vec<CommitPhaseMultiStep<F, M>>
where
    F: Field,
    M: Mmcs<F>,
{
    let mut current_indices = start_indices.to_vec();

    folded_polynomial_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let log_arity = log_arities[i];
            let arity = 1 << log_arity;

            // Index of each query's group (row in the committed matrix).
            let group_indices: Vec<usize> = current_indices
                .iter()
                .map(|&index| index >> log_arity)
                .collect();

            // One shared proof that all queried groups are correct.
            let (opened_rows, opening_proof) = config.mmcs.open_multi_batch(&group_indices, commit);

            // For each query, keep all siblings (exclude the queried element).
            let sibling_values: Vec<Vec<F>> = izip!(&current_indices, opened_rows)
                .map(|(&index, mut rows)| {
                    // Each group opening should contain just the values in this group.
                    assert_eq!(rows.len(), 1);
                    let opened_row = rows.pop().unwrap();
                    assert_eq!(
                        opened_row.len(),
                        arity,
                        "Committed data should have arity {} elements",
                        arity
                    );
                    let index_in_group = index % arity;
                    opened_row
                        .into_iter()
                        .enumerate()
                        .filter(|(j, _)| *j != index_in_group)
                        .map(|(_, v)| v)
                        .collect()
                })
                .collect();

            current_indices = group_indices;

            // Add the siblings and the shared proof to the vector.
            CommitPhaseMultiStep {
                log_arity: log_arity as u8,
                sibling_values,
                opening_proof,
            }
        })
        .collect()
}

/// Given the query indices, produce one batch multi-opening proof for each collection
/// of matrices combined into a single mmcs commitment.
///
/// In cases where the maximum height of a batch of matrices is smaller than the
/// global max height, shift the indices down to compensate.
///
/// Arguments:
/// - `log_global_max_height`: The log of the maximum height of the input matrices.
/// - `indices`: The indices to open the matrices at, one per query.
/// - `prover_data_with_opening_points`: A list of pairs of a batch commitment to a collection
///   of matrices and a list of points to open those matrices at.
/// - `mmcs`: The mixed matrix commitment scheme used to produce the batch commitments.
#[inline]
fn open_inputs<Val, Challenge, InputMmcs>(
    log_global_max_height: usize,
    indices: &[usize],
    prover_data_with_opening_points: &[ProverDataWithOpeningPoints<
        '_,
        Challenge,
        InputMmcs::ProverData<RowMajorMatrix<Val>>,
    >],
    mmcs: &InputMmcs,
) -> Vec<BatchMultiOpening<Val, InputMmcs>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
{
    // This gives the verifier access to evaluations `f(x)` from which it can compute
    // `(f(zeta) - f(x))/(zeta - x)` and then combine them together and roll into FRI
    // as appropriate.
    prover_data_with_opening_points
        .iter()
        .map(|(data, _)| {
            let log_max_height = log2_strict_usize(mmcs.get_max_height(data));
            let bits_reduced = log_global_max_height - log_max_height;
            // If a matrix is smaller than global max height, we roll it into
            // fri in a later round.
            let reduced_indices: Vec<usize> =
                indices.iter().map(|&index| index >> bits_reduced).collect();
            let (opened_values, opening_proof) = mmcs.open_multi_batch(&reduced_indices, data);
            BatchMultiOpening {
                opened_values,
                opening_proof,
            }
        })
        .collect()
}
