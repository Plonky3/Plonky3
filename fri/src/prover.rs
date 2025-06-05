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

use crate::{CommitPhaseProofStep, FriFoldingStrategy, FriParameters, FriProof, QueryProof};

/// Create a proof that an opening `f(zeta)` is correct by proving that the
/// function `(f(x) - f(zeta))/(x - zeta)` is low degree.
///
/// This further supports proving a batch of these claims for a collection of polynomials of shrinking degrees.
/// Polynomials of equal degree can be combined using randomness before calling this function.
///
/// The Soundness error from prove_fri comes from the paper:
/// Proximity Gaps for Reed-Solomon Codes (https://eprint.iacr.org/2020/654)
/// and is either `rate^{num_queries}` or `rate^{num_queries/2}` depending on if you rely on conjectured or
/// proven soundness. Particularly safety conscious users may want to set `num_queries` slightly higher than
/// this to account for the fact that most implementations batch inputs using a single random challenge
/// instead of one challenge for each polynomial and due to the birthday paradox,
/// there is a non trivial chance that two queried indices will be equal.
///
/// Arguments:
/// - `folding`: The FRI folding scheme to use.
/// - `params`: The parameters for the specific FRI protocol instance.
/// - `inputs`: The evaluation vectors of the polynomials.
/// - `challenger`: The Fiat-Shamir challenger to use for sampling challenges.
/// - `open_input`: A function that takes an index and produces proofs that the initial values in
///   inputs at that index (Or at `index >> i` for smaller `f`'s) are correct.
#[instrument(name = "FRI prover", skip_all)]
pub fn prove_fri<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
    open_input: impl Fn(usize) -> Folding::InputProof,
) -> FriProof<Challenge, M, Challenger::Witness, Folding::InputProof>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    assert!(!inputs.is_empty());
    assert!(
        inputs
            .iter()
            .tuple_windows()
            .all(|(l, r)| l.len() >= r.len()),
        "Inputs are not sorted in descending order of length."
    );

    let log_max_height = log2_strict_usize(inputs[0].len());
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

    // Produce a proof of work witness before receiving any query challenges.
    // This helps to prevent grinding attacks.
    let pow_witness = challenger.grind(params.proof_of_work_bits);

    let query_proofs = info_span!("query phase").in_scope(|| {
        // Sample num_queries indexes to check.
        // The probability that no two FRI indices are equal (ignoring extra query index bits) is:
        // (Grabbed this from wikipedia page on the birthday problem)
        // N!/(N^{num_queries} * (N - num_queries)!) ~ (1 - 1/N)^{num_queries * (num_queries - 1)/2}
        //                                           ~ (1 - num_queries^2/2N)
        // Here N = 2^log_max_height.
        // With num_queries = 100, N = 2^20, this is 0.995 so there is a .5% chance of a collision.
        // Due to this, security conscious users may want to set num_queries a little higher than the
        // theoretical minimum.
        iter::repeat_with(|| {
            let index = challenger.sample_bits(log_max_height + folding.extra_query_index_bits());
            // For each index, create a proof that the folding operations along the chain:
            // round 0: index, round 1: index >> 1, round 2: index >> 2, ... are correct.
            QueryProof {
                input_proof: open_input(index),
                commit_phase_openings: answer_query(
                    params,
                    &commit_phase_result.data,
                    index >> folding.extra_query_index_bits(),
                ),
            }
        })
        .take(params.num_queries)
        .collect()
    });

    FriProof {
        commit_phase_commits: commit_phase_result.commits,
        query_proofs,
        final_poly: commit_phase_result.final_poly,
        pow_witness,
    }
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    final_poly: Vec<F>,
}

/// Perform the commit phase of the FRI protocol.
///
/// In each round we reduce our evaluations over `H` to evaluations over `H^2` by defining
/// ```text
///     f_{i + 1}(x^2) = (f_i(x) + f_i(-x))/2 + beta_i (f_i(x) - f_i(-x))/2x
/// ```
/// We then commit to the evaluation vector of `f_{i + 1}` over `H^2`.
///
/// Once the degree of our polynomial falls below `final_poly_degree`, we compute the coefficients of our
/// polynomial and return it along with all intermediate evaluations and our commitments to them.
///
/// Arguments:
/// - TODO (Once the renaming is done, fix this up)
/// - `config`, `parameters`: Together, these contain all information needed to define the FRI protocol.
///    E.g. the folding scheme, the code rate, the final polynomial size.
/// - `inputs`: The evaluation vectors of the polynomials. These must be sorted in descending order of length and each
///   evaluation vector must be in bit reversed order.
/// - `challenger`: The Fiat-Shamir challenger to use for sampling challenges.
#[instrument(name = "commit phase", skip_all)]
fn commit_phase<Folding, Val, Challenge, M, Challenger>(
    folding: &Folding,
    params: &FriParameters<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
) -> CommitPhaseResult<Challenge, M>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + CanObserve<M::Commitment>,
    Folding: FriFoldingStrategy<Val, Challenge>,
{
    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = vec![];
    let mut data = vec![];

    while folded.len() > params.blowup() * params.final_poly_len() {
        // As folded is in bit reversed order, it looks like:
        //      `[f_i(h^0), f_i(h^{N/2}), f_i(h^{N/4}), f_i(h^{3N/4}), ...] = [f_i(1), f_i(-1), f_i(h^{N/4}), f_i(-h^{N/4}), ...]`
        // so the relevant evaluations are adjacent and we can just reinterpret the vector as a matrix of width 2.
        let leaves = RowMajorMatrix::new(folded, 2);

        // Commit to these evaluations and observe the commitment.
        let (commit, prover_data) = params.mmcs.commit_matrix(leaves);
        challenger.observe(commit.clone());
        commits.push(commit);

        // Get the Fiat-Shamir challenge for this round.
        let beta: Challenge = challenger.sample_algebra_element();

        // We passed ownership of `folded` to the MMCS, so get a reference to it
        let leaves = params.mmcs.get_matrices(&prover_data).pop().unwrap();
        // Do the folding operation:
        //      `f_{i + 1}'(x^2) = (f_i(x) + f_i(-x))/2 + beta_i (f_i(x) - f_i(-x))/2x`
        folded = folding.fold_matrix(beta, leaves.as_view());

        data.push(prover_data);

        // If we have reached the size of the next input vector, we can add it to the current vector.
        if let Some(v) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            // Each element of `inputs_iter` is a reduced opening polynomial, which is itself a
            // random linear combination `f_{i, 0} + alpha f_{i, 1} + ...`, but when we add it
            // to the current folded polynomial, we need to multiply by a new random factor since
            // `f_{i, 0}` has no leading coefficient.
            izip!(&mut folded, v).for_each(|(c, x)| *c += beta.square() * x);
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
    for &x in &final_poly {
        challenger.observe_algebra_element(x);
    }

    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
}

/// Given an `index` produce a proof that the chain of folds at `index, index >> 1, ... ` are correct.
/// This is the prover's complement to the verifier's [`verify_query`] function.
///
/// In addition to the output of this function, the prover must also supply the verifier with the input values
/// (with associated opening proofs). These are produced by the `open_input` function passed into `prove_fri`.
///
/// For each `i` in `[0, ..., num_folds)` this returns the value at `(index >> i) ^ 1` in round `i` along with
/// an opening proof. The verifier can then use the values in round `i` at `index >> i` and `(index >> i) ^ 1`
/// along with possibly an input value to compute the value at `index >> (i + 1)` in round `i + 1`.
///
/// We repeat until we reach the final round where the verifier can check the value against the
/// polynomial they were sent.
///
/// Arguments:
/// - `config`: The FRI configuration file containing the user set parameters.
/// - `folded_polynomial_commits`: The commitments to the intermediate stage polynomials.
/// - `start_index`: The opening index for the unfolded polynomial. For folded polynomials
///   we use this this index right shifted by the number of folds.
fn answer_query<F, M>(
    config: &FriParameters<M>,
    folded_polynomial_commits: &[M::ProverData<RowMajorMatrix<F>>],
    start_index: usize,
) -> Vec<CommitPhaseProofStep<F, M>>
where
    F: Field,
    M: Mmcs<F>,
{
    folded_polynomial_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            // After i folding rounds, the current index we are looking at is `index >> i`.
            let index_i = start_index >> i;
            let index_i_sibling = index_i ^ 1;
            let index_pair = index_i >> 1;

            // Get a proof that the pair of indices are correct.
            let (mut opened_rows, opening_proof) =
                config.mmcs.open_batch(index_pair, commit).unpack();

            // opened_rows should contain just the value at index_i and its sibling.
            // We just need to get the sibling.
            assert_eq!(opened_rows.len(), 1);
            let opened_row = &opened_rows.pop().unwrap();
            assert_eq!(opened_row.len(), 2, "Committed data should be in pairs");
            let sibling_value = opened_row[index_i_sibling % 2];

            // Add the sibling and the proof to the vector.
            CommitPhaseProofStep {
                sibling_value,
                opening_proof,
            }
        })
        .collect()
}
