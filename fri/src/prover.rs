use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_util::{log2_strict_usize, reverse_slice_index_bits};
use tracing::{debug_span, info_span, instrument};

use crate::{CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof, QueryProof};

#[instrument(name = "FRI prover", skip_all)]
pub fn prove<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
    open_input: impl Fn(usize) -> G::InputProof,
) -> FriProof<Challenge, M, Challenger::Witness, G::InputProof>
where
    Val: Field,
    Challenge: ExtensionField<Val> + TwoAdicField,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
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
    if config.log_final_poly_len > 0 {
        assert!(log_min_height > config.log_final_poly_len + config.log_blowup);
    }

    let commit_phase_result = commit_phase(g, config, inputs, challenger);

    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let query_proofs = info_span!("query phase").in_scope(|| {
        iter::repeat_with(|| challenger.sample_bits(log_max_height + g.extra_query_index_bits()))
            .take(config.num_queries)
            .map(|index| QueryProof {
                input_proof: open_input(index),
                commit_phase_openings: answer_query(
                    config,
                    &commit_phase_result.data,
                    index >> g.extra_query_index_bits(),
                ),
            })
            .collect()
    });

    FriProof {
        commit_phase_commits: commit_phase_result.commits,
        query_proofs,
        final_poly: commit_phase_result.final_poly,
        log_max_height,
        pow_witness,
    }
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    final_poly: Vec<F>,
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
) -> CommitPhaseResult<Challenge, M>
where
    Val: Field,
    Challenge: ExtensionField<Val> + TwoAdicField,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    // To illustrate the folding logic with arity > 2, let's go through an example.
    // Suppose `inputs` consists of three polynomials with degrees 16, 8, and 4, and
    // suppose that arity = 4 and final_poly_len = 1.
    // There will be two FRI commitment layers: one at height 16 and one at height 4.

    // The first commitment layer will consist of two matrices:
    // - one of dimensions 4x4 corresponding to the first polynomial's evaluations
    // - one of dimensions 4x2 corresponding to the second polynomial's evaluations

    // The polynomial folding happens incrementally as follows: the first polynomial is folded
    // once so its number of evaluations is halved, the second polynomial's evaluations are added
    // to that, and then the sum is folded by 2 further to reduce the number of evaluations to 4.

    // At that point, the third polynomial's evaluations are added to the running sum, and that sum
    // is committed to form the second FRI commitment layer. The only matrix in that layer is of
    // dimensions 4x1. The final polynomial's evaluation can then be computed through those 4
    // evaluations.

    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = vec![];
    let mut data = vec![];

    let arity = config.arity();

    while folded.len() > config.blowup() * config.final_poly_len() {
        let cur_arity = arity.min(folded.len());

        let next_folded_len = folded.len() / cur_arity;

        // First, we collect the polynomial evaluations that will be committed this round.
        // Those are `folded` and polynomials in `inputs` not consumed yet with number of
        // evaluations more than `next_folded_len`
        let mut polys_before_next_round = vec![];

        let mut cur_folded_len = folded.len();
        while cur_folded_len > next_folded_len {
            if let Some(poly_eval) = inputs_iter.next_if(|v| v.len() == cur_folded_len) {
                let width = poly_eval.len() / next_folded_len;
                let poly_eval_matrix = RowMajorMatrix::new(poly_eval, width);
                polys_before_next_round.push(poly_eval_matrix);
            }

            cur_folded_len /= 2;
        }

        let folded_matrix = RowMajorMatrix::new(folded.clone(), cur_arity);
        let matrices_to_commit: Vec<DenseMatrix<Challenge>> = iter::once(folded_matrix)
            .chain(polys_before_next_round)
            .collect();

        let (commit, prover_data) = config.mmcs.commit(matrices_to_commit);
        challenger.observe(commit.clone());

        // Next, we fold `folded` and `polys_before_next_round` to prepare for the next round
        let beta: Challenge = challenger.sample_ext_element();

        // Get a reference to the committed matrices
        let leaves = config.mmcs.get_matrices(&prover_data);
        let mut leaves_iter = leaves.into_iter().peekable();
        // Skip `folded`
        leaves_iter.next();

        while folded.len() > next_folded_len {
            let matrix_to_fold = RowMajorMatrix::new(folded, 2);
            folded = g.fold_matrix(beta, matrix_to_fold);

            if let Some(poly_eval) = leaves_iter.next_if(|v| v.values.len() == folded.len()) {
                izip!(&mut folded, &poly_eval.values).for_each(|(f, v)| *f += *v);
            }
        }

        commits.push(commit);
        data.push(prover_data);

        // We directly add the next polynomial's evaluations into `folded` in case their lengths match
        if let Some(poly_eval) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            izip!(&mut folded, poly_eval).for_each(|(c, x)| *c += x);
        }
    }

    // After repeated folding steps, we end up working over a coset hJ instead of the original
    // domain. The IDFT we apply operates over a subgroup J, not hJ. This means the polynomial we
    // recover is G(x), where G(x) = F(hx), and F is the polynomial whose evaluations we actually
    // observed. For our current construction, this does not cause issues since degree properties
    // and zero-checks remain valid. If we changed our domain construction (e.g., using multiple
    // cosets), we would need to carefully reconsider these assumptions.

    reverse_slice_index_bits(&mut folded);
    // TODO: For better performance, we could run the IDFT on only the first half
    //       (or less, depending on `log_blowup`) of `final_poly`.
    let final_poly = debug_span!("idft final poly").in_scope(|| Radix2Dit::default().idft(folded));

    // The evaluation domain is "blown-up" relative to the polynomial degree of `final_poly`,
    // so all coefficients after the first final_poly_len should be zero.
    debug_assert!(
        final_poly
            .iter()
            .skip(config.final_poly_len())
            .all(|x| x.is_zero()),
        "All coefficients beyond final_poly_len must be zero"
    );

    // Observe all coefficients of the final polynomial.
    for &x in &final_poly {
        challenger.observe_ext_element(x);
    }

    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
}

fn answer_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::ProverData<RowMajorMatrix<F>>],
    index: usize,
) -> Vec<CommitPhaseProofStep<F, M>>
where
    F: Field,
    M: Mmcs<F>,
{
    commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let index_row = index >> ((i + 1) * config.arity_bits);

            let (opened_rows, opening_proof) = config.mmcs.open_batch(index_row, commit);

            CommitPhaseProofStep {
                opened_rows,
                opening_proof,
            }
        })
        .collect()
}
