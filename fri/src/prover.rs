use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{fold, CommitPhaseProofStep, FriConfig, FriProof, QueryProof};

#[instrument(name = "FRI prover", skip_all)]
pub fn prove<F, EF, M, Challenger>(
    config: &FriConfig<M>,
    input: &[Option<Vec<EF>>; 32],
    challenger: &mut Challenger,
) -> (FriProof<EF, M, Challenger::Witness>, Vec<usize>)
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    Challenger: GrindingChallenger + CanObserve<M::Commitment> + FieldChallenger<F>,
{
    // For now, the `Some` inputs must all come from polynomials whose log-degree is a multiple of
    // `config.log_arity`.
    assert!(input.iter().all(|x| x.is_none()
        || (log2_strict_usize(x.as_ref().unwrap().len()) - config.log_blowup) % config.log_arity
            == 0));

    let log_max_height = input.iter().rposition(Option::is_some).unwrap();
    println!("Prover log_max_height: {}", log_max_height);

    // let normalize_phase_result = normalize_phase(config, input, log_max_height, challenger);

    let commit_phase_result = commit_phase(config, input, log_max_height, challenger);

    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    println!("Prover query_indices: {:?}", query_indices);

    let query_proofs = info_span!("query phase").in_scope(|| {
        query_indices
            .iter()
            .map(|&index| answer_query(config, &commit_phase_result.data, index))
            .collect()
    });

    println!(
        "Prover commit_phase_commits: {:?}",
        commit_phase_result.commits.len()
    );

    (
        FriProof {
            commit_phase_commits: commit_phase_result.commits,
            query_proofs,
            final_poly: commit_phase_result.final_poly,
            pow_witness,
        },
        query_indices,
    )
}

fn answer_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::ProverData<RowMajorMatrix<F>>],
    index: usize,
) -> QueryProof<F, M>
where
    F: Field,
    M: Mmcs<F>,
{
    let log_arity = config.log_arity;
    let commit_phase_openings = commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let index_i = index >> (i * log_arity);
            let mask = (1 << log_arity) - 1;
            let folded_index = index_i >> log_arity;
            let index_self = index_i & mask;
            // let index_pair = index_i >> 1;

            let (mut opened_rows, opening_proof) = config.mmcs.open_batch(folded_index, commit);
            println!("Folded index: {}", folded_index);
            assert_eq!(opened_rows.len(), 1);
            let opened_row = opened_rows.pop().unwrap();
            assert_eq!(
                opened_row.len(),
                1 << log_arity,
                "Committed data should be in tuples of size arity."
            );
            // println!("Opened row: {:?}", opened_row);

            // let tmp = opened_row.remove(index_self);
            // println!("Eval for prover: {}", tmp);
            let siblings = opened_row;

            println!("Opening at index: {}", index_i);
            println!("Folded index: {}", folded_index);
            println!("The index-self: {}", index_self);

            CommitPhaseProofStep {
                siblings,
                opening_proof,
            }
        })
        .collect();

    QueryProof {
        commit_phase_openings,
    }
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<F, EF, M, Challenger>(
    config: &FriConfig<M>,
    input: &[Option<Vec<EF>>; 32],
    log_max_height: usize,
    challenger: &mut Challenger,
) -> CommitPhaseResult<EF, M>
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    Challenger: CanObserve<M::Commitment> + FieldChallenger<F>,
{
    let mut current = input[log_max_height].as_ref().unwrap().clone();

    let mut commits = vec![];
    let mut data = vec![];
    let mut phase_counter = 0;

    for log_folded_height in (config.log_blowup..(log_max_height + 1 - config.log_arity))
        .rev()
        .step_by(config.log_arity)
    {
        phase_counter += 1;
        println!("Commit phase log_folded_height: {}", log_folded_height);
        let leaves = RowMajorMatrix::new(current.clone(), 1 << config.log_arity);
        println!("Dimensions: {:?}, {}", leaves.width(), leaves.height());
        let (commit, prover_data) = config.mmcs.commit_matrix(leaves);
        challenger.observe(commit.clone());
        commits.push(commit);
        data.push(prover_data);

        let beta: EF = challenger.sample_ext_element();
        current = fold(current, beta, config.log_arity);

        if let Some(v) = &input[log_folded_height] {
            current.iter_mut().zip_eq(v).for_each(|(c, v)| *c += *v);
        }
    }

    println!("Did {} commit phase steps", phase_counter);

    println!("Current: {:?}", current);

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(current.len(), config.blowup());

    let final_poly = current[0];

    println!("Final poly: {}", final_poly);
    for x in current {
        assert_eq!(x, final_poly);
    }
    challenger.observe_ext_element(final_poly);

    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    final_poly: F,
}
