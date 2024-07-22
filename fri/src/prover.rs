use alloc::vec;
use alloc::vec::Vec;
use core::iter;
use p3_util::{split_bits, SliceExt, VecExt};
use std::collections::VecDeque;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use tracing::{info_span, instrument};

use crate::{
    Codeword, CommitPhaseProofStep, FoldableCodeFamily, FriConfig, FriProof, LinearCodeFamily,
    QueryProof,
};

#[instrument(name = "FRI prover", skip_all)]
pub fn prove<Code, Val, Challenge, Challenger, M>(
    config: &FriConfig<M>,
    inputs: Vec<Codeword<Challenge, Code>>,
    challenger: &mut Challenger,
) -> (FriProof<Challenge, M, Challenger::Witness>, Vec<usize>)
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    Code: FoldableCodeFamily<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    M: Mmcs<Challenge>,
{
    assert!(inputs.iter().all(|cw| cw.is_full()));

    // sorted strictly decreasing
    assert!(inputs
        .iter()
        .tuple_windows()
        .all(|(l, r)| l.code.log_word_len() > r.code.log_word_len()));

    let index_bits = inputs[0].code.log_word_len();

    let CommitPhaseResult {
        commits: commit_phase_commits,
        data: commit_phase_data,
        final_polys,
    } = info_span!("commit phase")
        .in_scope(|| commit_phase::<Code, _, _, _, _>(&config, inputs, challenger));

    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let query_indices: Vec<_> = iter::repeat_with(|| challenger.sample_bits(index_bits))
        .take(config.num_queries)
        .collect();

    let query_proofs = info_span!("query phase").in_scope(|| {
        query_indices
            .iter()
            .map(|&index| QueryProof {
                commit_phase_openings: answer_query(config, &commit_phase_data, index),
            })
            .collect()
    });

    (
        FriProof {
            commit_phase_commits,
            query_proofs,
            final_polys,
            pow_witness,
        },
        query_indices,
    )
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    final_polys: Vec<Vec<F>>,
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<Code, Val, Challenge, M, Challenger>(
    config: &FriConfig<M>,
    inputs: Vec<Codeword<Challenge, Code>>,
    challenger: &mut Challenger,
) -> CommitPhaseResult<Challenge, M>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    Code: FoldableCodeFamily<Challenge>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + CanObserve<M::Commitment>,
{
    let mut log_word_len = inputs[0].code.log_word_len();
    let mut inputs = inputs.into_iter().peekable();
    let mut folded: Vec<Codeword<Challenge, Code>> = vec![];

    // let mut commits_and_data = vec![];
    // let mut final_polys = vec![];

    while log_word_len > config.log_max_final_word_len {
        let log_folded_word_len = log_word_len - config.log_folding_arity;

        log_word_len = log_folded_word_len;
    }

    while let Some(log_word_len) = inputs
        .peek()
        .map(|cw| cw.code.log_word_len())
        .filter(|&l| l > config.log_max_final_word_len)
    {
        let log_folded_word_len = log_word_len - config.log_folding_arity;
    }

    todo!()

    /*
    for log_word_len in step_log_word_lens {
        folded.extend(inputs.peeking_take_while(|cw| cw.code.log_word_len() > log_word_len));

        let mats: Vec<_> = folded
            .iter()
            .map(|cw| {
                RowMajorMatrix::new(
                    cw.word.clone(),
                    1 << (cw.code.log_word_len() - log_word_len),
                )
            })
            .collect();

        let (commit, data) = mmcs.commit(mats);
        challenger.observe(commit.clone());
        commits_and_data.push((commit, data));

        let beta: Challenge = challenger.sample_ext_element();

        for cw in &mut folded {
            cw.fold_to_log_word_len(log_word_len, beta);
        }

        Codeword::sum_words_from_same_code(&mut folded);
    }

    assert_eq!(folded.len(), 1);

    let final_poly = folded.pop().unwrap().decode();
    challenger.observe_ext_element_slice(&final_poly);

    let (commits, data) = commits_and_data.into_iter().unzip();
    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
    */
}

fn answer_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_data: &[M::ProverData<RowMajorMatrix<F>>],
    mut index: usize,
) -> Vec<CommitPhaseProofStep<F, M>>
where
    F: Field,
    M: Mmcs<F>,
{
    let mut steps = vec![];
    for data in commit_phase_data {
        let (folded_index, index_in_subgroup) = split_bits(index, config.log_folding_arity);
        let (mut siblings, proof) = config.mmcs.open_batch(folded_index, data);
        for sibs in &mut siblings {
            sibs.remove(index_in_subgroup >> (config.log_folding_arity - sibs.log_strict_len()));
        }
        steps.push(CommitPhaseProofStep { siblings, proof });
        index = folded_index;
    }
    steps
}
