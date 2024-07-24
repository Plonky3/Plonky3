use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;
use core::iter;
use p3_util::{split_bits, SliceExt, VecExt};

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use tracing::{info_span, instrument};

use crate::{Codeword, CommitPhaseProofStep, FoldableCodeFamily, FriConfig, FriProof, QueryProof};

#[instrument(name = "FRI prover", skip_all)]
pub fn prove<Code, Val, Challenge, Challenger, M, InputProof>(
    config: &FriConfig<M>,
    codewords: Vec<Codeword<Challenge, Code>>,
    challenger: &mut Challenger,
    prove_input: impl Fn(usize) -> InputProof,
) -> FriProof<Challenge, M, Challenger::Witness, InputProof>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    Code: FoldableCodeFamily<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    M: Mmcs<Challenge>,
{
    assert!(codewords.iter().all(|cw| cw.is_full()));

    let index_bits = codewords
        .iter()
        .map(|cw| cw.code.log_word_len())
        .max()
        .unwrap();

    let CommitPhaseResult {
        commits: commit_phase_commits,
        data: commit_phase_data,
        final_polys,
    } = info_span!("commit phase").in_scope(|| commit_phase(&config, codewords, challenger));

    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let query_indices: Vec<_> = iter::repeat_with(|| challenger.sample_bits(index_bits))
        .take(config.num_queries)
        .collect();

    let query_proofs = info_span!("query phase").in_scope(|| {
        query_indices
            .iter()
            .map(|&index| QueryProof {
                input_proof: prove_input(index),
                commit_phase_openings: answer_query(config, &commit_phase_data, index),
            })
            .collect()
    });

    FriProof {
        commit_phase_commits,
        query_proofs,
        final_polys,
        pow_witness,
    }
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    final_polys: Vec<Vec<F>>,
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<Code, Val, Challenge, M, Challenger>(
    config: &FriConfig<M>,
    codewords: Vec<Codeword<Challenge, Code>>,
    challenger: &mut Challenger,
) -> CommitPhaseResult<Challenge, M>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    Code: FoldableCodeFamily<Challenge>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + CanObserve<M::Commitment>,
{
    let mut commits_and_data = vec![];

    let final_polys = config
        .fold_codewords(codewords, |log_folded_word_len, to_fold| {
            let (commit, _data) = commits_and_data.pushed_ref(
                config.mmcs.commit(
                    to_fold
                        .iter()
                        .map(|cw| {
                            RowMajorMatrix::new(
                                cw.word.clone(),
                                1 << (cw.code.log_word_len() - log_folded_word_len),
                            )
                        })
                        .collect(),
                ),
            );
            challenger.observe(commit.clone());
            Result::<_, Infallible>::Ok(challenger.sample_ext_element())
        })
        .unwrap()
        .into_iter()
        .map(|cw| cw.decode())
        .collect_vec();

    for fp in &final_polys {
        challenger.observe_ext_element_slice(&fp);
    }

    let (commits, data) = commits_and_data.into_iter().unzip();
    CommitPhaseResult {
        commits,
        data,
        final_polys,
    }
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
            let bits_reduced = config.log_folding_arity - sibs.log_strict_len();
            sibs.remove(index_in_subgroup >> bits_reduced);
        }
        steps.push(CommitPhaseProofStep { siblings, proof });
        index = folded_index;
    }
    steps
}
