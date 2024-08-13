use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{fold, CommitPhaseProofStep, FriConfig, FriProof, NormalizeQueryProof, QueryProof};

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
    let log_max_height = input.iter().rposition(Option::is_some).unwrap();

    let normalize_phase_result = normalize_phase(config, input, log_max_height, challenger);

    let commit_phase_result = commit_phase(
        config,
        &normalize_phase_result.normalized_inputs,
        challenger,
    );

    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    // Fold the inputs which do not conform to the standardized shape and provide the proofs
    // necessary to evaluate the folded polynomials at the appropriate query indices.
    let normalize_query_proofs = info_span!("normalize query phase").in_scope(|| {
        query_indices
            .iter()
            .map(|&index| NormalizeQueryProof {
                normalize_phase_openings: normalize_phase_result
                    .data
                    .iter()
                    .zip_eq(
                        normalize_phase_result
                            .commits
                            .iter()
                            .map(|(_, height)| *height),
                    )
                    .map(|(commitment, height)| {
                        let shift = (height - config.log_blowup) % config.log_arity;
                        answer_query_single_step(
                            config,
                            commitment,
                            index >> (shift + log_max_height - height),
                            shift,
                        )
                    })
                    .collect(),
            })
            .collect()
    });

    // The query proofs are as for log_arity = 1 but the index may have too many bits because the normalized
    // max height may not be the same as the original max height.
    let query_proofs = info_span!("query phase").in_scope(|| {
        let shift = (log_max_height - config.log_blowup) % config.log_arity;
        query_indices
            .iter()
            .map(|&index| answer_query(config, &commit_phase_result.data, index >> shift))
            .collect()
    });

    (
        FriProof {
            commit_phase_commits: commit_phase_result.commits,
            normalize_phase_commits: normalize_phase_result.commits,
            normalize_query_proofs,
            query_proofs,
            final_poly: commit_phase_result.final_poly,
            pow_witness,
        },
        query_indices,
    )
}

/// A function to answer a single step of the query phase.
fn answer_query_single_step<F: Field, M: Mmcs<F>>(
    config: &FriConfig<M>,
    commit: &M::ProverData<RowMajorMatrix<F>>,
    index: usize,
    log_num_leaves: usize,
) -> CommitPhaseProofStep<F, M> {
    let (mut opened_rows, opening_proof) = config.mmcs.open_batch(index, commit);

    assert_eq!(opened_rows.len(), 1);
    let opened_row = opened_rows.pop().unwrap();
    assert_eq!(
        opened_row.len(),
        1 << log_num_leaves,
        "Committed data should be in tuples of size arity."
    );

    let siblings = opened_row;

    CommitPhaseProofStep {
        siblings,
        opening_proof,
    }
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
            let folded_index = index_i >> log_arity;
            answer_query_single_step(config, commit, folded_index, config.log_arity)
        })
        .collect();

    QueryProof {
        commit_phase_openings,
    }
}

#[instrument(name = "commit phase", skip_all)]
fn normalize_phase<F, EF, M, Challenger>(
    config: &FriConfig<M>,
    input: &[Option<Vec<EF>>; 32],
    log_max_height: usize,
    challenger: &mut Challenger,
) -> NormalizePhaseResult<EF, M>
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    Challenger: CanObserve<M::Commitment> + FieldChallenger<F>,
{
    let mut commits = vec![];
    let mut data = vec![];

    let mut normalized_inputs = core::array::from_fn(|i| {
        if i >= config.log_blowup && (i - config.log_blowup) % config.log_arity == 0 {
            input[i].clone()
        } else {
            None
        }
    });

    for log_height in (config.log_blowup..log_max_height + 1)
        .rev()
        .filter(|&i| (i - config.log_blowup) % config.log_arity != 0 && input[i].is_some())
    {
        let current = input[log_height].as_ref().unwrap().clone();
        let num_folds = (log_height - config.log_blowup) % config.log_arity;
        let leaves = RowMajorMatrix::new(current.clone(), 1 << num_folds);

        let (commit, prover_data) = config.mmcs.commit_matrix(leaves);
        challenger.observe(commit.clone());
        commits.push((commit, log_height));
        data.push(prover_data);

        let beta: EF = challenger.sample_ext_element();
        match &mut normalized_inputs[log_height - num_folds] {
            Some(v) => {
                v.iter_mut()
                    .zip_eq(fold(current, beta, num_folds))
                    .for_each(|(v_elem, c_elem)| *v_elem += c_elem);
            }
            None => {
                normalized_inputs[log_height - num_folds] = Some(fold(current, beta, num_folds));
            }
        }
    }

    NormalizePhaseResult {
        commits,
        data,
        normalized_inputs,
    }
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<F, EF, M, Challenger>(
    config: &FriConfig<M>,
    input: &[Option<Vec<EF>>; 32],
    challenger: &mut Challenger,
) -> CommitPhaseResult<EF, M>
where
    F: Field,
    EF: TwoAdicField + ExtensionField<F>,
    M: Mmcs<EF>,
    Challenger: CanObserve<M::Commitment> + FieldChallenger<F>,
{
    // By the time the prover gets to this phase, the `Some` inputs must all come from polynomials
    // whose log-degree is a multiple of `config.log_arity`.
    assert!(input.iter().all(|x| x.is_none()
        || (log2_strict_usize(x.as_ref().unwrap().len()) - config.log_blowup) % config.log_arity
            == 0));

    let log_max_height = input.iter().rposition(Option::is_some).unwrap();

    let mut current = input[log_max_height].as_ref().unwrap().clone();

    let mut commits = vec![];
    let mut data = vec![];

    for log_folded_height in (config.log_blowup..(log_max_height + 1 - config.log_arity))
        .rev()
        .step_by(config.log_arity)
    {
        // A row of `leaves` is the information necessary to open the folded polynomial at a given
        // index.
        let leaves = RowMajorMatrix::new(current.clone(), 1 << config.log_arity);

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

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(current.len(), config.blowup());

    let final_poly = current[0];

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

struct NormalizePhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<(M::Commitment, usize)>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    normalized_inputs: [Option<Vec<F>>; 32],
}
