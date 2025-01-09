use alloc::vec;
use alloc::vec::Vec;
use core::iter;
use core::simd::ToBytes;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::config::RoundConfig;
use crate::coset::Radix2Coset;
use crate::proof::RoundProof;
use crate::utils::sample_integer;
use crate::{StirConfig, StirParameters, StirProof};

pub struct StirWitness<F: TwoAdicField, M: Mmcs<F>> {
    pub(crate) domain: Radix2Coset<F>,
    // pub(crate) polynomial: DensePolynomial<F>,
    pub(crate) merkle_tree: M::ProverData<RowMajorMatrix<F>>,
    pub(crate) folded_evals: RowMajorMatrix<F>,
    pub(crate) round: usize,
    pub(crate) folding_randomness: F,
}

pub fn fold_evals<F: TwoAdicField>(
    evals: RowMajorMatrix<F>,
    folding_randomness: F,
    folding_factor: usize,
) -> Vec<F> {
    todo!()
}

pub fn commit<F, M>(config: &StirConfig<M>, input: Vec<F>) -> (StirWitness<F, M>, M::Commitment)
where
    F: TwoAdicField,
    M: Mmcs<F>,
{
    let domain = Radix2Coset::new_from_degree_and_rate(
        config.log_starting_degree(),
        config.log_starting_inv_rate(),
    );

    let folded_evals = RowMajorMatrix::new(input, 2);
    let (commitment, merkle_tree) = config.mmcs.commit_matrix(folded_evals);

    (
        StirWitness {
            domain,
            merkle_tree,
            folded_evals,
            round: 0,
            folding_randomness: F::one(),
        },
        commitment,
    )
}

pub fn prove<F, M, Challenger>(
    config: &StirConfig<M>,
    input: Vec<F>,
    challenger: &mut Challenger,
) -> StirProof<F, M, Challenger::Witness>
where
    F: TwoAdicField,
    M: Mmcs<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    assert!(input.len() <= 1 << (config.log_starting_degree() + config.log_starting_inv_rate()));

    // NP TODO: Should the prover call commit like in Plonky3's FRI?
    // or should be called separately like in Giacomo's code?
    let (witness, commitment) = commit(config, input);

    // Observe the commitment
    challenger.observe(commitment);
    let folding_randomness = challenger.sample_ext_element();

    let mut witness = StirWitness {
        folding_randomness,
        ..witness
    };

    let mut round_proofs = vec![];
    for _ in 0..config.num_rounds {
        let (witness, round_proof) = prove_round(config, &mut witness, challenger);
        round_proofs.push(round_proof);
    }

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
        pow_witness,
    }
}

fn prove_round<F, M, Challenger>(
    config: &StirConfig<M>,
    witness: StirWitness<F, M>,
    challenger: &mut Challenger,
) -> (StirWitness<F, M>, RoundProof<F, M>)
where
    F: TwoAdicField,
    M: Mmcs<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger + CanObserve<M::Commitment>,
{
    // De-structure the round-specific configuration and the witness
    let RoundConfig {
        log_folding_factor,
        log_evaluation_domain_size,
        pow_bits,
        num_queries,
        ood_samples,
        log_inv_rate,
    } = config.round_config(witness.round);

    let StirWitness {
        domain,
        merkle_tree,
        folded_evals,
        round,
        folding_randomness,
    } = witness;

    // Compute the evaluations of the folded polynomial
    let evals = fold_evals(folded_evals, folding_randomness, 1 << log_folding_factor);

    // Shrink the evaluation domain by a factor of 2 (log_scale_factor = 1)
    let new_domain = domain.shrink(1);

    // Stack the new folded evaluations, commit and observe the commitment
    let stacked_evals = RowMajorMatrix::new(evals, 2);
    let (commitment, merkle_tree) = config.mmcs.commit_matrix(stacked_evals);
    challenger.observe(commitment);

    // Sample a new folding randomness
    let folding_randomness = challenger.sample_ext_element();

    // ========= OOD SAMPLING =========

    let ood_samples = (0..ood_samples).map(|_| challenger.sample_ext_element());

    // Evaluate the polynomial at the OOD samples
    let betas = ood_samples
        .cloned()
        .map(|x| domain.evaluate_interpolation(&stacked_evals, x));

    // Observe the betas
    challenger.observe_slice(&betas);

    // ========= STIR MESSAGE =========

    // Sample ramdomness for degree correction
    let comb_randomness = challenger.sample_ext_element();

    // Sample randomness for the next folding
    let folding_randomness = challenger.sample_ext_element();

    // Sample queried indices of elements in L^k
    let scaling_factor = 1 << (domain.log_size() - log_folding_factor);

    // NP TODO: No index deduplication
    let queried_indices = (0..num_queries).map(|_| sample_integer(challenger, scaling_factor));

    // Proof of work witness
    let pow_witness = challenger.grind(pow_bits);

    // ========= QUERY PROOFS =========

    let query_proofs: Vec<(Vec<Vec<F>>, Vec<M::Proof>)> =
        queried_indices.map(|index| config.mmcs.open_batch(index, &merkle_tree));
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
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = vec![];
    let mut data = vec![];

    // f_{folded, beta}(h^2) = 1/2 * [beta/h * f(h) - beta/h * f(-h)  ]

    // log_domain =    0,  1,  2, 3, 4, 5, 6, 7
    // f(domain) =     10, -1, 2, 5, 3, 7, 7, 2

    //  10, -1, 2, 5,
    //  3, 7, 7, 2

    //                              0,  2,  4,  6
    // f_{folded, beta}(domain) =

    let mut i = 0;
    while folded.len() > config.blowup() {
        let leaves = RowMajorMatrix::new(folded, 2);
        let (commit, prover_data) = config.mmcs.commit_matrix(leaves);
        challenger.observe(commit.clone());

        let beta: Challenge = challenger.sample_ext_element();
        // We passed ownership of `current` to the MMCS, so get a reference to it
        let leaves = config.mmcs.get_matrices(&prover_data).pop().unwrap();
        folded = g.fold_matrix(beta, leaves.as_view());

        commits.push(commit);
        data.push(prover_data);

        if let Some(v) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            izip!(&mut folded, v).for_each(|(c, x)| *c += x);
        }
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(folded.len(), config.blowup());
    let final_poly = folded[0];
    for x in folded {
        assert_eq!(x, final_poly);
    }
    challenger.observe_ext_element(final_poly);

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
            let index_i = index >> i;
            let index_i_sibling = index_i ^ 1;
            let index_pair = index_i >> 1;

            let (mut opened_rows, opening_proof) = config.mmcs.open_batch(index_pair, commit);
            assert_eq!(opened_rows.len(), 1);
            let opened_row = opened_rows.pop().unwrap();
            assert_eq!(opened_row.len(), 2, "Committed data should be in pairs");
            let sibling_value = opened_row[index_i_sibling % 2];

            CommitPhaseProofStep {
                sibling_value,
                opening_proof,
            }
        })
        .collect()
}
