use alloc::vec;
use alloc::vec::Vec;

use itertools::izip;
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::Field;
use p3_matrix::Dimensions;

use crate::{CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof};

#[derive(Debug)]
pub enum FriError<CommitMmcsErr> {
    InvalidProofShape,
    CommitPhaseMmcsError(CommitMmcsErr),
    FinalPolyMismatch,
    InvalidPowWitness,
}

#[derive(Debug)]
pub struct FriChallenges<F> {
    pub query_indices: Vec<usize>,
    pub betas: Vec<F>,
}

pub fn verify<G, F, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    proof: &FriProof<F, M, Challenger::Witness, G::InputProof>,
    challenger: &mut Challenger,
    open_input: impl Fn(usize, &G::InputProof) -> [F; 32],
) -> Result<(), FriError<M::Error>>
where
    F: Field,
    M: Mmcs<F>,
    Challenger: GrindingChallenger + CanObserve<M::Commitment> + CanSample<F>,
    G: FriGenericConfig<F>,
{
    let betas: Vec<F> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample()
        })
        .collect();

    if proof.query_proofs.len() != config.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    // Check PoW.
    if !challenger.check_witness(config.proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    let log_max_height = proof.commit_phase_commits.len() + config.log_blowup;

    for qp in &proof.query_proofs {
        let index = challenger.sample_bits(log_max_height + g.extra_query_index_bits());
        let ro = open_input(index, &qp.input_proof);

        let folded_eval = verify_query(
            g,
            config,
            &proof.commit_phase_commits,
            index >> g.extra_query_index_bits(),
            &qp.commit_phase_openings,
            &betas,
            &ro,
            log_max_height,
        )?;

        if folded_eval != proof.final_poly {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_query<G, F, M>(
    g: &G,
    config: &FriConfig<M>,
    commit_phase_commits: &[M::Commitment],
    mut index: usize,
    commit_phase_openings: &[CommitPhaseProofStep<F, M>],
    betas: &[F],
    reduced_openings: &[F; 32],
    log_max_height: usize,
) -> Result<F, FriError<M::Error>>
where
    F: Field,
    M: Mmcs<F>,
    G: FriGenericConfig<F>,
{
    let mut folded_eval = F::zero();

    for (log_folded_height, commit, step, &beta) in izip!(
        (0..log_max_height).rev(),
        commit_phase_commits,
        commit_phase_openings,
        betas,
    ) {
        folded_eval += reduced_openings[log_folded_height + 1];

        let index_sibling = index ^ 1;
        let index_pair = index >> 1;

        let mut evals = vec![folded_eval; 2];
        evals[index_sibling % 2] = step.sibling_value;

        let dims = &[Dimensions {
            width: 2,
            height: 1 << log_folded_height,
        }];
        config
            .mmcs
            .verify_batch(
                commit,
                dims,
                index_pair,
                &[evals.clone()],
                &step.opening_proof,
            )
            .map_err(FriError::CommitPhaseMmcsError)?;

        index = index_pair;

        // If verification is extremely performance-critical (such as in recursive setting),
        // this can be changed to a stateful API to save intermediate computations.
        folded_eval = g.fold_row(index, log_folded_height, beta, evals.into_iter());
    }

    debug_assert!(index < config.blowup(), "index was {}", index);

    Ok(folded_eval)
}
