use alloc::vec;
use alloc::vec::Vec;

use itertools::izip;
use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{AbstractField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::reverse_bits_len;

use crate::{FriConfig, FriProof, QueryProof};

#[derive(Debug)]
pub enum VerificationError<CommitMmcsErr> {
    InvalidProofShape,
    CommitPhaseMmcsError(CommitMmcsErr),
    FinalPolyMismatch,
    InvalidPowWitness,
}

pub type VerificationErrorForFriConfig<FC> = VerificationError<
    <<FC as FriConfig>::CommitPhaseMmcs as Mmcs<<FC as FriConfig>::Challenge>>::Error,
>;

type VerificationResult<FC, T> = Result<T, VerificationErrorForFriConfig<FC>>;

#[derive(Debug)]
pub struct FriChallenges<Challenge> {
    pub query_indices: Vec<usize>,
    betas: Vec<Challenge>,
}

pub fn verify_shape_and_sample_challenges<FC: FriConfig>(
    config: &FC,
    proof: &FriProof<FC>,
    challenger: &mut FC::Challenger,
) -> VerificationResult<FC, FriChallenges<FC::Challenge>> {
    let betas: Vec<FC::Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample()
        })
        .collect();

    if proof.query_proofs.len() != config.num_queries() {
        return Err(VerificationError::InvalidProofShape);
    }

    // Check PoW.
    if !challenger.check_witness(config.proof_of_work_bits(), proof.pow_witness) {
        return Err(VerificationError::InvalidPowWitness);
    }

    let log_max_height = proof.commit_phase_commits.len() + config.log_blowup();

    let query_indices: Vec<usize> = (0..config.num_queries())
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    Ok(FriChallenges {
        query_indices,
        betas,
    })
}

pub fn verify_challenges<FC: FriConfig>(
    config: &FC,
    proof: &FriProof<FC>,
    challenges: &FriChallenges<FC::Challenge>,
    reduced_openings: &[[FC::Challenge; 32]],
) -> VerificationResult<FC, ()> {
    let log_max_height = proof.commit_phase_commits.len() + config.log_blowup();
    for (&index, query_proof, ro) in izip!(
        &challenges.query_indices,
        &proof.query_proofs,
        reduced_openings
    ) {
        let folded_eval = verify_query(
            config,
            &proof.commit_phase_commits,
            index,
            query_proof,
            &challenges.betas,
            ro,
            log_max_height,
        )?;

        if folded_eval != proof.final_poly {
            return Err(VerificationError::FinalPolyMismatch);
        }
    }

    Ok(())
}

fn verify_query<FC: FriConfig>(
    config: &FC,
    commit_phase_commits: &[<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::Commitment],
    mut index: usize,
    proof: &QueryProof<FC>,
    betas: &[FC::Challenge],
    reduced_openings: &[FC::Challenge; 32],
    log_max_height: usize,
) -> VerificationResult<FC, FC::Challenge> {
    let mut folded_eval = FC::Challenge::zero();
    let mut x = FC::Challenge::two_adic_generator(log_max_height)
        .exp_u64(reverse_bits_len(index, log_max_height) as u64);

    for (log_folded_height, commit, step, &beta) in izip!(
        (0..log_max_height).rev(),
        commit_phase_commits,
        &proof.commit_phase_openings,
        betas,
    ) {
        folded_eval += reduced_openings[log_folded_height + 1];

        let index_sibling = index ^ 1;
        let index_pair = index >> 1;

        let mut evals = vec![folded_eval; 2];
        evals[index_sibling % 2] = step.sibling_value;

        let dims = &[Dimensions {
            width: 2,
            height: (1 << log_folded_height),
        }];
        config
            .commit_phase_mmcs()
            .verify_batch(
                commit,
                dims,
                index_pair,
                &[evals.clone()],
                &step.opening_proof,
            )
            .map_err(VerificationError::CommitPhaseMmcsError)?;

        let mut xs = [x; 2];
        xs[index_sibling % 2] *= FC::Challenge::two_adic_generator(1);
        // interpolate and evaluate at beta
        folded_eval = evals[0] + (beta - xs[0]) * (evals[1] - evals[0]) / (xs[1] - xs[0]);

        index = index_pair;
        x = x.square();
    }

    debug_assert!(index == 0 || index == 1);
    debug_assert!(x.is_one() || x == FC::Challenge::two_adic_generator(1));

    Ok(folded_eval)
}
