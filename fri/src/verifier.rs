use alloc::vec;
use alloc::vec::Vec;

use itertools::izip;
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{AbstractField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::{FriConfig, FriProof, InputOpening, QueryProof};

#[derive(Debug)]
pub enum VerificationError<InputMmcsErr, CommitMmcsErr> {
    InvalidProofShape,
    InputMmcsError(InputMmcsErr),
    CommitPhaseMmcsError(CommitMmcsErr),
    FinalPolyMismatch,
    InvalidPowWitness,
}

pub type VerificationErrorForFriConfig<FC> = VerificationError<
    <<FC as FriConfig>::InputMmcs as Mmcs<<FC as FriConfig>::Val>>::Error,
    <<FC as FriConfig>::CommitPhaseMmcs as Mmcs<<FC as FriConfig>::Challenge>>::Error,
>;

type VerificationResult<FC, T> = Result<T, VerificationErrorForFriConfig<FC>>;

pub fn verify<FC: FriConfig>(
    config: &FC,
    /*
    input_mmcs: &[FC::InputMmcs],
    input_dims: &[Vec<Dimensions>],
    input_commits: &[<FC::InputMmcs as Mmcs<FC::Val>>::Commitment],
    */
    proof: &FriProof<FC>,
    input: &[Vec<FC::Challenge>],
    challenger: &mut FC::Challenger,
) -> VerificationResult<FC, Vec<usize>> {
    // let alpha: FC::Challenge = challenger.sample_ext_element();
    let betas: Vec<FC::Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample_ext_element()
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

    for (&index, query_proof, reduced_openings) in izip!(&query_indices, &proof.query_proofs, input)
    {
        /*
        let index = challenger.sample_bits(log_max_height);

        let reduced_openings = verify_input(
            input_mmcs,
            input_commits,
            input_dims,
            &query_proof.input_openings,
            index,
            alpha,
            log_max_height,
        )?;
        let reduced_openings = todo!();
        */

        let folded_eval = verify_query(
            config,
            &proof.commit_phase_commits,
            index,
            query_proof,
            &betas,
            reduced_openings,
            log_max_height,
        )?;

        if folded_eval != proof.final_poly {
            return Err(VerificationError::FinalPolyMismatch);
        }
    }

    Ok(query_indices)
}

fn verify_input<FC: FriConfig>(
    input_mmcs: &[FC::InputMmcs],
    input_commits: &[<FC::InputMmcs as Mmcs<FC::Val>>::Commitment],
    input_dims: &[Vec<Dimensions>],
    input_openings: &[InputOpening<FC>],
    index: usize,
    alpha: FC::Challenge,
    log_max_height: usize,
) -> VerificationResult<FC, Vec<FC::Challenge>> {
    let mut openings_by_log_height: Vec<Vec<FC::Val>> = vec![vec![]; log_max_height + 1];
    for (mmcs, commit, dims, opening) in
        izip!(input_mmcs, input_commits, input_dims, input_openings)
    {
        mmcs.verify_batch(
            commit,
            dims,
            index,
            &opening.opened_values,
            &opening.opening_proof,
        )
        .map_err(VerificationError::InputMmcsError)?;
        for (mat_dims, mat_opening) in izip!(dims, &opening.opened_values) {
            let log_height = log2_strict_usize(mat_dims.height);
            openings_by_log_height[log_height].extend_from_slice(mat_opening);
        }
    }
    let reduced_openings = openings_by_log_height
        .into_iter()
        .map(|o| {
            o.into_iter()
                .zip(alpha.powers())
                .map(|(y, alpha_pow)| alpha_pow * y)
                .sum()
        })
        .collect();
    Ok(reduced_openings)
}

fn verify_query<FC: FriConfig>(
    config: &FC,
    commit_phase_commits: &[<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::Commitment],
    mut index: usize,
    proof: &QueryProof<FC>,
    betas: &[FC::Challenge],
    reduced_openings: &[FC::Challenge],
    log_max_height: usize,
) -> VerificationResult<FC, FC::Challenge> {
    let mut folded_eval = FC::Challenge::zero();
    let mut x = FC::Challenge::two_adic_generator(log_max_height)
        .exp_u64(reverse_bits_len(index, log_max_height) as u64);

    for (log_folded_height, commit, step, &beta, &reduced_opening_for_height) in izip!(
        (0..log_max_height).rev(),
        commit_phase_commits,
        &proof.commit_phase_openings,
        betas,
        reduced_openings.iter().rev()
    ) {
        folded_eval += reduced_opening_for_height;

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
