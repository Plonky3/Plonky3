use alloc::vec;
use alloc::vec::Vec;

use itertools::izip;
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
use p3_commit::Mmcs;
use p3_field::{AbstractField, TwoAdicField};
use p3_matrix::Dimensions;

use crate::query_index::query_index_sibling;
use crate::{FriConfig, FriProof, InputOpening, QueryProof};

#[derive(Debug)]
pub enum VerificationError<InputMmcsErr, CommitMmcsErr> {
    InvalidProofShape,
    InputMmcsError(InputMmcsErr),
    CommitPhaseMmcsError(CommitMmcsErr),
}

pub type VerificationErrorForFriConfig<FC> = VerificationError<
    <<FC as FriConfig>::InputMmcs as Mmcs<<FC as FriConfig>::Challenge>>::Error,
    <<FC as FriConfig>::CommitPhaseMmcs as Mmcs<<FC as FriConfig>::Challenge>>::Error,
>;

type VerificationResult<T, FC> = Result<T, VerificationErrorForFriConfig<FC>>;

pub(crate) fn verify<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    input_dims: &[Vec<Dimensions>],
    input_commits: &[<FC::InputMmcs as Mmcs<FC::Challenge>>::Commitment],
    proof: &FriProof<FC>,
    challenger: &mut FC::Challenger,
) -> Result<(), VerificationErrorForFriConfig<FC>> {
    let alpha: FC::Challenge = challenger.sample_ext_element();

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

    // TODO: check proof shape
    let log_max_height = proof.commit_phase_commits.len() + config.log_blowup();

    for query_proof in &proof.query_proofs {
        let index = challenger.sample_bits(log_max_height);

        let reduced_opening = verify_input(
            input_mmcs,
            input_commits,
            input_dims,
            &query_proof.input_openings,
            index,
            alpha,
        )?;

        verify_query(
            config,
            &proof.commit_phase_commits,
            index,
            query_proof,
            &betas,
            reduced_opening,
            log_max_height,
        )?;
    }

    Ok(())
}

fn verify_input<FC: FriConfig>(
    input_mmcs: &[FC::InputMmcs],
    input_commits: &[<FC::InputMmcs as Mmcs<FC::Challenge>>::Commitment],
    input_dims: &[Vec<Dimensions>],
    input_openings: &[InputOpening<FC>],
    index: usize,
    alpha: FC::Challenge,
) -> VerificationResult<FC::Challenge, FC> {
    let mut reduced_opening = FC::Challenge::zero();

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

        for vals in opening.opened_values.iter() {
            for v in vals {
                reduced_opening *= alpha;
                reduced_opening += *v;
            }
        }
    }

    Ok(reduced_opening)
}

fn verify_query<FC: FriConfig>(
    config: &FC,
    commit_phase_commits: &[<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::Commitment],
    index: usize,
    proof: &QueryProof<FC>,
    betas: &[FC::Challenge],
    mut folded_eval: FC::Challenge,
    log_max_height: usize,
) -> Result<(), VerificationErrorForFriConfig<FC>> {
    let mut x = FC::Challenge::two_adic_generator(log_max_height).exp_u64(index as u64);

    for (i, (commit, step, &beta)) in
        izip!(commit_phase_commits, &proof.commit_phase_openings, betas).enumerate()
    {
        let log_height = log_max_height - i;
        let index_sibling = query_index_sibling(index, log_height);
        let index_pair = index_sibling >> 1;

        let mut evals = vec![folded_eval; 2];
        evals[index_sibling % 2] = step.sibling_value;

        let dims = &[Dimensions {
            width: 2,
            height: (1 << (log_height - 1)),
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

        x = x.square();
    }

    Ok(())
}
