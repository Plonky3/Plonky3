use alloc::vec;
use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
use p3_commit::Mmcs;
use p3_field::{AbstractField, TwoAdicField};
use p3_matrix::Dimensions;

use crate::query_index::query_index_sibling;
use crate::{FriConfig, FriProof, InputOpening, QueryProof};

pub enum VerificationError<FC: FriConfig> {
    InvalidProofShape,
    InputMmcsError(<FC::InputMmcs as Mmcs<FC::Challenge>>::Error),
    CommitPhaseMmcsError(<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::Error),
}

pub(crate) fn verify<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    input_commits: &[<FC::InputMmcs as Mmcs<FC::Challenge>>::Commitment],
    proof: &FriProof<FC>,
    challenger: &mut FC::Challenger,
) -> Result<(), VerificationError<FC>> {
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

        // TODO: this assumes all matrices are max height, should have this passed in
        let input_dims = query_proof
            .input_openings
            .iter()
            .map(|opening| {
                opening
                    .opened_values
                    .iter()
                    .map(|vals| Dimensions {
                        width: vals.len(),
                        height: 1 << log_max_height,
                    })
                    .collect_vec()
            })
            .collect_vec();

        let initial_eval = verify_input(
            input_mmcs,
            input_commits,
            &input_dims,
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
            initial_eval,
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
) -> Result<FC::Challenge, VerificationError<FC>> {
    let mut reduced = FC::Challenge::zero();

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
        .map_err(|e| VerificationError::InputMmcsError(e))?;

        for vals in &opening.opened_values {
            for v in vals {
                reduced *= alpha;
                reduced += *v;
            }
        }
    }

    Ok(reduced)
}

fn verify_query<FC: FriConfig>(
    config: &FC,
    commit_phase_commits: &[<FC::CommitPhaseMmcs as Mmcs<FC::Challenge>>::Commitment],
    index: usize,
    proof: &QueryProof<FC>,
    betas: &[FC::Challenge],
    mut folded_eval: FC::Challenge,
    log_max_height: usize,
) -> Result<(), VerificationError<FC>> {
    let mut x = FC::Challenge::generator()
        * FC::Challenge::two_adic_generator(log_max_height).exp_u64(index as u64);

    for (i, (commit, step, &beta)) in
        izip!(commit_phase_commits, &proof.commit_phase_openings, betas).enumerate()
    {
        let log_height = log_max_height - i;
        let index_sibling = query_index_sibling(index, log_height);
        let index_pair = index_sibling >> 1;

        let dims = &[Dimensions {
            width: 2,
            height: (1 << (log_height - 1)),
        }];
        let mut evals = vec![folded_eval; 2];
        evals[index_sibling % 2] = step.sibling_value;

        config
            .commit_phase_mmcs()
            .verify_batch(
                commit,
                dims,
                index_pair,
                &[evals.clone()],
                &step.opening_proof,
            )
            .map_err(|e| VerificationError::CommitPhaseMmcsError(e))?;

        let mut xs = [x; 2];
        xs[index_sibling % 2] *= FC::Challenge::two_adic_generator(1);
        // interpolate and evaluate at beta
        folded_eval = evals[0] + (beta - xs[0]) * (evals[1] - evals[0]) / (xs[1] - xs[0]);

        x = x.square();
    }

    Ok(())
}
