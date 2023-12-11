use alloc::vec;
use alloc::vec::Vec;
use itertools::izip;
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
use p3_commit::Mmcs;
use p3_field::{AbstractField, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;

use crate::{query_index::query_index_sibling, FriConfig, FriProof, InputOpening, QueryProof};

pub(crate) fn verify<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    input_dims: &[Vec<Dimensions>],
    input_commits: &[<FC::InputMmcs as Mmcs<FC::Val>>::Commitment],
    proof: &FriProof<FC>,
    challenger: &mut FC::Challenger,
) -> Result<(), ()> {
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
        return Err(());
    }

    let log_max_height = proof.commit_phase_commits.len() + config.log_blowup();

    for query_proof in &proof.query_proofs {
        let index = challenger.sample_bits(log_max_height);

        let reduced_openings = verify_input(
            config,
            input_mmcs,
            input_commits,
            input_dims,
            &query_proof.input_openings,
            index,
            alpha,
            log_max_height,
        )?;

        let folded_eval = verify_query(
            config,
            &proof.commit_phase_commits,
            index,
            query_proof,
            &betas,
            &reduced_openings,
            log_max_height,
        )?;

        assert_eq!(folded_eval, proof.final_poly);
    }

    Ok(())
}

fn verify_input<FC: FriConfig>(
    config: &FC,
    input_mmcs: &[FC::InputMmcs],
    input_commits: &[<FC::InputMmcs as Mmcs<FC::Val>>::Commitment],
    input_dims: &[Vec<Dimensions>],
    input_openings: &[InputOpening<FC>],
    index: usize,
    alpha: FC::Challenge,
    log_max_height: usize,
) -> Result<Vec<FC::Challenge>, ()> {
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
        .unwrap();
        // .map_err(|_e| ())?;
        dbg!(dims, opening.opened_values.len());
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
    dbg!(&reduced_openings);
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
) -> Result<FC::Challenge, ()> {
    let mut folded_eval = FC::Challenge::zero();
    // let mut folded_eval = *reduced_openings.last().unwrap();
    // let mut folded_eval = *reduced_openings.last().unwrap();

    let mut x = FC::Challenge::two_adic_generator(log_max_height).exp_u64(index as u64);

    for (i, (commit, step, &beta)) in
        izip!(commit_phase_commits, &proof.commit_phase_openings, betas).enumerate()
    {
        let log_height = log_max_height - i;
        dbg!(log_height);

        folded_eval += reduced_openings[log_height];
        dbg!(folded_eval);

        // let index_sibling = query_index_sibling(index, log_height);
        // let index_pair = index_sibling >> 1;
        let index_sibling = index ^ 1;
        let index_pair = index >> 1;
        println!("i={i} log_height={log_height} index={index:#08b} index_pair={index_pair:#08b}");

        let mut evals = vec![folded_eval; 2];
        evals[index_sibling % 2] = step.sibling_value;

        let dims = &[Dimensions {
            width: 2,
            height: (1 << (log_height - 1)),
        }];
        println!("verifying batch");
        config
            .commit_phase_mmcs()
            .verify_batch(
                commit,
                dims,
                index_pair,
                &[evals.clone()],
                &step.opening_proof,
            )
            .unwrap();
        // .map_err(|_e| ())?;
        println!("verified batch");
        // .map_err(VerificationError::CommitPhaseMmcsError)?;

        let mut xs = [x; 2];
        xs[index_sibling % 2] *= FC::Challenge::two_adic_generator(1);
        // interpolate and evaluate at beta
        folded_eval = evals[0] + (beta - xs[0]) * (evals[1] - evals[0]) / (xs[1] - xs[0]);

        dbg!(folded_eval);

        /*
        if log_height > config.log_blowup() {
            folded_eval += reduced_openings[log_height - 1 - config.log_blowup()];
        }
        */

        index = index_pair;
        x = x.square();
    }

    Ok(folded_eval)
}
