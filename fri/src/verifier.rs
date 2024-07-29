use alloc::vec;
use alloc::vec::Vec;

use core::iter::zip;
use itertools::izip;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::reverse_bits_len;

use crate::{FriConfig, FriProof, QueryProof};

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

pub fn verify_shape_and_sample_challenges<F, EF, M, Challenger>(
    config: &FriConfig<M>,
    proof: &FriProof<EF, M, Challenger::Witness>,
    challenger: &mut Challenger,
) -> Result<FriChallenges<EF>, FriError<M::Error>>
where
    F: Field,
    EF: ExtensionField<F>,
    M: Mmcs<EF>,
    Challenger: GrindingChallenger + CanObserve<M::Commitment> + FieldChallenger<F>,
{
    let betas: Vec<EF> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample_ext_element()
        })
        .collect();
    challenger.observe_ext_element(proof.final_poly);
    if proof.query_proofs.len() != config.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    // Check PoW.
    if !challenger.check_witness(config.proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    let log_max_height = config.log_arity * proof.commit_phase_commits.len() + config.log_blowup;

    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    println!("Verifier query_indices: {:?}", query_indices);

    Ok(FriChallenges {
        query_indices,
        betas,
    })
}

pub fn verify_challenges<F, M, Witness>(
    config: &FriConfig<M>,
    proof: &FriProof<F, M, Witness>,
    challenges: &FriChallenges<F>,
    reduced_openings: &[[F; 32]],
) -> Result<(), FriError<M::Error>>
where
    F: TwoAdicField,
    M: Mmcs<F>,
{
    println!(
        "Number of commit phase stesp: {}",
        proof.commit_phase_commits.len()
    );
    let log_max_height = config.log_arity * proof.commit_phase_commits.len() + config.log_blowup;
    println!("Verifier phase log_max_height: {}", log_max_height);
    println!(
        "Verify Challenges Query indices: {:?}",
        challenges.query_indices
    );
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
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(())
}

fn verify_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::Commitment],
    mut index: usize,
    proof: &QueryProof<F, M>,
    betas: &[F],
    reduced_openings: &[F; 32],
    log_max_height: usize,
) -> Result<F, FriError<M::Error>>
where
    F: TwoAdicField,
    M: Mmcs<F>,
{
    let mut folded_eval = F::zero();
    let mut x = F::two_adic_generator(log_max_height)
        .exp_u64(reverse_bits_len(index, log_max_height) as u64);

    // TODO: Log_folded_height is a misnomer now, should rename.
    for ((i, log_folded_height), commit, step, &beta) in izip!(
        (config.log_blowup + config.log_arity - 1..(log_max_height))
            .rev()
            .step_by(config.log_arity)
            .enumerate(),
        commit_phase_commits,
        &proof.commit_phase_openings,
        betas,
    ) {
        folded_eval += reduced_openings[log_folded_height + 1];

        println!("Verifier phase log_folded_height: {}", log_folded_height);

        println!("Verifier phase index: {}", index);
        let mask = (1 << config.log_arity) - 1;
        let index_self_in_siblings = index & mask;
        println!("verifier phase index self: {}", index_self_in_siblings);
        let index_set = index >> config.log_arity;
        println!("verifier phase index set: {}", index_set);

        // println!("sibling vals: {:?}", step.siblings);
        println!("folded eval: {:?}", folded_eval);

        let mut evals: Vec<F> = step.siblings.clone();
        evals.insert(index_self_in_siblings, folded_eval);

        // println!("evals: {:?}", evals);

        assert_eq!(evals.len(), 1 << config.log_arity);

        let dims = &[Dimensions {
            width: 1 << config.log_arity,
            height: 1 << (log_folded_height + 1 - config.log_arity),
        }];

        println!("Dims: {:?}", dims);

        config
            .mmcs
            .verify_batch(
                commit,
                dims,
                index_set,
                &[evals.clone()],
                &step.opening_proof,
            )
            .map_err(|e| {
                println!("ERROR: {:?},", e);
                FriError::CommitPhaseMmcsError(e)
            })?;
        let g = F::two_adic_generator(config.log_arity);

        let mut xs = vec![x; 1 << config.log_arity];
        for (i, y) in g.powers().take(1 << config.log_arity).enumerate() {
            let index_of_x_times_y = ...;
            xs[index_of_x_times_y] *=y;
        }

        // interpolate and evaluate at beta
        folded_eval = interpolate_lagrange_and_evaluate(&xs, &evals, beta);

        index = index_set;
        x = x.exp_power_of_2(config.log_arity);
    }

    // debug_assert!(index < config.blowup(), "index was {}", index);
    // debug_assert_eq!(x.exp_power_of_2(config.log_blowup), F::one());

    Ok(folded_eval)
}

fn interpolate_lagrange_and_evaluate<F: TwoAdicField>(xs: &[F], ys: &[F], beta: F) -> F {
    assert_eq!(xs.len(), ys.len());

    let mut result = F::zero();

    for (i, (&xi, &yi)) in zip(xs, ys).enumerate() {
        let mut prod = yi;

        let mut normalizing_factor = F::one();

        for (j, &xj) in xs.iter().enumerate() {
            if i != j {
                normalizing_factor = normalizing_factor * (xi - xj);
            }
        }

        //
        prod = prod / normalizing_factor;

        let mut term = prod;

        for (j, &xj) in xs.iter().enumerate() {
            if i != j {
                term = term * (beta - xj);
            }
        }

        result = result + term;
    }

    result
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;

    use crate::verifier::interpolate_lagrange_and_evaluate;

    #[test]
    fn test_lagrange_interpolate() {
        let xs = [5, 1, 3, 9].map(BabyBear::from_canonical_u32);
        let ys = [1, 2, 3, 4].map(BabyBear::from_canonical_u32);

        for (x, y) in xs.iter().zip(ys.iter()) {
            assert_eq!(interpolate_lagrange_and_evaluate(&xs, &ys, *x), *y);
        }
    }
}
