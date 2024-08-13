use alloc::vec;
use alloc::vec::Vec;

use core::iter::zip;
use itertools::{izip, Itertools};
use std::iter::once;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::reverse_bits_len;

use crate::{CommitPhaseProofStep, FriConfig, FriProof, NormalizeQueryProof, QueryProof};

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
    pub normalize_betas: Vec<F>,
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
    let normalize_betas: Vec<EF> = proof
        .normalize_phase_commits
        .iter()
        .map(|(comm, _)| {
            challenger.observe(comm.clone());
            challenger.sample_ext_element()
        })
        .collect();

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

    if proof.normalize_query_proofs.len() != config.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    // // For each query, we must open the same number of commitments as appear in the shape.
    // for query_proof in &proof.normalize_query_proof {
    //     if query_proof.normalize_phase_openings.len() != proof.shape.len() {
    //         return Err(FriError::InvalidProofShape);
    //     }
    // }

    // Check PoW.
    if !challenger.check_witness(config.proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    println!(
        "Number of commit phase steps: {}",
        proof.commit_phase_commits.len(),
    );

    let log_max_normalized_height =
        config.log_arity * proof.commit_phase_commits.len() + config.log_blowup;

    let log_max_height = once(log_max_normalized_height)
        .chain(proof.normalize_phase_commits.iter().map(|(_, h)| *h))
        .max()
        .unwrap();

    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    // println!("Verifier query_indices: {:?}", query_indices);

    Ok(FriChallenges {
        query_indices,
        betas,
        normalize_betas,
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
    // println!(
    //     "Number of commit phase stesp: {}",
    //     proof.commit_phase_commits.len()
    // );
    let log_max_normalized_height =
        config.log_arity * proof.commit_phase_commits.len() + config.log_blowup;
    let log_max_height = once(log_max_normalized_height)
        .chain(proof.normalize_phase_commits.iter().map(|(_, h)| *h))
        .max()
        .unwrap();
    // println!("Verifier phase log_max_height: {}", log_max_height);
    // println!(
    //     "Verify Challenges Query indices: {:?}",
    //     challenges.query_indices
    // );
    println!(
        "Log max normalized height: {}, log max height: {}",
        log_max_normalized_height, log_max_height
    );
    for (&index, query_proof, normalize_query_proof, ro) in izip!(
        &challenges.query_indices,
        &proof.query_proofs,
        &proof.normalize_query_proofs,
        reduced_openings
    ) {
        let normalized_openings = normalize_openings(
            config,
            &proof.normalize_phase_commits,
            index,
            normalize_query_proof,
            &challenges.normalize_betas,
            ro,
            log_max_height,
        )?;
        println!("Openings successfully normalized");
        let folded_eval = verify_query(
            config,
            &proof.commit_phase_commits,
            index >> (log_max_height - log_max_normalized_height),
            query_proof,
            &challenges.betas,
            &normalized_openings,
        )?;

        // println!("Verifier phase folded_eval: {:?}", folded_eval);

        if folded_eval != proof.final_poly {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(())
}

fn normalize_openings<F: TwoAdicField, M: Mmcs<F>>(
    config: &FriConfig<M>,
    normalize_phase_commits: &[(M::Commitment, usize)],
    index: usize,
    normalize_proof: &NormalizeQueryProof<F, M>,
    betas: &[F],
    reduced_openings: &[F; 32],
    log_max_height: usize,
) -> Result<[F; 32], FriError<M::Error>> {
    let mut new_openings: [F; 32] = std::array::from_fn(|i| {
        if i >= config.log_blowup && (i - config.log_blowup) % config.log_arity == 0 {
            reduced_openings[i]
        } else {
            F::zero()
        }
    });

    let x = F::two_adic_generator(log_max_height)
        .exp_u64(reverse_bits_len(index, log_max_height) as u64);

    for ((commit, log_height), step, beta) in izip!(
        normalize_phase_commits.iter(),
        normalize_proof.normalize_phase_openings.iter(),
        betas
    ) {
        // We shouldn't have normalize commitments where the height is equal to a multiple of the arity
        // added to the log_blowup.
        assert!((log_height - config.log_blowup) % config.log_arity != 0);
        let new_x = x.exp_power_of_2(log_max_height - log_height);
        let num_folds = (log_height - config.log_blowup) % config.log_arity;
        let folded_height = log_height - num_folds;

        assert!((folded_height - config.log_blowup) % config.log_arity == 0);
        println!("About to verify a fold step in normalize_openings");

        new_openings[folded_height] += verify_fold_step(
            reduced_openings[*log_height],
            *beta,
            num_folds,
            step,
            commit,
            index >> (log_max_height - log_height),
            *log_height - num_folds,
            &config.mmcs,
            new_x,
        )?;
    }

    Ok(new_openings)
}

fn verify_fold_step<F: TwoAdicField, M: Mmcs<F>>(
    folded_eval: F,
    beta: F,
    num_folds: usize,
    step: &CommitPhaseProofStep<F, M>,
    commit: &M::Commitment,
    index: usize,
    log_folded_height: usize,
    mmcs: &M,
    x: F,
) -> Result<F, FriError<M::Error>> {
    let mask = (1 << num_folds) - 1;
    let index_self_in_siblings = index & mask;
    let index_set = index >> num_folds;

    // Fold in the normalized phase commits between log_folded_height and log_folded_height +
    // config.log_arity.
    // for i in 1..config.log_arity {
    //     let evals = normalize_phase_commits.iter().filter_map(|(commit, h)| {
    //         if h == log_folded_height + i {
    //             Some()
    //         } else {
    //             None
    //         }
    //     });
    // }

    let evals: Vec<F> = step.siblings.clone();
    assert_eq!(evals[index_self_in_siblings], folded_eval);

    assert_eq!(evals.len(), 1 << num_folds);

    let dims = &[Dimensions {
        width: 1 << num_folds,
        height: 1 << (log_folded_height),
    }];

    println!("Dims: {:?}", dims);
    println!("Log folded height: {}", log_folded_height);

    mmcs.verify_batch(
        commit,
        dims,
        index_set,
        &[evals.clone()],
        &step.opening_proof,
    )
    .map_err(|e| FriError::CommitPhaseMmcsError(e))?;

    let g = F::two_adic_generator(num_folds);

    let mut ord_idx = index_self_in_siblings;
    let mut ord_evals = vec![];

    let xs = g.powers().take(1 << num_folds).map(|y| x * y).collect_vec();
    for _ in 0..(1 << num_folds) {
        ord_evals.push(evals[ord_idx]);
        ord_idx = next_index_in_coset(ord_idx, num_folds);
    }

    // Interpolate and evaluate at beta
    Ok(interpolate_lagrange_and_evaluate(&xs, &ord_evals, beta))
}

fn verify_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::Commitment],
    mut index: usize,
    proof: &QueryProof<F, M>,
    betas: &[F],
    reduced_openings: &[F; 32],
) -> Result<F, FriError<M::Error>>
where
    F: TwoAdicField,
    M: Mmcs<F>,
{
    let log_max_normalized_height =
        config.log_arity * commit_phase_commits.len() + config.log_blowup;

    for (_, ro) in reduced_openings.iter().enumerate().filter(|(i, _)| {
        (i >= &config.log_blowup) && (i - config.log_blowup) % config.log_arity != 0
    }) {
        assert!(ro.is_zero());
    }

    let mut folded_eval = reduced_openings[log_max_normalized_height];

    let mut x = F::two_adic_generator(log_max_normalized_height)
        .exp_u64(reverse_bits_len(index, log_max_normalized_height) as u64);

    // TODO: Log_folded_height is a misnomer now, should rename.
    for (log_folded_height, commit, step, &beta) in izip!(
        (config.log_blowup..log_max_normalized_height + 1 - config.log_arity)
            .rev()
            .step_by(config.log_arity),
        commit_phase_commits,
        &proof.commit_phase_openings,
        betas,
    ) {
        folded_eval = verify_fold_step(
            folded_eval,
            beta,
            config.log_arity,
            step,
            commit,
            index,
            log_folded_height,
            &config.mmcs,
            x,
        )?;
        index = index >> config.log_arity;
        x = x.exp_power_of_2(config.log_arity);

        folded_eval += reduced_openings[log_folded_height];
    }

    Ok(folded_eval)
}

fn next_index_in_coset(index: usize, log_arity: usize) -> usize {
    let mut result = reverse_bits_len(index, log_arity);
    result += 1;
    reverse_bits_len(result, log_arity)
}

// Inefficient algorithm for interpolation and evaluation of a polynomial at a point.
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
