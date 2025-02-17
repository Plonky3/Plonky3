use alloc::vec::Vec;
use core::iter;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::reverse_bits_len;

use crate::{CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof, NormalizeQueryProof};

#[derive(Debug)]
pub enum FriError<CommitMmcsErr, InputError> {
    InvalidProofShape,
    CommitPhaseMmcsError(CommitMmcsErr),
    InputError(InputError),
    FinalPolyMismatch,
    InvalidPowWitness,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn verify_normalization_phase<F, G, M>(
    g: &G,
    config: &FriConfig<M>,
    normalize_phase_commits: &[(M::Commitment, usize)],
    index: usize,
    normalize_proof: &NormalizeQueryProof<F, M>,
    betas: &[F],
    reduced_openings: &[(usize, F)],
    log_max_height: usize,
) -> Result<Vec<(usize, F)>, FriError<M::Error, G::InputError>>
where
    F: TwoAdicField,
    G: FriGenericConfig<F>,
    M: Mmcs<F>,
{
    // Populate the return value with zeros, or with the reduced openings at the correct indices.
    let mut new_openings = [F::ZERO; 32];
    reduced_openings
        .iter()
        .cloned()
        .for_each(|(log_height, reduced_opening)| {
            if (log_height - config.log_final_poly_len - config.log_blowup) % config.log_arity == 0
            {
                new_openings[log_height] = reduced_opening
            }
        });

    for ((commit, log_height), step, reduced_opening, beta) in izip!(
        normalize_phase_commits.iter(),
        normalize_proof.normalize_phase_openings.iter(),
        reduced_openings.iter().filter(|(log_height, _)| {
            (log_height >= &config.log_blowup)
                && (log_height - config.log_final_poly_len - config.log_blowup) % config.log_arity
                    != 0
        }),
        betas
    ) {
        debug_assert_eq!(*log_height, reduced_opening.0);
        let num_folds =
            (log_height - config.log_final_poly_len - config.log_blowup) % config.log_arity;
        let log_folded_height = log_height - num_folds;

        // Verify the fold step and update the new openings. `folded_height` is the closest
        // "normalized" height to `log_height`. `step` and `commit` give us the information necessary
        // to fold the unnormalized opening from `log_height` multiple steps down to `folded_height`.
        new_openings[log_folded_height] += verify_fold_step(
            g,
            reduced_opening.1,
            *beta,
            num_folds,
            step,
            commit,
            index >> (log_max_height - log_height),
            *log_height - num_folds,
            &config.mmcs,
        )?;
    }

    Ok(new_openings
        .into_iter()
        .enumerate()
        .rev()
        .filter_map(|(i, opening)| {
            if !opening.is_zero() {
                Some((i, opening))
            } else {
                None
            }
        })
        .collect_vec())
}

pub fn verify<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    proof: &FriProof<Challenge, M, Challenger::Witness, G::InputProof>,
    challenger: &mut Challenger,
    open_input: impl Fn(usize, &G::InputProof) -> Result<Vec<(usize, Challenge)>, G::InputError>,
) -> Result<(), FriError<M::Error, G::InputError>>
where
    Val: Field,
    Challenge: ExtensionField<Val> + TwoAdicField,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    // The prover sampled the normalize betas first, so we need to sample these first here.
    let normalize_betas: Vec<Challenge> = proof
        .normalize_phase_commits
        .iter()
        .map(|(comm, _)| {
            challenger.observe(comm.clone());
            challenger.sample_ext_element()
        })
        .collect();

    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample_ext_element()
        })
        .collect();

    // Observe all coefficients of the final polynomial.
    proof
        .final_poly
        .iter()
        .for_each(|x| challenger.observe_ext_element(*x));

    if proof.query_proofs.len() != config.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    // Check PoW.
    if !challenger.check_witness(config.proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    // The max height passed into `verify_challenges` can be computed from the config and the number
    // of commit phase steps.
    let log_max_normalized_height = config.log_arity * proof.commit_phase_commits.len()
        + config.log_final_poly_len
        + config.log_blowup;

    // The overall max height is either the maximum of the heights associated with the normalize
    // phase commits, or it's the max normalized height.
    let log_max_height = iter::once(log_max_normalized_height)
        .chain(proof.normalize_phase_commits.iter().map(|(_, h)| *h))
        .max()
        .unwrap();

    for (qp, np) in izip!(&proof.query_proofs, &proof.normalize_query_proofs) {
        let index = challenger.sample_bits(log_max_height + g.extra_query_index_bits());
        let ro = open_input(index, &qp.input_proof).map_err(FriError::InputError)?;

        debug_assert!(
            ro.iter().tuple_windows().all(|((l, _), (r, _))| l > r),
            "reduced openings sorted by height descending"
        );

        // The function `verify_query` expects its `reduced_openings` argument to have a "normalized"
        // shape (all non-zero entries must be at indices that are multiples of `config.log_arity`
        // added to the log blowup factor), so we first normalize the openings.
        let normalized_openings = verify_normalization_phase(
            g,
            config,
            &proof.normalize_phase_commits,
            index >> g.extra_query_index_bits(),
            np,
            &normalize_betas,
            &ro,
            log_max_height,
        )?;

        let folded_eval = verify_query(
            g,
            config,
            index >> (log_max_height - log_max_normalized_height) >> g.extra_query_index_bits(),
            izip!(
                &betas,
                &proof.commit_phase_commits,
                &qp.commit_phase_openings
            ),
            betas.len(),
            normalized_openings,
        )?;

        let final_poly_index = index
            >> (log_max_height - log_max_normalized_height)
            >> (config.log_arity * proof.commit_phase_commits.len());

        let mut eval = Challenge::ZERO;
        // We open the final polynomial at index `final_poly_index`, which corresponds to evaluating
        // the polynomial at x^k, where x is the 2-adic generator of order `max_height` and k is
        // `reverse_bits_len(final_poly_index, log_max_height)`.
        let x = Challenge::two_adic_generator(log_max_height)
            .exp_u64(reverse_bits_len(final_poly_index, log_max_height) as u64);
        let mut x_pow = Challenge::ONE;

        // Evaluate the final polynomial at x.
        for coeff in &proof.final_poly {
            eval += *coeff * x_pow;
            x_pow *= x;
        }

        if eval != folded_eval {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(())
}

type CommitStep<'a, F, M> = (
    &'a F,
    &'a <M as Mmcs<F>>::Commitment,
    &'a CommitPhaseProofStep<F, M>,
);

fn verify_query<'a, G, F, M>(
    g: &G,
    config: &FriConfig<M>,
    mut index: usize,
    steps: impl Iterator<Item = CommitStep<'a, F, M>>,
    step_len: usize,
    reduced_openings: Vec<(usize, F)>,
) -> Result<F, FriError<M::Error, G::InputError>>
where
    F: TwoAdicField,
    M: Mmcs<F> + 'a,
    G: FriGenericConfig<F>,
{
    // Deduce the max normalized height as in `verify_shape_and_sample_challenges`.
    let log_max_normalized_height =
        config.log_arity * step_len + config.log_final_poly_len + config.log_blowup;

    // Check that the reduced openings are in a "normalized" shape.
    for (_, ro) in reduced_openings.iter().filter(|(log_height, _)| {
        (log_height >= &config.log_blowup)
            && (log_height - config.log_final_poly_len - config.log_blowup) % config.log_arity != 0
    }) {
        assert!(ro.is_zero());
    }

    let mut ro_iter = reduced_openings.into_iter().peekable();
    let mut folded_eval = ro_iter.next().unwrap().1;

    for (log_folded_height, (&beta, comm, step)) in izip!(
        (config.log_final_poly_len + config.log_blowup
            ..(log_max_normalized_height + 1).saturating_sub(config.log_arity))
            .rev()
            .step_by(config.log_arity),
        steps
    ) {
        folded_eval = verify_fold_step(
            g,
            folded_eval,
            beta,
            config.log_arity,
            step,
            comm,
            index,
            log_folded_height,
            &config.mmcs,
        )?;
        index >>= config.log_arity;

        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height) {
            folded_eval += ro;
        }
    }
    debug_assert!(
        index < config.blowup() * config.final_poly_len(),
        "index was {}",
        index,
    );
    debug_assert!(
        ro_iter.next().is_none(),
        "verifier reduced_openings were not in descending order?"
    );
    Ok(folded_eval)
}

/// Verify a single FRI fold consistency check.
///
/// The functions `verify_query` and `verify_normalization_phase` both call this function.
#[allow(clippy::too_many_arguments)]
fn verify_fold_step<G: FriGenericConfig<F>, F: TwoAdicField, M: Mmcs<F>>(
    g: &G,
    folded_eval: F,
    beta: F,
    num_folds: usize,
    step: &CommitPhaseProofStep<F, M>,
    commit: &M::Commitment,
    index: usize,
    log_folded_height: usize,
    mmcs: &M,
) -> Result<F, FriError<M::Error, G::InputError>> {
    let mask = (1 << num_folds) - 1;
    let index_self_in_siblings = index & mask;
    let index_set = index >> num_folds;

    let mut evals: Vec<F> = step.sibling_values.clone();
    evals.insert(index_self_in_siblings, folded_eval);

    // `commit` should be a commitment to a matrix with 2^num_folds columns and 2^log_folded_height
    // rows.
    debug_assert_eq!(evals.len(), 1 << num_folds);
    let dims = &[Dimensions {
        width: 1 << num_folds,
        height: 1 << (log_folded_height),
    }];

    mmcs.verify_batch(
        commit,
        dims,
        index_set,
        &[evals.clone()],
        &step.opening_proof,
    )
    .map_err(FriError::CommitPhaseMmcsError)?;

    Ok(g.fold_row(index_set, log_folded_height, num_folds, beta, evals))
}
