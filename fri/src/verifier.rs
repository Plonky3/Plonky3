use alloc::vec;
use alloc::vec::Vec;

use itertools::{Itertools, izip};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::reverse_bits_len;

use crate::{CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof};

#[derive(Debug)]
pub enum FriError<CommitMmcsErr, InputError> {
    InvalidProofShape,
    CommitPhaseMmcsError(CommitMmcsErr),
    InputError(InputError),
    FinalPolyMismatch,
    InvalidPowWitness,
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
    // Get an extension field challenge for each commitment.
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample_algebra_element()
        })
        .collect();

    // Observe all coefficients of the final polynomial.
    proof
        .final_poly
        .iter()
        .for_each(|x| challenger.observe_algebra_element(*x));

    if proof.query_proofs.len() != config.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    // Check the Proof of Work witness.
    if !challenger.check_witness(config.proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    // The size of the initial evaluation domain should be equal to the log_blowup plus
    // the log of the initial trace length (trace polynomial degree). This second quantity
    // will be equal to the number of commit phase commits plus the log of the final polynomial length.
    let log_max_height =
        proof.commit_phase_commits.len() + config.log_blowup + config.log_final_poly_len;

    for qp in &proof.query_proofs {
        let index = challenger.sample_bits(log_max_height + g.extra_query_index_bits());
        let ro = open_input(index, &qp.input_proof).map_err(FriError::InputError)?;

        debug_assert!(
            ro.iter().tuple_windows().all(|((l, _), (r, _))| l > r),
            "reduced openings sorted by height descending"
        );

        let folded_eval = verify_query(
            g,
            config,
            index >> g.extra_query_index_bits(),
            izip!(
                &betas,
                &proof.commit_phase_commits,
                &qp.commit_phase_openings
            ),
            ro,
            log_max_height,
        )?;

        let final_poly_index = index >> (proof.commit_phase_commits.len());

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
    reduced_openings: Vec<(usize, F)>,
    log_max_height: usize,
) -> Result<F, FriError<M::Error, G::InputError>>
where
    F: Field,
    M: Mmcs<F> + 'a,
    G: FriGenericConfig<F>,
{
    let mut folded_eval = F::ZERO;
    let mut ro_iter = reduced_openings.into_iter().peekable();

    for (log_folded_height, (&beta, comm, opening)) in izip!((0..log_max_height).rev(), steps) {
        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height + 1) {
            folded_eval += ro;
        }

        let index_sibling = index ^ 1;
        let index_pair = index >> 1;

        let mut evals = vec![folded_eval; 2];
        evals[index_sibling % 2] = opening.sibling_value;

        let dims = &[Dimensions {
            width: 2,
            height: 1 << log_folded_height,
        }];
        config
            .mmcs
            .verify_batch(
                comm,
                dims,
                index_pair,
                &[evals.clone()],
                &opening.opening_proof,
            )
            .map_err(FriError::CommitPhaseMmcsError)?;

        index = index_pair;

        folded_eval = g.fold_row(index, log_folded_height, beta, evals.into_iter());
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
