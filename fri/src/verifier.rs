use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Dimensions, Matrix};
use p3_util::reverse_bits_len;
use tracing::debug;

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

    // TODO[osama]: make sure this is correct
    let log_max_height = proof.commit_phase_commits.len() * config.arity_bits
        + config.log_blowup
        + config.log_final_poly_len;

    debug!("in verifier::verify, log_max_height: {:?}", log_max_height);

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

        let final_poly_index = index >> (proof.commit_phase_commits.len() * config.arity_bits);

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
    mut steps: impl Iterator<Item = CommitStep<'a, F, M>>,
    reduced_openings: Vec<(usize, F)>,
    log_max_height: usize,
) -> Result<F, FriError<M::Error, G::InputError>>
where
    F: Field,
    M: Mmcs<F> + 'a,
    G: FriGenericConfig<F>,
{
    debug!(
        "verifying query proof, index: {:?}, reduced_openings: {:#?}",
        index, reduced_openings
    );

    let mut folded_eval = F::ZERO;
    let mut ro_iter = reduced_openings.into_iter().peekable();

    let mut log_folded_height = log_max_height;

    while log_folded_height > config.log_blowup + config.log_final_poly_len {
        debug!(
            "query iteration, log_folded_height: {:?}",
            log_folded_height
        );

        if let Some((lh, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height) {
            debug!("adding directly from reduced openings at lh: {:?}", lh);
            folded_eval += ro;
        }

        let (&beta, comm, opening) = steps.next().unwrap();

        let index_row = index >> config.arity_bits;

        // Verify that `folded_eval` and evals from reduced_openings match opened_rows
        assert_eq!(folded_eval, opening.opened_rows[0][index % config.arity()]);
        for row in opening.opened_rows.iter().skip(1) {
            // TODO[osama]: consider adding an assert for the length of row to make sure the right thing is being read
            let (lh, ro) = ro_iter.next().unwrap();
            debug!("lh: {:?}", lh);
            debug!("row.len(): {:?}", row.len());
            let current_index = index >> (log_folded_height - lh);
            assert_eq!(
                ro,
                row[current_index % row.len()],
                "reduced opening mismatch at level {:?}",
                lh
            );
        }

        let dims: Vec<_> = opening
            .opened_rows
            .iter()
            .map(|opened_row| Dimensions {
                width: opened_row.len(),
                height: 1 << (log_folded_height - config.arity_bits), // TODO[osama]: revisit
            })
            .collect();

        debug!("dims: {:#?}", dims);

        config
            .mmcs
            .verify_batch(
                comm,
                &dims,
                index_row,
                &opening.opened_rows,
                &opening.opening_proof,
            )
            .map_err(FriError::CommitPhaseMmcsError)?;

        // Do the folding logic

        let mut opened_rows_iter = opening.opened_rows.iter().peekable();
        let mut folded_row = opened_rows_iter.next().unwrap().clone();
        let mut cnt = 1;
        while folded_row.len() > 1 {
            index >>= 1;
            log_folded_height -= 1;

            debug!("folding iteration: folded.len(): {:?}", folded_row.len());

            debug!("cnt: {:?}", cnt);
            // TODO[osama]: factor this out
            let folded_matrix = RowMajorMatrix::new(folded_row, 2);
            folded_row = folded_matrix
                .par_rows()
                .enumerate()
                .map(|(i, row)| {
                    debug!(
                        "folding row at index: {:?}",
                        (index_row << (config.arity_bits - cnt)) + i
                    );
                    g.fold_row(
                        (index_row << (config.arity_bits - cnt)) + i,
                        log_folded_height,
                        beta,
                        row.into_iter(),
                    )
                })
                .collect();

            cnt += 1;

            if let Some(poly_eval) = opened_rows_iter.next_if(|v| v.len() == folded_row.len()) {
                izip!(&mut folded_row, poly_eval).for_each(|(f, v)| *f += *v);
            }
        }

        folded_eval = folded_row.remove(0);
        assert!(folded_row.is_empty());
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
