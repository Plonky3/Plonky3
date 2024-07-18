use alloc::vec;
use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Dimensions;
use p3_util::{bitmask, log2_strict_usize, SliceExt, VecExt};

use crate::{Codeword, CommitPhaseProofStep, FoldableLinearCode, FriConfig, FriProof};

#[derive(Debug)]
pub enum FriError<CommitMmcsErr, InputError> {
    InvalidProofShape,
    CommitPhaseMmcsError(CommitMmcsErr),
    InputError(InputError),
    FinalPolyMismatch,
    InvalidPowWitness,
}

pub fn verify<Code, Val, Challenge, M, Challenger, InputProof, InputError>(
    config: &FriConfig<M>,
    proof: &FriProof<Challenge, M, Challenger::Witness, InputProof>,
    challenger: &mut Challenger,
    open_input: impl Fn(usize, &InputProof) -> Result<Vec<Codeword<Challenge, Code>>, InputError>,
) -> Result<(), FriError<M::Error, InputError>>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    Code: FoldableLinearCode<Challenge>,
{
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample_ext_element()
        })
        .collect();

    for &symbol in &proof.final_poly {
        challenger.observe_ext_element(symbol);
    }

    let log_final_poly_len = proof
        .final_poly
        .log_len()
        .ok_or(FriError::InvalidProofShape)?;

    if proof.query_proofs.len() != config.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    // Check PoW.
    if !challenger.check_witness(config.proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    let log_folding_arities = proof
        .query_proofs
        .iter()
        .map(|qp| qp.log_folding_arities())
        .all_equal_value()
        .unwrap();

    assert_eq!(log_folding_arities.len(), betas.len());

    let log_max_msg_len = log_final_poly_len + log_folding_arities.iter().sum::<usize>();

    for qp in &proof.query_proofs {
        let mut index = challenger.sample_bits(config.log_blowup + log_max_msg_len);
        let mut inputs = open_input(index, &qp.input_proof)
            .map_err(FriError::InputError)?
            .into_iter()
            .peekable();

        let mut log_msg_len = log_max_msg_len;
        let mut folded_evals: Vec<Codeword<Challenge, Code>> = vec![];

        assert_eq!(proof.commit_phase_commits.len(), betas.len());
        assert_eq!(proof.commit_phase_commits.len(), log_folding_arities.len());
        assert_eq!(
            proof.commit_phase_commits.len(),
            qp.commit_phase_openings.len()
        );

        for (comm, &beta, &log_arity, step) in izip!(
            &proof.commit_phase_commits,
            &betas,
            &log_folding_arities,
            &qp.commit_phase_openings,
        ) {
            let log_folded_msg_len = log_msg_len - log_arity;

            folded_evals
                .extend(inputs.peeking_take_while(|cw| cw.code.log_msg_len() > log_folded_msg_len));

            let folded_index = index >> log_arity;

            assert_eq!(step.openings.len(), folded_evals.len());
            for (eval, siblings) in izip!(&mut folded_evals, &step.openings) {
                *eval = eval.expand(siblings.to_vec());
            }

            let openings = folded_evals
                .iter()
                .map(|eval| eval.word.clone())
                .collect_vec();

            let dims = folded_evals
                .iter()
                .map(|eval| Dimensions {
                    width: eval.word.len(),
                    height: 1 << eval.index_bits(),
                })
                .collect_vec();

            config
                .mmcs
                .verify_batch(comm, &dims, folded_index, &openings, &step.proof)
                .map_err(FriError::CommitPhaseMmcsError)?;

            for eval in &mut folded_evals {
                eval.fold_to_point(beta);
            }

            Codeword::sum_words_from_same_code(&mut folded_evals);

            log_msg_len = log_folded_msg_len;
            index = folded_index;
        }

        assert_eq!(inputs.next(), None);
        assert_eq!(folded_evals.len(), 1);
        let eval = folded_evals.pop().unwrap();
        assert_eq!(eval.word.len(), 1);
        assert_eq!(
            eval.code.encoded_at_point(&proof.final_poly, eval.index),
            eval.word[0]
        );
    }

    Ok(())
}

type CommitStep<'a, F, M> = (
    &'a F,
    &'a <M as Mmcs<F>>::Commitment,
    &'a CommitPhaseProofStep<F, M>,
);

fn verify_query<'a, F, M, InputError>(
    config: &FriConfig<M>,
    mut index: usize,
    steps: impl Iterator<Item = CommitStep<'a, F, M>>,
    reduced_openings: Vec<(usize, F)>,
) -> Result<F, FriError<M::Error, InputError>>
where
    F: Field,
    M: Mmcs<F> + 'a,
{
    let mut ro_iter = reduced_openings.into_iter().peekable();
    let (mut log_height, mut folded_eval) = ro_iter.next().unwrap();

    /*
    for (&beta, comm, opening) in steps {
        log_height -= 1;

        let index_sibling = index ^ 1;
        let index_pair = index >> 1;

        let mut evals = vec![folded_eval; 2];
        evals[index_sibling % 2] = opening.sibling_value;

        let dims = &[Dimensions {
            width: 2,
            height: 1 << log_height,
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
        folded_eval = g.fold_row(index, log_height, beta, evals.into_iter());

        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_height) {
            folded_eval += ro;
        }
    }

    debug_assert!(
        ro_iter.next().is_none(),
        "verifier reduced_openings were not in descending order?",
    );
    */

    Ok(folded_eval)
}
