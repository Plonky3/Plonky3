use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Dimensions;
use p3_util::{split_bits, SliceExt};

use crate::{Codeword, CommitPhaseProofStep, FoldableLinearCodeFamily, FriConfig, FriProof};

#[derive(Debug)]
pub enum FriError<CommitMmcsErr> {
    InvalidProofShape,
    FinalPolyMismatch,
    InvalidPowWitness,
    CommitPhaseMmcsError(CommitMmcsErr),
}
pub fn verify<Code, Val, Challenge, M, Challenger>(
    config: &FriConfig<M>,
    proof: &FriProof<Challenge, M, Challenger::Witness>,
    log_word_lens: &[usize],
    query_inputs: &[Vec<Challenge>],
    challenger: &mut Challenger,
    // open_input: impl Fn(usize, &InputProof) -> Result<Vec<Codeword<Challenge, Code>>, InputError>,
) -> Result<Vec<usize>, FriError<M::Error>>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    Code: FoldableLinearCodeFamily<Challenge>,
{
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample_ext_element()
        })
        .collect();

    challenger.observe_ext_element_slice(&proof.final_poly);

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
        .map_err(|_| FriError::InvalidProofShape)?;

    assert_eq!(log_folding_arities.len(), betas.len());

    let index_bits =
        config.log_blowup + log_final_poly_len + log_folding_arities.iter().sum::<usize>();

    let query_indices: Vec<_> = iter::repeat_with(|| challenger.sample_bits(index_bits))
        .take(config.num_queries)
        .collect();

    for (inputs, qp) in izip!(query_inputs, &proof.query_proofs) {
        let index = challenger.sample_bits(index_bits);
        // let inputs = open_input(index, &qp.input_proof).map_err(FriError::InputError)?;

        let final_sample = verify_query::<Code, _, _, _>(
            &config,
            // index_bits,
            index,
            izip!(log_word_lens.iter().copied(), inputs.iter().copied()),
            izip!(
                &proof.commit_phase_commits,
                &betas,
                &qp.commit_phase_openings
            ),
        )?;

        debug_assert_eq!(final_sample.word.len(), 1);
        if final_sample
            .code
            .encoded_at_point(&proof.final_poly, final_sample.index)
            != final_sample.word[0]
        {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(query_indices)
}

type CommitStep<'a, F, M> = (
    &'a <M as Mmcs<F>>::Commitment,
    &'a F,
    &'a CommitPhaseProofStep<F, M>,
);

fn verify_query<'a, Code, F, M, InputError>(
    config: &FriConfig<M>,
    mut index: usize,
    // log_word_lens: &[usize],
    // mut log_word_len: usize,
    inputs: impl Iterator<Item = (usize, F)>,
    steps: impl Iterator<Item = CommitStep<'a, F, M>>,
) -> Result<Codeword<F, Code>, FriError<M::Error>>
where
    F: Field,
    Code: FoldableLinearCodeFamily<F>,
    M: Mmcs<F> + 'a,
{
    let mut inputs = inputs.peekable();
    let mut log_word_len = inputs.peek().unwrap().0;
    let mut prev_inputs: Vec<(usize, F)> = vec![];

    for (comm, &beta, step) in steps {
        let log_folded_word_len = log_word_len - config.log_folding_arity;
        let (folded_index, index_in_subgroup) = split_bits(index, config.log_folding_arity);

        // log_word_len -= config.log_folding_arity;

        let openings: Vec<Vec<F>> = step
            .siblings
            .iter()
            .zip(prev_inputs.drain(..).chain(&mut inputs))
            .map(|(sibs, (log_input_word_len, sample))| {
                let log_sibs_len = log_input_word_len - log_folded_word_len;
                assert_eq!(sibs.len() + 1, 1 << log_sibs_len);
                let mut sibs = sibs.to_vec();
                sibs.insert(
                    index_in_subgroup >> (config.log_folding_arity - log_sibs_len),
                    sample,
                );
                sibs
            })
            .collect();

        // samples.extend(inputs.peeking_take_while(|cw| cw.code.log_word_len() > log_word_len));

        // for (sibs, sample) in

        for (sample, sibs) in izip!(&mut samples, &step.siblings) {
            *sample = sample.expand(sibs.to_vec());
        }

        verify_step(&config.mmcs, comm, &samples, &step.proof)
            .map_err(FriError::CommitPhaseMmcsError)?;

        /*
        for sample in &mut samples {
            sample.fold_to_log_word_len(log_word_len, beta);
        }
        */

        Codeword::sum_words_from_same_code(&mut samples);
    }

    debug_assert_eq!(inputs.next(), None);
    debug_assert_eq!(samples.len(), 1);
    Ok(samples.pop().unwrap())
}

fn verify_step<F, Code, M>(
    mmcs: &M,
    comm: &M::Commitment,
    samples: &[Codeword<F, Code>],
    proof: &M::Proof,
) -> Result<(), M::Error>
where
    F: Field,
    Code: FoldableLinearCodeFamily<F>,
    M: Mmcs<F>,
{
    let index = samples.iter().map(|cw| cw.index).all_equal_value().unwrap();
    let openings = samples.iter().map(|eval| eval.word.clone()).collect_vec();
    let dims = samples
        .iter()
        .map(|sample| Dimensions {
            width: sample.word.len(),
            height: 1 << sample.index_bits(),
        })
        .collect_vec();

    mmcs.verify_batch(comm, &dims, index, &openings, proof)
}
