use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Dimensions;
use p3_util::{split_bits, SliceExt, VecExt};

use crate::{Codeword, CommitPhaseProofStep, FoldableCodeFamily, FriConfig, FriProof};

#[derive(Debug)]
pub enum FriError<CommitMmcsErr> {
    InvalidProofShape,
    FinalPolyMismatch,
    InvalidPowWitness,
    CommitPhaseMmcsError(CommitMmcsErr),
}

pub fn verify<Code, Val, Challenge, M, Challenger>(
    config: &FriConfig<M>,
    codes: &[Code],
    query_samples: &[Vec<Challenge>],
    proof: &FriProof<Challenge, M, Challenger::Witness>,
    challenger: &mut Challenger,
) -> Result<Vec<usize>, FriError<M::Error>>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    Code: FoldableCodeFamily<Challenge>,
{
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample_ext_element()
        })
        .collect();

    for fp in &proof.final_polys {
        challenger.observe_ext_element_slice(&fp);
    }

    // Check PoW.
    if !challenger.check_witness(config.proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    if proof.query_proofs.len() != config.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    let index_bits = codes.iter().map(|c| c.log_word_len()).max().unwrap();

    let query_indices: Vec<_> = iter::repeat_with(|| challenger.sample_bits(index_bits))
        .take(config.num_queries)
        .collect();

    for (samples, query_proof) in izip!(query_samples, &proof.query_proofs) {
        let index = challenger.sample_bits(index_bits);

        let codewords = izip!(codes, samples)
            .map(|(c, &s)| Codeword::sample(c.clone(), index, s))
            .collect_vec();

        // let inputs = open_input(index, &qp.input_proof).map_err(FriError::InputError)?;

        let final_sample = verify_query(
            &config,
            codewords,
            izip!(
                &proof.commit_phase_commits,
                &betas,
                &query_proof.commit_phase_openings
            ),
        )?;

        /*
        debug_assert_eq!(final_sample.word.len(), 1);
        if final_sample
            .code
            .encoded_at_point(&proof.final_poly, final_sample.index)
            != final_sample.word[0]
        {
            return Err(FriError::FinalPolyMismatch);
        }
        */
    }

    Ok(query_indices)
}

type CommitStep<'a, F, M> = (
    &'a <M as Mmcs<F>>::Commitment,
    &'a F,
    &'a CommitPhaseProofStep<F, M>,
);

fn verify_query<'a, Code, F, M>(
    config: &FriConfig<M>,
    mut codewords: Vec<Codeword<F, Code>>,
    mut steps: impl Iterator<Item = CommitStep<'a, F, M>>,
) -> Result<Codeword<F, Code>, FriError<M::Error>>
where
    F: Field,
    Code: FoldableCodeFamily<F>,
    M: Mmcs<F> + 'a,
{
    let final_polys = config
        .fold_codewords(codewords, |to_fold| {
            let (comm, &beta, step) = steps.next().ok_or(FriError::InvalidProofShape)?;
            //
            todo!()
        })?
        .into_iter()
        .collect_vec();

    /*
    while let Some(log_max_word_len) = codewords
        .iter()
        .map(|cw| cw.code.log_word_len())
        .max()
        .filter(|&l| l > config.log_max_final_word_len)
    {
        let log_folded_word_len = log_max_word_len - config.log_folding_arity;
        let to_fold = codewords
            .extract(|cw| cw.code.log_word_len() > log_folded_word_len)
            .collect_vec();

        let (comm, &beta, step) = steps.next().ok_or(FriError::InvalidProofShape)?;

        // ...

        for mut cw in to_fold {
            cw.fold_repeatedly(cw.code.log_word_len() - log_folded_word_len, beta);
            if let Some(cw2) = codewords.iter_mut().find(|cw2| cw2.code == cw.code) {
                izip!(&mut cw2.word, cw.word).for_each(|(l, r)| *l += r);
            } else {
                codewords.push(cw);
            }
        }
    }
    */

    todo!()

    /*
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
    */
}

fn verify_step<F, Code, M>(
    mmcs: &M,
    comm: &M::Commitment,
    samples: &[Codeword<F, Code>],
    proof: &M::Proof,
) -> Result<(), M::Error>
where
    F: Field,
    Code: FoldableCodeFamily<F>,
    M: Mmcs<F>,
{
    let index = samples.iter().map(|cw| cw.index).all_equal_value().unwrap();
    let openings = samples.iter().map(|eval| eval.word.clone()).collect_vec();
    let dims = samples
        .iter()
        .map(|sample| Dimensions {
            height: 1 << sample.index_bits(),
            width: sample.word.len(),
        })
        .collect_vec();

    mmcs.verify_batch(comm, &dims, index, &openings, proof)
}
