use alloc::vec::Vec;
use core::iter;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Dimensions;

use crate::{Codeword, FoldableCodeFamily, FriConfig, FriProof};

#[derive(Debug)]
pub enum FriError<CommitMmcsErr> {
    InvalidProofShape,
    FinalPolyMismatch,
    InvalidPowWitness,
    CommitPhaseMmcsError(CommitMmcsErr),
}

pub fn verify<Code, Val, Challenge, M, Challenger, InputProof>(
    config: &FriConfig<M>,
    codes: &[Code],
    proof: &FriProof<Challenge, M, Challenger::Witness, InputProof>,
    challenger: &mut Challenger,
    open_input: impl Fn(usize, &InputProof) -> Vec<Challenge>,
) -> Result<(), FriError<M::Error>>
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

    for (&index, query_proof) in izip!(&query_indices, &proof.query_proofs) {
        let samples = open_input(index, &query_proof.input_proof);

        let codewords = izip!(codes, samples)
            .map(|(c, s)| Codeword::point_sample(c.clone(), index, s))
            .collect_vec();

        let mut steps = izip!(
            &proof.commit_phase_commits,
            &betas,
            &query_proof.commit_phase_openings
        );

        let final_samples = config
            .fold_codewords(codewords, |_, to_fold| {
                let (comm, &beta, step) = steps.next().ok_or(FriError::InvalidProofShape)?;
                for (sibs, cw) in izip!(&step.siblings, to_fold.iter_mut()) {
                    cw.expand(sibs.clone());
                }

                let index = to_fold.iter().map(|cw| cw.index).all_equal_value().unwrap();
                let openings = to_fold.iter().map(|cw| cw.word.clone()).collect_vec();
                let dims = to_fold
                    .iter()
                    .map(|cw| Dimensions {
                        height: 1 << cw.index_bits(),
                        width: cw.word.len(),
                    })
                    .collect_vec();

                config
                    .mmcs
                    .verify_batch(comm, &dims, index, &openings, &step.proof)
                    .map_err(FriError::CommitPhaseMmcsError)?;

                Ok(beta)
            })?
            .into_iter()
            .collect_vec();

        for (cw, fp) in izip!(final_samples, &proof.final_polys) {
            debug_assert_eq!(cw.word.len(), 1);
            if cw.word[0] != cw.code.encoded_at_index(fp, cw.index) {
                return Err(FriError::FinalPolyMismatch);
            }
        }
    }

    Ok(())
}
