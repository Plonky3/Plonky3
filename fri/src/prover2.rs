use alloc::vec;
use alloc::vec::Vec;
use core::{iter, mem};
use p3_matrix::Matrix;
use p3_util::{bitmask, SliceExt, VecExt};

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::{Codeword, CommitPhaseProofStep, FoldableLinearCode, FriConfig, FriProof, QueryProof};

#[instrument(name = "FRI prover", skip_all)]
pub fn prove<Code, Val, Challenge, M, Challenger, InputProof>(
    config: &FriConfig<M>,
    inputs: Vec<Codeword<Challenge, Code>>,
    challenger: &mut Challenger,
    prove_input: impl Fn(usize) -> InputProof,
) -> FriProof<Challenge, M, Challenger::Witness, InputProof>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    Code: FoldableLinearCode<Challenge>,
{
    // check sorted strictly decreasing
    assert!(inputs
        .iter()
        .tuple_windows()
        .all(|(l, r)| l.word.len() >= r.word.len()));

    let log_max_height = inputs[0].word.log_strict_len();

    let commit_phase_result = commit_phase(config, inputs, challenger);

    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let query_proofs = info_span!("query phase").in_scope(|| {
        iter::repeat_with(|| challenger.sample_bits(log_max_height))
            .take(config.num_queries)
            .map(|index| QueryProof {
                input_proof: prove_input(index),
                commit_phase_openings: answer_query(config, &commit_phase_result.data, index),
            })
            .collect()
    });

    FriProof {
        commit_phase_commits: commit_phase_result.commits,
        query_proofs,
        final_poly: commit_phase_result.final_poly,
        pow_witness,
    }
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    final_poly: Vec<F>,
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<Code, Val, Challenge, M, Challenger>(
    config: &FriConfig<M>,
    inputs: Vec<Codeword<Challenge, Code>>,
    challenger: &mut Challenger,
) -> CommitPhaseResult<Challenge, M>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + CanObserve<M::Commitment>,
    Code: FoldableLinearCode<Challenge>,
{
    let mut inputs = inputs.into_iter().peekable();
    let mut log_height = log2_strict_usize(inputs.peek().unwrap().word.len());
    let mut folded: Vec<Codeword<Challenge, Code>> = vec![];
    let mut commits_and_data = vec![];

    while inputs.peek().is_some() || log_height > config.log_max_final_poly_len + config.log_blowup
    {
        let log_folded_height = log_height - config.log_folding_arity;

        // Append inputs with length greater than folded height
        folded.extend(inputs.peeking_take_while(|cw| cw.word.log_strict_len() > log_folded_height));

        let mats: Vec<_> = folded
            .iter()
            .map(|cw| {
                RowMajorMatrix::new(
                    cw.word.clone(),
                    1 << (cw.word.log_strict_len() - log_folded_height),
                )
            })
            .collect();
        assert!(mats.iter().all(|m| m.height() == 1 << log_folded_height));

        let (commit, _data) = commits_and_data.pushed_ref(config.mmcs.commit(mats));
        challenger.observe(commit.clone());

        let beta: Challenge = challenger.sample_ext_element();

        for cw in &mut folded {
            let log_arity = cw.word.log_strict_len() - log_folded_height;
            cw.repeatedly_fold(log_arity, beta);
        }

        Codeword::sum_words_from_same_code(&mut folded);
        log_height = log_folded_height;
    }

    assert_eq!(folded.len(), 1);
    let final_poly = folded.pop().unwrap().decode();
    for &sym in &final_poly {
        challenger.observe_ext_element(sym);
    }

    let (commits, data) = commits_and_data.into_iter().unzip();
    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
}

fn answer_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_data: &[M::ProverData<RowMajorMatrix<F>>],
    index: usize,
) -> Vec<CommitPhaseProofStep<F, M>>
where
    F: Field,
    M: Mmcs<F>,
{
    dbg!(index);
    commit_phase_data
        .iter()
        .enumerate()
        .map(|(i, data)| {
            // let folded_index = index >> ((i + 1) * config.log_folding_arity);

            let index = index >> (i * config.log_folding_arity);
            let folded_index = index >> config.log_folding_arity;
            let index_in_subgroup = index & bitmask(config.log_folding_arity);

            dbg!(folded_index, index_in_subgroup);

            let (mut openings, proof) = config.mmcs.open_batch(folded_index, data);
            for o in &mut openings {
                o.remove(index_in_subgroup >> (config.log_folding_arity - o.log_strict_len()));
            }
            CommitPhaseProofStep { openings, proof }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_challenger::{CanSample, CanSampleBits, DuplexChallenger};
    use p3_commit::ExtensionMmcs;
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
    use p3_field::{dot_product, extension::BinomialExtensionField, TwoAdicField};
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_util::{reverse_bits_len, reverse_slice_index_bits};
    use rand::{
        distributions::{Distribution, Standard},
        Rng, SeedableRng,
    };
    use rand_chacha::ChaCha20Rng;

    use crate::verifier::verify;

    use super::*;

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs = FieldMerkleTreeMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        8,
    >;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type MyFriConfig = FriConfig<ChallengeMmcs>;

    #[derive(Debug, PartialEq, Eq, Clone)]
    struct RsCode<F> {
        log_blowup: usize,
        log_msg_len: usize,
        _phantom: PhantomData<F>,
    }
    impl<F> RsCode<F> {
        fn new_rs(log_blowup: usize, log_msg_len: usize) -> Self {
            Self {
                log_blowup,
                log_msg_len,
                _phantom: PhantomData,
            }
        }
    }
    impl<F: TwoAdicField> FoldableLinearCode<F> for RsCode<F> {
        /*
        fn new(log_blowup: usize, log_msg_len: usize) -> Self {
            Self::new_rs(log_blowup, log_msg_len)
        }
        */
        fn log_blowup(&self) -> usize {
            self.log_blowup
        }
        fn log_msg_len(&self) -> usize {
            self.log_msg_len
        }
        fn encoded_at_point(&self, msg: &[F], index: usize) -> F {
            let x = F::two_adic_generator(self.log_word_len())
                .exp_u64(reverse_bits_len(index, self.log_word_len()) as u64);
            dot_product(msg.iter().copied(), x.powers())
        }
        fn encode(&self, message: &[F]) -> Vec<F> {
            let mut coeffs = message.to_vec();
            assert_eq!(coeffs.log_strict_len(), self.log_msg_len);
            coeffs.resize(coeffs.len() << self.log_blowup, F::zero());
            let mut evals = Radix2Dit::default().dft(coeffs.to_vec());
            reverse_slice_index_bits(&mut evals);
            evals
        }
        fn decode(&self, codeword: &[F]) -> Vec<F> {
            let mut evals = codeword.to_vec();
            reverse_slice_index_bits(&mut evals);
            assert_eq!(evals.log_strict_len(), self.log_msg_len + self.log_blowup);
            let mut coeffs = Radix2Dit::default().idft(evals);
            assert!(coeffs.drain((1 << self.log_msg_len)..).all(|x| x.is_zero()));
            coeffs
        }
        fn folded_code(&self) -> Self {
            Self::new_rs(self.log_blowup, self.log_msg_len - 1)
        }
        fn fold_word_at_point(&self, beta: F, index: usize, (e0, e1): (F, F)) -> F {
            let subgroup_start = F::two_adic_generator(self.log_msg_len + self.log_blowup)
                .exp_u64(reverse_bits_len(index, self.log_msg_len + self.log_blowup - 1) as u64);
            let mut xs = F::two_adic_generator(1)
                .shifted_powers(subgroup_start)
                .take(2)
                .collect_vec();
            reverse_slice_index_bits(&mut xs);
            // interpolate and evaluate at beta
            e0 + (beta - xs[0]) * (e1 - e0) / (xs[1] - xs[0])
        }

        /*
        fn fold_once(&mut self, beta: F, codeword: &mut Vec<F>) {
            assert!(self.log_height > self.log_blowup);
            assert_eq!(codeword.log_strict_len(), self.log_height);

            let g_inv = F::two_adic_generator(self.log_height).inverse();
            let one_half = F::two().inverse();
            let half_beta = beta * one_half;

            let mut powers = g_inv
                .shifted_powers(half_beta)
                .take(codeword.len() / 2)
                .collect_vec();
            reverse_slice_index_bits(&mut powers);

            *codeword = codeword
                .drain(..)
                .tuples()
                .zip(powers)
                .map(|((lo, hi), power)| (one_half + power) * lo + (one_half - power) * hi)
                .collect();

            self.log_height -= 1;
        }
        */
    }

    fn rand_codewords<F: TwoAdicField>(
        r: &mut impl Rng,
        log_blowup: usize,
        log_msg_lens: &[usize],
    ) -> Vec<Codeword<F, RsCode<F>>>
    where
        Standard: Distribution<F>,
    {
        log_msg_lens
            .iter()
            .map(|&l| {
                let code = RsCode::new_rs(log_blowup, l);
                let word = code.encode(&(0..(1 << l)).map(|_| r.gen()).collect_vec());
                Codeword {
                    code,
                    index: 0,
                    word,
                }
            })
            .collect()
    }

    #[test]
    fn test_rs_code() {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let mut cw = rand_codewords::<Challenge>(&mut rng, 1, &[5])
            .pop()
            .unwrap();

        cw.decode();
        cw.fold(rng.gen());
        cw.decode();
        cw.fold(rng.gen());
        cw.decode();
        cw.fold(rng.gen());
    }

    #[test]
    fn test_fri_rs() {
        let mut rng = ChaCha20Rng::seed_from_u64(0);

        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear::default(),
            &mut rng,
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));

        let config = MyFriConfig {
            log_blowup: 1,
            log_max_final_poly_len: 3,
            log_folding_arity: 1,
            num_queries: 1,
            proof_of_work_bits: 8,
            mmcs,
        };

        let chal = Challenger::new(perm.clone());

        let inputs = rand_codewords::<Challenge>(&mut rng, config.log_blowup, &[8, 7, 6, 5]);

        // let result = commit_phase::<RsCode<Challenge>, _, _, _, _>(&fri_config, inputs, &mut chal);

        let mut p_chal = chal.clone();
        let proof = prove::<RsCode<Challenge>, _, _, _, _, ()>(
            &config,
            inputs.clone(),
            &mut p_chal,
            |_| (),
        );

        for (qi, qp) in proof.query_proofs.iter().enumerate() {
            println!("query {qi}:");
            for (si, step) in qp.commit_phase_openings.iter().enumerate() {
                println!("  step {si}:");
                for (oi, opening) in step.openings.iter().enumerate() {
                    println!("    opening {oi}:");
                    for (vi, val) in opening.iter().enumerate() {
                        println!("      val {vi}: {val}");
                    }
                }
            }
        }

        // dbg!(&proof.query_proofs[0].commit_phase_openings[0].openings);

        let mut v_chal = chal.clone();
        let log_max_word_len = inputs.iter().map(|i| i.code.log_word_len()).max().unwrap();
        verify::<RsCode<Challenge>, _, _, _, _, (), ()>(
            &config,
            &proof,
            &mut v_chal,
            |index, _p| {
                Ok(inputs
                    .iter()
                    .map(|i| {
                        i.segment(
                            i.code.log_word_len(),
                            index >> (log_max_word_len - i.code.log_word_len()),
                        )
                    })
                    .collect())
            },
        )
        .unwrap();

        assert_eq!(
            <Challenger as CanSample<Val>>::sample(&mut p_chal),
            v_chal.sample(),
            "challengers have same state",
        );

        // dbg!(&proof);
    }
}
