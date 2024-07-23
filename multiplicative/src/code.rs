use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use itertools::Itertools;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_util::{reverse_bits_len, reverse_slice_index_bits, SliceExt};

use p3_field::TwoAdicField;
use p3_fri::{CodeFamily, FoldableCodeFamily, LinearCodeFamily};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsCode<F> {
    log_blowup: usize,
    log_message_len: usize,
    _phantom: PhantomData<F>,
}

impl<F> RsCode<F> {
    pub fn new(log_blowup: usize, log_message_len: usize) -> Self {
        Self {
            log_blowup,
            log_message_len,
            _phantom: PhantomData,
        }
    }
}

impl<F: TwoAdicField> CodeFamily<F> for RsCode<F> {
    fn log_blowup(&self) -> usize {
        self.log_blowup
    }
    fn log_message_len(&self) -> usize {
        self.log_message_len
    }

    fn encode(&self, message: &[F]) -> Vec<F> {
        let mut coeffs = message.to_vec();
        assert_eq!(coeffs.log_strict_len(), self.log_message_len);
        coeffs.resize(coeffs.len() << self.log_blowup, F::zero());
        let mut evals = Radix2Dit::default().dft(coeffs.to_vec());
        reverse_slice_index_bits(&mut evals);
        evals
    }
    fn decode(&self, word: &[F]) -> Vec<F> {
        let mut evals = word.to_vec();
        reverse_slice_index_bits(&mut evals);
        assert_eq!(
            evals.log_strict_len(),
            self.log_message_len + self.log_blowup
        );
        let mut coeffs = Radix2Dit::default().idft(evals);
        assert!(coeffs
            .drain((1 << self.log_message_len)..)
            .all(|x| x.is_zero()));
        coeffs
    }
}

impl<F: TwoAdicField> LinearCodeFamily<F> for RsCode<F> {}

impl<F: TwoAdicField> FoldableCodeFamily<F> for RsCode<F> {
    fn folded_code(mut self) -> Self {
        self.log_message_len -= 1;
        self
    }
    fn fold_word_at_index(&self, beta: F, pair_index: usize, (e0, e1): (F, F)) -> F {
        let subgroup_start = F::two_adic_generator(self.log_message_len + self.log_blowup).exp_u64(
            reverse_bits_len(pair_index, self.log_message_len + self.log_blowup - 1) as u64,
        );
        let mut xs = F::two_adic_generator(1)
            .shifted_powers(subgroup_start)
            .take(2)
            .collect_vec();
        reverse_slice_index_bits(&mut xs);
        // interpolate and evaluate at beta
        e0 + (beta - xs[0]) * (e1 - e0) / (xs[1] - xs[0])
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_commit::ExtensionMmcs;
    use p3_field::{extension::BinomialExtensionField, Field};
    use p3_fri::{Codeword, FriConfig};
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

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

    fn default_perm() -> Perm {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear::default(),
            &mut rng,
        )
    }

    fn default_cfg() -> MyFriConfig {
        let hash = MyHash::new(default_perm());
        let compress = MyCompress::new(default_perm());
        let mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));

        MyFriConfig {
            log_folding_arity: 1,
            log_max_final_word_len: 1,
            num_queries: 2,
            proof_of_work_bits: 1,
            mmcs,
        }
    }

    #[test]
    fn it_works() {
        let config = default_cfg();
        let chal = Challenger::new(default_perm());

        let code = RsCode::new(1, 5);

        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let msg: Vec<Challenge> = (0..(1 << code.log_message_len()))
            .map(|_| rng.gen())
            .collect();

        let cw = Codeword::full(code.clone(), code.encode(&msg));
        let cws = vec![cw];

        let mut p_chal = chal.clone();
        let (proof, p_indices) = p3_fri::prover::prove(&config, cws.clone(), &mut p_chal);

        dbg!(&proof.final_polys);
        dbg!(&p_indices);

        let mut v_chal = chal.clone();
        let query_samples: Vec<Vec<Challenge>> = p_indices
            .iter()
            .map(|&index| cws.iter().map(|cw| cw.word[index]).collect())
            .collect();
        let v_indices = p3_fri::verifier::verify(
            &config,
            &[code.clone()],
            &query_samples,
            &proof,
            &mut v_chal,
        )
        .unwrap();

        dbg!(&v_indices);

        assert_eq!(
            <Challenger as CanSample<Val>>::sample(&mut p_chal),
            v_chal.sample(),
            "challengers have same state",
        );
    }
}
