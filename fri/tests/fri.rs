use std::marker::PhantomData;

use itertools::Itertools;
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::{CanSample, DuplexChallenger};
use p3_commit::ExtensionMmcs;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{dot_product, Field, TwoAdicField};
use p3_fri::prover::prove;
use p3_fri::verifier::verify;
use p3_fri::{Codeword, FoldableLinearCodeFamily, FriConfig};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::{reverse_bits_len, reverse_slice_index_bits, SliceExt};
use rand::distributions::{Distribution, Standard};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    FieldMerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
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
impl<F: TwoAdicField> FoldableLinearCodeFamily<F> for RsCode<F> {
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
        log_blowup: 1,
        log_max_final_poly_len: 0,
        log_folding_arity: 1,
        num_queries: 10,
        proof_of_work_bits: 1,
        mmcs,
    }
}

fn do_test_fri(config: MyFriConfig, log_msg_lens: &[usize]) {
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let chal = Challenger::new(default_perm());
    let inputs = rand_codewords::<Challenge>(&mut rng, config.log_blowup, log_msg_lens);

    let mut p_chal = chal.clone();
    let proof = prove(&config, inputs.clone(), &mut p_chal, |_| ());

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

    let mut v_chal = chal.clone();
    let log_max_word_len = inputs.iter().map(|i| i.code.log_word_len()).max().unwrap();
    verify(&config, &proof, &mut v_chal, |index, &proof| {
        if proof == () {
            Ok(inputs
                .iter()
                .map(|i| {
                    i.restrict(
                        i.code.log_word_len(),
                        index >> (log_max_word_len - i.code.log_word_len()),
                    )
                })
                .collect())
        } else {
            Err(())
        }
    })
    .unwrap();

    assert_eq!(
        <Challenger as CanSample<Val>>::sample(&mut p_chal),
        v_chal.sample(),
        "challengers have same state",
    );
}

#[test]
fn test_fri() {
    do_test_fri(MyFriConfig { ..default_cfg() }, &[8]);
    do_test_fri(MyFriConfig { ..default_cfg() }, &[8, 7, 6]);

    do_test_fri(
        MyFriConfig {
            log_blowup: 2,
            ..default_cfg()
        },
        &[8, 7, 6],
    );

    do_test_fri(
        MyFriConfig {
            log_max_final_poly_len: 4,
            ..default_cfg()
        },
        &[8, 7, 6],
    );

    do_test_fri(
        MyFriConfig {
            log_blowup: 2,
            log_max_final_poly_len: 4,
            ..default_cfg()
        },
        &[8, 7, 6],
    );

    do_test_fri(
        MyFriConfig {
            log_blowup: 1,
            log_folding_arity: 2,
            log_max_final_poly_len: 4,
            ..default_cfg()
        },
        &[8, 7, 6],
    );
}
